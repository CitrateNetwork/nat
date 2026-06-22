//! WP-4 — the end-to-end trainable model + training loop (NAT-S2).
//!
//! Wires the pieces from WP-1/2/3 into one model trained end to end:
//! a learned **embedding** (token ids → vectors) → the learned **router**
//! (WP-3, gating from the pooled embedding) → the **zone spine** (WP-1) →
//! the **differentiable merge** (WP-2, score = activation × confidence) → a
//! readout. One optimizer trains every parameter — embedding, router, every
//! zone, the merge score heads, the readout — so gradients reach the whole pass.
//!
//! It runs on the 3-zone {HP,PF,CX} subset (ADR-0008), is seeded-reproducible,
//! emits a [`StepContribution`] per step (the settlement seam — `reward_weight =
//! compute × data_quality`), and checkpoints to disk. The task here is a scaled
//! synthetic-but-structured one (predict a binned token-sum), enough to prove the
//! loop trains end to end; real-corpus shards (`nat-data`) are the next data thread
//! (DGX_HANDOFF §5.3).

use crate::router::LearnedRouter;
use crate::seed::{name_seed, seeded_uniform, SplitMix64};
use crate::trainable::{TrainableZonePass, ZonePassConfig};
use candle_core::{Device, Result, Tensor, Var};
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::{loss, Optimizer};
use nat_sidecar::Sidecar;
use nat_train::StepContribution;
use nat_types::{ZoneId, Q16};

/// Configuration for the end-to-end trainable model.
#[derive(Debug, Clone)]
pub struct NatTrainConfig {
    /// The learned zones (e.g. the 3-zone {HP,PF,CX} subset, ADR-0008).
    pub zones: Vec<ZoneId>,
    pub vocab: usize,
    pub seq_len: usize,
    /// Embedding width; also the router's feature width (pooled embedding).
    pub d_emb: usize,
    /// Per-zone token width inside the spine (`slice_w` must be a multiple).
    pub d_model: usize,
    /// Per-zone summary width.
    pub d_out: usize,
    /// Output width (number of classes for the task).
    pub n_classes: usize,
    /// Router hidden width.
    pub hidden: usize,
    pub tau: f64,
    pub seed: u64,
    /// Task data quality in [0,1] — the `data_quality` term of the settlement seam
    /// (a synthetic placeholder until real `nat-data` quality scores feed it).
    pub data_quality: f32,
    /// Normalized compute units per token — a placeholder proxy for real
    /// FLOP-seconds metering (settlement seam open item #2).
    pub compute_per_token: f32,
}

impl NatTrainConfig {
    /// A small 3-zone config that trains quickly on CPU and GPU.
    pub fn small_3zone() -> Self {
        NatTrainConfig {
            zones: vec![ZoneId::HP, ZoneId::PF, ZoneId::CX],
            vocab: 16,
            seq_len: 6,
            d_emb: 16, // in_dim = 6*16 = 96; slice = 32; d_model 8 → seq 4
            d_model: 8,
            d_out: 8,
            n_classes: 4,
            hidden: 16,
            tau: 1.0,
            seed: 2026,
            data_quality: 0.9,
            compute_per_token: 0.01,
        }
    }

    fn in_dim(&self) -> usize {
        self.seq_len * self.d_emb
    }
}

/// The end-to-end trainable NAT model: embedding + router + zone spine + merge.
pub struct NatTrainModel {
    emb_varmap: candle_nn::VarMap,
    emb_table: Tensor, // (vocab, d_emb), shares storage with the embedding var
    router: LearnedRouter,
    spine: TrainableZonePass,
    cfg: NatTrainConfig,
    device: Device,
}

impl NatTrainModel {
    pub fn new(cfg: &NatTrainConfig) -> Result<Self> {
        let dev = crate::device::device();

        // Embedding table as a trainable variable in its own map.
        let emb_varmap = candle_nn::VarMap::new();
        let table = seeded_uniform(
            (cfg.vocab, cfg.d_emb),
            0.1,
            name_seed(cfg.seed, "embedding"),
            &dev,
        )?;
        let var = Var::from_tensor(&table)?;
        let emb_table = var.as_tensor().clone();
        emb_varmap
            .data()
            .lock()
            .unwrap()
            .insert("embedding.weight".to_string(), var);

        let router = LearnedRouter::with_zones(
            &Sidecar::default_l0(),
            &cfg.zones,
            cfg.d_emb,
            cfg.hidden,
            cfg.seed,
        )?;

        let spine = TrainableZonePass::new(&ZonePassConfig {
            zones: cfg.zones.clone(),
            in_dim: cfg.in_dim(),
            d_model: cfg.d_model,
            d_out: cfg.d_out,
            out_dim: cfg.n_classes,
            tau: cfg.tau,
            seed: cfg.seed,
        })?;

        Ok(NatTrainModel {
            emb_varmap,
            emb_table,
            router,
            spine,
            cfg: cfg.clone(),
            device: dev,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn backend(&self) -> &'static str {
        crate::device::backend_label()
    }

    /// All trainable variables, across embedding + router + spine.
    fn all_vars(&self) -> Vec<Var> {
        let mut v = self.emb_varmap.all_vars();
        v.extend(self.router.varmap().all_vars());
        v.extend(self.spine.varmap().all_vars());
        v
    }

    /// Total trainable parameter count (embedding + router + spine) — the budget
    /// the H-01 dense baseline must match (ADR-0005).
    pub fn param_count(&self) -> usize {
        self.all_vars()
            .iter()
            .map(|v| v.as_tensor().elem_count())
            .sum()
    }

    /// Forward: token ids `(batch, seq_len)` → class logits `(batch, n_classes)`.
    /// Embedding → pooled-embedding router gate → spine slices + cores → merge
    /// (score = activation × confidence) → readout.
    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let (b, seq) = ids.dims2()?;
        let flat_ids = ids.flatten_all()?;
        let emb = self
            .emb_table
            .index_select(&flat_ids, 0)?
            .reshape((b, seq, self.cfg.d_emb))?;
        let spine_in = emb.reshape((b, self.cfg.in_dim()))?;
        let feat = emb.mean(1)?; // (b, d_emb) — pooled embedding for the router
        let acts = self.router.activations(&feat)?; // (b, n_zones)
        self.spine.forward_modulated(&spine_in, &acts)
    }

    /// Cross-entropy of the current model on `(ids, targets)` — used for held-out
    /// evaluation.
    pub fn loss_on(&self, ids: &Tensor, targets: &Tensor) -> Result<f32> {
        loss::cross_entropy(&self.forward(ids)?, targets)?.to_scalar::<f32>()
    }

    /// Train on `(ids, targets)` for `steps` of AdamW, returning a
    /// [`StepContribution`] per step (the settlement seam). The model is mutated in
    /// place (the optimizer updates every variable).
    pub fn train(
        &mut self,
        ids: &Tensor,
        targets: &Tensor,
        steps: usize,
        lr: f64,
    ) -> Result<Vec<StepContribution>> {
        let (b, seq) = ids.dims2()?;
        let tokens = (b * seq) as u64;
        let mut opt = AdamW::new(
            self.all_vars(),
            ParamsAdamW {
                lr,
                ..Default::default()
            },
        )?;
        let mut contributions = Vec::with_capacity(steps);
        for step in 0..steps {
            let l = loss::cross_entropy(&self.forward(ids)?, targets)?;
            opt.backward_step(&l)?;
            contributions.push(self.step_contribution(step, tokens));
        }
        Ok(contributions)
    }

    /// The settlement-seam contribution for one step: `reward_weight =
    /// compute_metered × data_quality`, on the Q16.16 path.
    fn step_contribution(&self, step: usize, tokens: u64) -> StepContribution {
        let compute_metered = Q16::from_f32(tokens as f32 * self.cfg.compute_per_token);
        let data_quality = Q16::from_f32(self.cfg.data_quality);
        // A deterministic per-step digest. NOT the full inference provenance trace
        // (that is emitted when the trained model runs inference); a training-step
        // commitment, reproducible from (seed, step, tokens).
        let mut h = SplitMix64::new(
            self.cfg.seed ^ (step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ tokens,
        );
        let provenance_hash = format!("{:016x}", h.next_u64());
        StepContribution {
            compute_metered,
            data_quality,
            tokens,
            provenance_hash,
        }
    }

    /// Save the model's parameters to a directory (three safetensors files).
    pub fn save(&self, dir: &std::path::Path) -> Result<()> {
        std::fs::create_dir_all(dir).map_err(candle_core::Error::wrap)?;
        self.emb_varmap.save(dir.join("embedding.safetensors"))?;
        self.router.varmap().save(dir.join("router.safetensors"))?;
        self.spine.varmap().save(dir.join("spine.safetensors"))?;
        Ok(())
    }

    /// Load parameters previously written by [`Self::save`] into this model
    /// (which must have the same config/shapes).
    pub fn load(&mut self, dir: &std::path::Path) -> Result<()> {
        self.emb_varmap.load(dir.join("embedding.safetensors"))?;
        self.router
            .varmap_mut()
            .load(dir.join("router.safetensors"))?;
        self.spine
            .varmap_mut()
            .load(dir.join("spine.safetensors"))?;
        Ok(())
    }
}

/// A scaled synthetic-but-structured task: random token sequences whose label is
/// the **binned token sum** (a monotonic, learnable target). Deterministic from
/// `seed`, so train/val splits are reproducible and disjoint by seed.
pub fn synthetic_task(
    n: usize,
    cfg: &NatTrainConfig,
    seed: u64,
    dev: &Device,
) -> Result<(Tensor, Tensor)> {
    let mut rng = SplitMix64::new(seed);
    let mut ids: Vec<u32> = Vec::with_capacity(n * cfg.seq_len);
    let mut targets: Vec<u32> = Vec::with_capacity(n);
    let max_sum = ((cfg.vocab - 1) * cfg.seq_len) as u64;
    for _ in 0..n {
        let mut sum = 0u64;
        for _ in 0..cfg.seq_len {
            let t = (rng.next_u64() % cfg.vocab as u64) as u32;
            ids.push(t);
            sum += t as u64;
        }
        // Bin the sum into [0, n_classes) — monotonic, learnable.
        let bin = (sum * cfg.n_classes as u64 / (max_sum + 1)).min(cfg.n_classes as u64 - 1);
        targets.push(bin as u32);
    }
    let ids = Tensor::from_vec(ids, (n, cfg.seq_len), dev)?;
    let targets = Tensor::from_vec(targets, (n,), dev)?;
    Ok((ids, targets))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn held_out_loss_drops_end_to_end() {
        // The whole pass (embedding → router → zones → merge → readout) trains:
        // loss on a HELD-OUT split falls. This is g3-train in miniature.
        let cfg = NatTrainConfig::small_3zone();
        let mut model = NatTrainModel::new(&cfg).unwrap();
        let (xtr, ytr) = synthetic_task(96, &cfg, 1, model.device()).unwrap();
        let (xva, yva) = synthetic_task(48, &cfg, 999, model.device()).unwrap();

        let before = model.loss_on(&xva, &yva).unwrap();
        model.train(&xtr, &ytr, 600, 0.05).unwrap();
        let after = model.loss_on(&xva, &yva).unwrap();
        assert!(
            after < before * 0.85,
            "held-out loss did not fall: {before} -> {after}"
        );
    }

    #[test]
    fn emits_step_contributions_with_reward_weight() {
        // Every step yields a StepContribution; reward_weight = compute × quality.
        let cfg = NatTrainConfig::small_3zone();
        let mut model = NatTrainModel::new(&cfg).unwrap();
        let (x, y) = synthetic_task(32, &cfg, 7, model.device()).unwrap();
        let contribs = model.train(&x, &y, 5, 0.05).unwrap();
        assert_eq!(contribs.len(), 5);
        let c = &contribs[0];
        assert_eq!(c.tokens, 32 * cfg.seq_len as u64);
        assert!(c.reward_weight() > Q16::ZERO);
        // reward_weight == compute_metered × data_quality (the seam contract).
        assert_eq!(c.reward_weight(), c.compute_metered.mul(c.data_quality));
        // Garbage data (quality 0) would earn zero — the seam's key property.
        let zeroq = StepContribution {
            data_quality: Q16::ZERO,
            ..c.clone()
        };
        assert_eq!(zeroq.reward_weight(), Q16::ZERO);
    }

    #[test]
    fn checkpoint_round_trips() {
        // Save → load into a FRESH (differently-seeded) model → identical forward.
        let cfg = NatTrainConfig::small_3zone();
        let mut a = NatTrainModel::new(&cfg).unwrap();
        let (x, y) = synthetic_task(32, &cfg, 3, a.device()).unwrap();
        a.train(&x, &y, 50, 0.05).unwrap();

        let dir = std::env::temp_dir().join(format!("nat_ckpt_{}", cfg.seed));
        a.save(&dir).unwrap();

        let mut b = NatTrainModel::new(&NatTrainConfig {
            seed: cfg.seed ^ 0xFFFF, // different init, so a match must come from load
            ..cfg.clone()
        })
        .unwrap();
        // Pre-load: outputs differ. Post-load: identical.
        let pre = b.forward(&x).unwrap().to_vec2::<f32>().unwrap();
        b.load(&dir).unwrap();
        let out_a = a.forward(&x).unwrap().to_vec2::<f32>().unwrap();
        let out_b = b.forward(&x).unwrap().to_vec2::<f32>().unwrap();
        let _ = std::fs::remove_dir_all(&dir);

        assert_ne!(pre, out_a, "fresh model should differ before load");
        assert_eq!(
            out_a, out_b,
            "loaded model must reproduce the saved forward"
        );
    }
}
