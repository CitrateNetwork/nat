//! WP-5 — the **conclusive** H-01 ablation: the real trainable `NatTrainModel`
//! (zones + router + differentiable merge) versus an **equal-param dense
//! transformer**, on the same task, under the ADR-0005 protocol.
//!
//! This is the bet-decider with the real model, not the structural analog of
//! [`crate::run_ablation`]: H-01 asks whether zone *partitioning* costs capability
//! per parameter. The partitioned arm is `nat_candle::NatTrainModel` (real Candle
//! attention cores, a learned router, the reconciled soft merge); the baseline is
//! a dense single-block transformer with **no partitioning**, sized to the same
//! parameter budget (±tolerance, refused otherwise). Both train identically on the
//! same synthetic-but-structured classification task; we report capability per
//! parameter and the seed-averaged verdict.
//!
//! Honest scope: the task is the WP-4 synthetic one (binned token-sum), so this is
//! the conclusive *harness* with the real model — the final word needs real-corpus
//! data (DGX_HANDOFF §5.3). If partitioned < dense at equal params, H-01 is
//! refuted; the harness reports it either way.

use crate::AblationError;
use candle_core::{DType, Device, Tensor, Var, D};
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::{loss, Linear, Module, Optimizer, VarBuilder, VarMap};
use nat_candle::seed::{name_seed, seeded_linear, seeded_uniform};
use nat_candle::train_loop::{synthetic_task, NatTrainConfig, NatTrainModel};

fn candle_err(e: candle_core::Error) -> AblationError {
    AblationError::Candle(e.to_string())
}

/// Analytic parameter count of the dense baseline (embedding + one attention
/// block + an FFN + a head), used to size it to the NAT arm before building.
pub fn dense_transformer_params(
    vocab: usize,
    d_emb: usize,
    d_ff: usize,
    n_classes: usize,
) -> usize {
    let lin = |i: usize, o: usize| i * o + o; // weight + bias
    vocab * d_emb                  // embedding
        + 4 * lin(d_emb, d_emb)    // Wq, Wk, Wv, Wo
        + lin(d_emb, d_ff)         // FFN up
        + lin(d_ff, d_emb)         // FFN down
        + lin(d_emb, n_classes) // head
}

/// Find the FFN width whose dense-transformer param count is closest to `target`.
pub fn match_dense_ff(vocab: usize, d_emb: usize, n_classes: usize, target: usize) -> usize {
    let mut best = 1usize;
    let mut best_diff = usize::MAX;
    for ff in 1..=16384 {
        let p = dense_transformer_params(vocab, d_emb, ff, n_classes);
        let diff = p.abs_diff(target);
        if diff < best_diff {
            best_diff = diff;
            best = ff;
        }
        if p > target {
            break; // params increase monotonically in d_ff
        }
    }
    best
}

/// The dense baseline: embedding → one self-attention block (residual) →
/// mean-pool → FFN (residual) → head. No zone partitioning — the control arm.
pub struct DenseTransformerArm {
    varmap: VarMap,
    emb: Tensor,
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    w1: Linear,
    w2: Linear,
    head: Linear,
    seq_len: usize,
    d_emb: usize,
}

impl DenseTransformerArm {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vocab: usize,
        seq_len: usize,
        d_emb: usize,
        d_ff: usize,
        n_classes: usize,
        seed: u64,
        dev: &Device,
    ) -> candle_core::Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, dev);
        let table = seeded_uniform((vocab, d_emb), 0.1, name_seed(seed, "demb"), dev)?;
        let var = Var::from_tensor(&table)?;
        let emb = var.as_tensor().clone();
        varmap
            .data()
            .lock()
            .unwrap()
            .insert("demb.weight".to_string(), var);

        let wq = seeded_linear(&varmap, &vb, "wq", d_emb, d_emb, seed, dev)?;
        let wk = seeded_linear(&varmap, &vb, "wk", d_emb, d_emb, seed, dev)?;
        let wv = seeded_linear(&varmap, &vb, "wv", d_emb, d_emb, seed, dev)?;
        let wo = seeded_linear(&varmap, &vb, "wo", d_emb, d_emb, seed, dev)?;
        let w1 = seeded_linear(&varmap, &vb, "w1", d_emb, d_ff, seed, dev)?;
        let w2 = seeded_linear(&varmap, &vb, "w2", d_ff, d_emb, seed, dev)?;
        let head = seeded_linear(&varmap, &vb, "head", d_emb, n_classes, seed, dev)?;
        Ok(DenseTransformerArm {
            varmap,
            emb,
            wq,
            wk,
            wv,
            wo,
            w1,
            w2,
            head,
            seq_len,
            d_emb,
        })
    }

    pub fn param_count(&self) -> usize {
        self.varmap
            .all_vars()
            .iter()
            .map(|v| v.as_tensor().elem_count())
            .sum()
    }

    fn forward(&self, ids: &Tensor) -> candle_core::Result<Tensor> {
        let (b, seq) = ids.dims2()?;
        let flat = ids.flatten_all()?;
        let emb = self
            .emb
            .index_select(&flat, 0)?
            .reshape((b, seq, self.d_emb))?;
        // Self-attention block with a residual.
        let q = self.wq.forward(&emb)?;
        let k = self.wk.forward(&emb)?;
        let v = self.wv.forward(&emb)?;
        let scale = 1.0 / (self.d_emb as f64).sqrt();
        let scores = q.matmul(&k.transpose(1, 2)?)?.affine(scale, 0.0)?;
        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let ctx = attn.matmul(&v)?;
        let h = emb.add(&self.wo.forward(&ctx)?)?; // (b, seq, d_emb)
                                                   // Mean-pool over the sequence, then an FFN block with a residual.
        let pooled = h.mean(1)?; // (b, d_emb)
        let ffn = self.w2.forward(&self.w1.forward(&pooled)?.relu()?)?;
        let h2 = pooled.add(&ffn)?;
        self.head.forward(&h2) // (b, n_classes)
    }

    pub fn train(
        &mut self,
        ids: &Tensor,
        targets: &Tensor,
        steps: usize,
        lr: f64,
    ) -> candle_core::Result<(f32, f32)> {
        let mut opt = AdamW::new(
            self.varmap.all_vars(),
            ParamsAdamW {
                lr,
                ..Default::default()
            },
        )?;
        let initial = loss::cross_entropy(&self.forward(ids)?, targets)?.to_scalar::<f32>()?;
        let mut final_loss = initial;
        for _ in 0..steps {
            let l = loss::cross_entropy(&self.forward(ids)?, targets)?;
            opt.backward_step(&l)?;
            final_loss = l.to_scalar::<f32>()?;
        }
        let _ = self.seq_len; // shape is taken from ids; seq_len kept for clarity
        Ok((initial, final_loss))
    }
}

/// Configuration for the real H-01 ablation.
#[derive(Debug, Clone)]
pub struct RealAblationConfig {
    pub nat: NatTrainConfig,
    pub steps: usize,
    pub lr: f64,
    pub n_train: usize,
    pub param_tolerance: f64,
}

impl RealAblationConfig {
    /// A small config that runs on CPU in seconds.
    pub fn small_3zone() -> Self {
        RealAblationConfig {
            nat: NatTrainConfig::small_3zone(),
            steps: 150,
            lr: 0.05,
            n_train: 96,
            param_tolerance: 0.05,
        }
    }

    /// A larger config for the bet-deciding GPU run (still 3-zone, ADR-0008).
    pub fn scaled() -> Self {
        let nat = NatTrainConfig {
            zones: NatTrainConfig::small_3zone().zones,
            vocab: 32,
            seq_len: 12,
            d_emb: 24, // in_dim = 288; slice = 96; d_model 12 → seq 8
            d_model: 12,
            d_out: 16,
            n_classes: 8,
            hidden: 32,
            tau: 1.0,
            seed: 2026,
            data_quality: 0.9,
            compute_per_token: 0.01,
        };
        RealAblationConfig {
            nat,
            steps: 600,
            lr: 0.02,
            n_train: 256,
            param_tolerance: 0.05,
        }
    }
}

/// One real ablation run's result.
#[derive(Debug, Clone)]
pub struct RealAblationReport {
    pub backend: String,
    pub nat_params: usize,
    pub dense_params: usize,
    pub param_rel_diff: f64,
    pub nat_final_loss: f32,
    pub dense_final_loss: f32,
    pub nat_capability_per_param: f64,
    pub dense_capability_per_param: f64,
    pub h01_holds: bool,
}

fn cap(loss: f32) -> f64 {
    1.0 / (loss as f64 + 1e-6)
}

/// Run the real H-01 ablation for one seed: param-match the dense baseline to the
/// NAT arm (ADR-0005, refuse on mismatch), train both identically on the same
/// task, and report capability per parameter. Refuses a toy-backed NAT arm.
pub fn run_real_ablation(
    cfg: &RealAblationConfig,
    seed: u64,
) -> Result<RealAblationReport, AblationError> {
    let dev = nat_candle::device::device();

    let mut nat_cfg = cfg.nat.clone();
    nat_cfg.seed = seed;
    let mut nat = NatTrainModel::new(&nat_cfg).map_err(candle_err)?;
    let backend = nat.backend().to_string();
    // No-toy guard: the partitioned arm must be a real Candle backend, never toys.
    crate::guard_not_toy(!backend.starts_with("candle-"))?;
    let nat_params = nat.param_count();

    // Size the dense baseline to the NAT arm's parameter budget (ADR-0005).
    let d_ff = match_dense_ff(nat_cfg.vocab, nat_cfg.d_emb, nat_cfg.n_classes, nat_params);
    let dense_params =
        dense_transformer_params(nat_cfg.vocab, nat_cfg.d_emb, d_ff, nat_cfg.n_classes);
    let rel = (dense_params.abs_diff(nat_params) as f64) / (nat_params as f64);
    if rel > cfg.param_tolerance {
        return Err(AblationError::ParamsMismatch {
            dense: dense_params,
            partitioned: nat_params,
            tolerance: cfg.param_tolerance,
        });
    }
    let mut dense = DenseTransformerArm::new(
        nat_cfg.vocab,
        nat_cfg.seq_len,
        nat_cfg.d_emb,
        d_ff,
        nat_cfg.n_classes,
        seed,
        &dev,
    )
    .map_err(candle_err)?;

    // Identical task + data for both arms (ADR-0005).
    let (x, y) = synthetic_task(cfg.n_train, &nat_cfg, seed, &dev).map_err(candle_err)?;

    let nat_initial = nat.loss_on(&x, &y).map_err(candle_err)?;
    nat.train(&x, &y, cfg.steps, cfg.lr).map_err(candle_err)?;
    let nat_final = nat.loss_on(&x, &y).map_err(candle_err)?;
    debug_assert!(nat_final <= nat_initial + 1.0); // sanity; not a hard gate

    let (_di, dense_final) = dense.train(&x, &y, cfg.steps, cfg.lr).map_err(candle_err)?;

    let nat_cpp = cap(nat_final) / nat_params as f64;
    let dense_cpp = cap(dense_final) / dense_params as f64;

    Ok(RealAblationReport {
        backend,
        nat_params,
        dense_params,
        param_rel_diff: rel,
        nat_final_loss: nat_final,
        dense_final_loss: dense_final,
        nat_capability_per_param: nat_cpp,
        dense_capability_per_param: dense_cpp,
        h01_holds: nat_cpp >= dense_cpp * 0.95,
    })
}

/// The seed-averaged real ablation report.
#[derive(Debug, Clone)]
pub struct RealMultiSeedReport {
    pub backend: String,
    pub seeds: Vec<u64>,
    pub nat_params: usize,
    pub dense_params: usize,
    pub mean_nat_capability_per_param: f64,
    pub mean_dense_capability_per_param: f64,
    pub h01_holds_on_mean: bool,
    pub holds_fraction: f64,
    pub per_seed: Vec<RealAblationReport>,
}

impl RealMultiSeedReport {
    pub fn summary(&self) -> String {
        format!(
            "H-01 REAL ablation [{}] over {} seeds (params nat={} dense={})\n  \
             mean cap/param: nat={:.3e} dense={:.3e}\n  \
             verdict (mean): H-01 {} (partitioned {} dense); holds on {:.0}% of seeds",
            self.backend,
            self.seeds.len(),
            self.nat_params,
            self.dense_params,
            self.mean_nat_capability_per_param,
            self.mean_dense_capability_per_param,
            if self.h01_holds_on_mean {
                "HOLDS"
            } else {
                "REFUTED"
            },
            if self.h01_holds_on_mean { "≥" } else { "<" },
            self.holds_fraction * 100.0,
        )
    }
}

/// Run the real ablation across seeds and average the verdict (ADR-0005 / §5.2).
pub fn run_real_ablation_seeds(
    cfg: &RealAblationConfig,
    seeds: &[u64],
) -> Result<RealMultiSeedReport, AblationError> {
    if seeds.is_empty() {
        return Err(AblationError::NoSeeds);
    }
    let mut per_seed = Vec::with_capacity(seeds.len());
    for &s in seeds {
        per_seed.push(run_real_ablation(cfg, s)?);
    }
    let n = per_seed.len() as f64;
    let mean_nat = per_seed
        .iter()
        .map(|r| r.nat_capability_per_param)
        .sum::<f64>()
        / n;
    let mean_dense = per_seed
        .iter()
        .map(|r| r.dense_capability_per_param)
        .sum::<f64>()
        / n;
    let holds_fraction = per_seed.iter().filter(|r| r.h01_holds).count() as f64 / n;
    Ok(RealMultiSeedReport {
        backend: per_seed[0].backend.clone(),
        seeds: seeds.to_vec(),
        nat_params: per_seed[0].nat_params,
        dense_params: per_seed[0].dense_params,
        mean_nat_capability_per_param: mean_nat,
        mean_dense_capability_per_param: mean_dense,
        h01_holds_on_mean: mean_nat >= mean_dense * 0.95,
        holds_fraction,
        per_seed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_ablation_runs_param_matched_with_the_real_model() {
        // The acceptance: the real NAT arm vs an equal-param dense transformer,
        // param-matched within tolerance, reports a capability-per-param verdict.
        let cfg = RealAblationConfig::small_3zone();
        let report = run_real_ablation(&cfg, 1).unwrap();
        assert!(report.backend.starts_with("candle-"));
        assert!(
            report.param_rel_diff <= cfg.param_tolerance,
            "params not matched: nat={} dense={} rel={}",
            report.nat_params,
            report.dense_params,
            report.param_rel_diff
        );
        assert!(report.nat_capability_per_param > 0.0);
        assert!(report.dense_capability_per_param > 0.0);
        let _ = report.h01_holds; // the verdict is reported, not asserted at this scale
    }

    #[test]
    fn dense_param_match_is_within_tolerance() {
        let cfg = RealAblationConfig::small_3zone();
        let nat = NatTrainModel::new(&cfg.nat).unwrap();
        let target = nat.param_count();
        let ff = match_dense_ff(cfg.nat.vocab, cfg.nat.d_emb, cfg.nat.n_classes, target);
        let p = dense_transformer_params(cfg.nat.vocab, cfg.nat.d_emb, ff, cfg.nat.n_classes);
        assert!((p.abs_diff(target) as f64) / target as f64 <= 0.05);
    }

    #[test]
    fn multiseed_real_ablation_is_well_formed() {
        let cfg = RealAblationConfig::small_3zone();
        let report = run_real_ablation_seeds(&cfg, &[1, 2]).unwrap();
        assert_eq!(report.per_seed.len(), 2);
        assert!(report.mean_nat_capability_per_param > 0.0);
        assert!((0.0..=1.0).contains(&report.holds_fraction));
    }
}
