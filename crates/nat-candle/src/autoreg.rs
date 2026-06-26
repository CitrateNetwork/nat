//! WP-D7 — the per-position autoregressive NAT language model (DATA-S1, toward L2).
//!
//! The earlier `NatTrainModel` predicts a *single* next byte from a fixed context
//! (it slices the input across zones and pools to one output). That is a
//! sequence-classification shape and wastes compute — one prediction per window.
//! This is the real LM shape: each zone is a **causal** sequence operator over the
//! *full* sequence, the zones are merged **per position**, and a readout produces
//! logits at **every** position. One sequence of length `L` yields `L-1`
//! next-token predictions, so the corpus is used far more efficiently.
//!
//! The zone structure is preserved: attention zones (HP/PF/CX) are causal
//! self-attention; SSM zones (SM/CB) are the causal decay-matrix recurrence (already
//! lower-triangular, hence causal by construction). The merge is the same
//! soft (activation×score) combine as WP-2, applied independently at each position.

use crate::seed::{
    name_seed, seeded_linear_dt, seeded_scalar_var_dt, seeded_uniform_dt, SplitMix64,
};
use candle_core::{backprop::GradStore, DType, Device, Result, Tensor, Var, D};
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::{loss, Linear, Module, Optimizer, VarBuilder, VarMap};
use nat_types::{CoreType, ZoneId};

/// A causal sequence operator: `(b, seq, d) -> (b, seq, d)`, attending only to the
/// present and past.
trait CausalCore {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
}

/// Causal single-head self-attention (HP/PF/CX).
struct CausalAttn {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    mask: Tensor, // (seq, seq): 0 on/below diagonal, large-negative above
    d: usize,
}

impl CausalCore for CausalAttn {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let q = self.wq.forward(x)?;
        let k = self.wk.forward(x)?;
        let v = self.wv.forward(x)?;
        let scale = 1.0 / (self.d as f64).sqrt();
        let scores = q.matmul(&k.transpose(1, 2)?)?.affine(scale, 0.0)?; // (b, seq, seq)
                                                                         // Mask-add + softmax in f32 (the mask is f32; softmax is precision-sensitive),
                                                                         // then cast the weights back to the compute dtype. No-op when already f32.
        let scores = scores.to_dtype(DType::F32)?.broadcast_add(&self.mask)?;
        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?.to_dtype(x.dtype())?;
        let ctx = attn.matmul(&v)?; // (b, seq, d)
        self.wo.forward(&ctx)
    }
}

/// Causal state-space core (SM/CB): `y = D·(x·Wb)` with a lower-triangular decay
/// matrix — causal by construction, no mask needed.
struct CausalSsm {
    wb: Linear,
    wc: Linear,
    wo: Linear,
    log_a: Tensor,
    tk: Tensor,   // (seq, seq): (t-k) on/below the diagonal, 0 above
    mask: Tensor, // (seq, seq): lower-triangular ones
}

impl CausalCore for CausalSsm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, seq, _d) = x.dims3()?;
        let proj = self.wb.forward(x)?; // (b, seq, d)
                                        // Build the decay matrix in f32 (exp/log are precision-sensitive; tk/mask are
                                        // f32), then cast to the compute dtype for the matmul. No-op when already f32.
        let decay_rate = self
            .log_a
            .to_dtype(DType::F32)?
            .exp()?
            .affine(1.0, 1.0)?
            .log()?
            .affine(-1.0, 0.0)?;
        let decay = self.tk.broadcast_mul(&decay_rate)?.exp()?.mul(&self.mask)?; // (seq, seq)
        let decay = decay
            .to_dtype(x.dtype())?
            .unsqueeze(0)?
            .broadcast_as((b, seq, seq))?;
        let y = decay.matmul(&proj)?; // (b, seq, d)
        self.wo.forward(&self.wc.forward(&y)?)
    }
}

/// Configuration for the autoregressive NAT LM.
#[derive(Debug, Clone)]
pub struct AutoregConfig {
    pub zones: Vec<ZoneId>,
    pub vocab: usize,
    pub seq_len: usize,
    /// Model width (embedding + per-zone hidden).
    pub d: usize,
    pub tau: f64,
    pub seed: u64,
}

impl AutoregConfig {
    /// A 3-zone byte-level autoregressive LM (vocab 256).
    pub fn byte_3zone() -> Self {
        AutoregConfig {
            zones: vec![ZoneId::HP, ZoneId::PF, ZoneId::CX],
            vocab: nat_data::tokenizer::BYTE_VOCAB,
            seq_len: 64,
            d: 48,
            tau: 1.0,
            seed: 2026,
        }
    }
}

/// The per-position autoregressive NAT model.
pub struct AutoregLm {
    varmap: VarMap,
    emb: Tensor, // (vocab, d)
    cores: Vec<Box<dyn CausalCore>>,
    score_heads: Vec<Linear>, // d -> 1, per zone (per-position score)
    readout: Linear,          // d -> vocab
    cfg: AutoregConfig,
    device: Device,
}

impl AutoregLm {
    /// Build with f32 weights (the default).
    pub fn new(cfg: &AutoregConfig) -> Result<Self> {
        Self::new_with_dtype(cfg, DType::F32)
    }

    /// Build with weights + activations in `dtype` (e.g. `DType::BF16` for the
    /// mixed-precision throughput path, SCALE-S1 WP-S2). The numerically-sensitive ops
    /// (attention/merge softmax, SSM decay exp/log, cross-entropy) run in f32 regardless,
    /// so `dtype == F32` is bit-identical to the original model.
    pub fn new_with_dtype(cfg: &AutoregConfig, dtype: DType) -> Result<Self> {
        let dev = crate::device::device();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &dev);
        let d = cfg.d;
        let seq = cfg.seq_len;

        // Embedding table.
        let table = seeded_uniform_dt(
            (cfg.vocab, d),
            0.1,
            name_seed(cfg.seed, "embedding"),
            dtype,
            &dev,
        )?;
        let var = candle_core::Var::from_tensor(&table)?;
        let emb = var.as_tensor().clone();
        varmap
            .data()
            .lock()
            .unwrap()
            .insert("embedding.weight".to_string(), var);

        // Constant causal masks (kept f32; added/used in f32-space ops).
        let attn_mask = causal_attn_mask(seq, &dev)?;
        let (tk, tri) = ssm_matrices(seq, &dev)?;

        let mut cores: Vec<Box<dyn CausalCore>> = Vec::with_capacity(cfg.zones.len());
        let mut score_heads = Vec::with_capacity(cfg.zones.len());
        for &z in &cfg.zones {
            let p = format!("zone_{}", z.as_str());
            let lin = |name: &str, i: usize, o: usize| {
                seeded_linear_dt(&varmap, &vb, name, i, o, cfg.seed, dtype, &dev)
            };
            let core: Box<dyn CausalCore> = match z.default_core() {
                CoreType::Attention => Box::new(CausalAttn {
                    wq: lin(&format!("{p}.wq"), d, d)?,
                    wk: lin(&format!("{p}.wk"), d, d)?,
                    wv: lin(&format!("{p}.wv"), d, d)?,
                    wo: lin(&format!("{p}.wo"), d, d)?,
                    mask: attn_mask.clone(),
                    d,
                }),
                CoreType::Ssm => Box::new(CausalSsm {
                    wb: lin(&format!("{p}.wb"), d, d)?,
                    wc: lin(&format!("{p}.wc"), d, d)?,
                    wo: lin(&format!("{p}.wo"), d, d)?,
                    log_a: seeded_scalar_var_dt(&varmap, &format!("{p}.log_a"), 0.0, dtype, &dev)?,
                    tk: tk.clone(),
                    mask: tri.clone(),
                }),
                CoreType::None => candle_core::bail!("zone {z:?} has no learned core"),
            };
            cores.push(core);
            score_heads.push(lin(&format!("score_{}", z.as_str()), d, 1)?);
        }
        let readout =
            seeded_linear_dt(&varmap, &vb, "readout", d, cfg.vocab, cfg.seed, dtype, &dev)?;

        Ok(AutoregLm {
            varmap,
            emb,
            cores,
            score_heads,
            readout,
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

    pub fn param_count(&self) -> usize {
        self.varmap
            .all_vars()
            .iter()
            .map(|v| v.as_tensor().elem_count())
            .sum()
    }

    /// Export the model to a GGUF file (lossless F32) with NAT metadata
    /// (g3-gguf / WP-1.4). The container round-trips through candle's GGUF reader.
    pub fn export_gguf(&self, path: &std::path::Path) -> Result<()> {
        let tensors: Vec<(String, Tensor)> = self
            .varmap
            .data()
            .lock()
            .unwrap()
            .iter()
            .map(|(n, v)| (n.clone(), v.as_tensor().clone()))
            .collect();
        let zones = self
            .cfg
            .zones
            .iter()
            .map(|z| z.as_str())
            .collect::<Vec<_>>()
            .join(",");
        let md = [
            ("general.architecture", crate::gguf::s("nat-autoreg")),
            ("general.name", crate::gguf::s("nat")),
            ("nat.vocab", crate::gguf::u(self.cfg.vocab)),
            ("nat.embedding_length", crate::gguf::u(self.cfg.d)),
            ("nat.context_length", crate::gguf::u(self.cfg.seq_len)),
            ("nat.zone_count", crate::gguf::u(self.cfg.zones.len())),
            ("nat.zones", crate::gguf::s(&zones)),
        ];
        crate::gguf::export(&tensors, &md, path)
    }

    /// Logits at every position: ids `(b, seq)` → `(b, seq, vocab)`.
    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let (b, seq) = ids.dims2()?;
        let h = self
            .emb
            .index_select(&ids.flatten_all()?, 0)?
            .reshape((b, seq, self.cfg.d))?;

        // Each zone runs causally over the full sequence; score per position.
        let mut zone_outs = Vec::with_capacity(self.cores.len());
        let mut scores = Vec::with_capacity(self.cores.len());
        for (i, core) in self.cores.iter().enumerate() {
            let zo = core.forward(&h)?; // (b, seq, d)
            scores.push(self.score_heads[i].forward(&zo)?); // (b, seq, 1)
            zone_outs.push(zo);
        }
        let nz = self.cores.len();
        // Per-position soft merge over zones.
        let score_refs: Vec<&Tensor> = scores.iter().collect();
        let scores = Tensor::cat(&score_refs, 2)?; // (b, seq, nz)
                                                   // Merge softmax in f32, then back to the compute dtype. No-op when already f32.
        let weights = candle_nn::ops::softmax(
            &scores
                .affine(1.0 / self.cfg.tau, 0.0)?
                .to_dtype(DType::F32)?,
            D::Minus1,
        )?
        .to_dtype(scores.dtype())?;
        let zo_refs: Vec<&Tensor> = zone_outs.iter().collect();
        let stacked = Tensor::stack(&zo_refs, 2)?; // (b, seq, nz, d)
        let w = weights.reshape((b, seq, nz, 1))?;
        let composed = stacked.broadcast_mul(&w)?.sum(2)?; // (b, seq, d)
        self.readout.forward(&composed) // (b, seq, vocab)
    }

    /// Next-token cross-entropy averaged over all positions of `ids` `(b, seq)`.
    pub fn loss_on(&self, ids: &Tensor) -> Result<f32> {
        self.loss_tensor(ids)?.to_scalar::<f32>()
    }

    /// Held-out loss evaluated in mini-batches to bound peak memory. The single-shot
    /// `loss_on` materializes a `(n, seq, vocab)` logit tensor over the whole set,
    /// which OOMs the GPU at large vocab (e.g. ~12.6 GB for 6000×64×8192). Because
    /// every sequence has the same length, the per-batch mean cross-entropy weighted
    /// by row count is exactly the full-set mean — so this returns the same number as
    /// `loss_on` would, just without the giant allocation.
    pub fn loss_on_batched(&self, ids: &Tensor, batch_size: usize) -> Result<f32> {
        let n = ids.dims2()?.0;
        if n == 0 {
            return Ok(0.0);
        }
        let bs = batch_size.clamp(1, n);
        let (mut weighted, mut rows) = (0.0f64, 0usize);
        let mut start = 0;
        while start < n {
            let len = (start + bs).min(n) - start;
            let xb = ids.narrow(0, start, len)?;
            let l = self.loss_tensor(&xb)?.to_scalar::<f32>()? as f64;
            weighted += l * len as f64;
            rows += len;
            start += len;
        }
        Ok((weighted / rows as f64) as f32)
    }

    fn loss_tensor(&self, ids: &Tensor) -> Result<Tensor> {
        let (b, seq) = ids.dims2()?;
        let logits = self.forward(ids)?; // (b, seq, vocab)
        let pred = logits
            .narrow(1, 0, seq - 1)?
            .contiguous()?
            .reshape((b * (seq - 1), self.cfg.vocab))?;
        let tgt = ids
            .narrow(1, 1, seq - 1)?
            .contiguous()?
            .reshape((b * (seq - 1),))?;
        // Cross-entropy (log-softmax + nll) in f32 for stability. No-op when f32.
        loss::cross_entropy(&pred.to_dtype(DType::F32)?, &tgt)
    }

    /// Mini-batch SGD over shuffled sequences. Targets are the inputs shifted by one
    /// (next-token), computed inside the loss. Delegates to the shared
    /// [`train_minibatched_impl`] so the NAT and dense arms train identically (ADR-0005).
    pub fn train_minibatched(
        &mut self,
        ids: &Tensor,
        epochs: usize,
        batch_size: usize,
        lr: f64,
        shuffle_seed: u64,
    ) -> Result<()> {
        let vars = self.varmap.all_vars();
        train_minibatched_impl(
            &self.device,
            vars,
            ids,
            epochs,
            batch_size,
            lr,
            shuffle_seed,
            None,
            |xb| self.loss_tensor(xb),
        )
    }

    /// Like [`Self::train_minibatched`] but writes a checkpoint to `dir`
    /// (`model.safetensors` + `meta.json`) at the end of every epoch, so a crashed
    /// multi-day run loses at most one epoch. NOTE: the AdamW optimizer state is **not**
    /// serialized — a resumed run reloads the weights and restarts the optimizer (and LR
    /// warmup). Weight-level crash-safety, not bit-identical continuation; serializing
    /// optimizer state is a follow-up (SCALE-S1 WP-S1).
    pub fn train_minibatched_checkpointed(
        &mut self,
        ids: &Tensor,
        epochs: usize,
        batch_size: usize,
        lr: f64,
        shuffle_seed: u64,
        dir: &std::path::Path,
    ) -> Result<()> {
        std::fs::create_dir_all(dir).map_err(candle_core::Error::wrap)?;
        let vars = self.varmap.all_vars();
        train_minibatched_impl(
            &self.device,
            vars,
            ids,
            epochs,
            batch_size,
            lr,
            shuffle_seed,
            Some((&self.varmap, dir)),
            |xb| self.loss_tensor(xb),
        )
    }

    /// Persist the model weights to `dir/model.safetensors`.
    pub fn save(&self, dir: &std::path::Path) -> Result<()> {
        std::fs::create_dir_all(dir).map_err(candle_core::Error::wrap)?;
        self.varmap.save(dir.join("model.safetensors"))
    }

    /// Load weights previously written by [`Self::save`] (or a checkpoint) into this
    /// model, which must have the same config/shapes.
    pub fn load(&mut self, dir: &std::path::Path) -> Result<()> {
        self.varmap.load(dir.join("model.safetensors"))
    }
}

/// A per-position **dense** autoregressive LM — the H-01 baseline for `AutoregLm`.
///
/// Same shape as `AutoregLm` (causal, predicts at every position, identical
/// embedding + readout), but the cores are NOT zone-partitioned: a single causal
/// self-attention block + a dense FFN, each with a residual — a standard
/// Transformer block. The FFN width (`d_ff`) is the knob the ablation tunes to
/// param-match the NAT arm (ADR-0005), so any held-out difference is attributable
/// to zone partitioning, not parameter count. Embedding/readout are bit-identical
/// in structure to the NAT arm (same vocab, same `d`), so the comparison isolates
/// the cores.
pub struct AutoregDenseLm {
    varmap: VarMap,
    emb: Tensor,
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    w1: Linear,
    w2: Linear,
    readout: Linear,
    mask: Tensor,
    vocab: usize,
    d: usize,
    device: Device,
}

impl AutoregDenseLm {
    /// Build with f32 weights (the default).
    pub fn new(vocab: usize, seq_len: usize, d: usize, d_ff: usize, seed: u64) -> Result<Self> {
        Self::new_with_dtype(vocab, seq_len, d, d_ff, seed, DType::F32)
    }

    /// Build with weights + activations in `dtype` (the H-01 ablation's dense arm must
    /// match the NAT arm's dtype — ADR-0005). f32 is bit-identical to the original.
    pub fn new_with_dtype(
        vocab: usize,
        seq_len: usize,
        d: usize,
        d_ff: usize,
        seed: u64,
        dtype: DType,
    ) -> Result<Self> {
        let dev = crate::device::device();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &dev);

        let table = seeded_uniform_dt((vocab, d), 0.1, name_seed(seed, "embedding"), dtype, &dev)?;
        let var = candle_core::Var::from_tensor(&table)?;
        let emb = var.as_tensor().clone();
        varmap
            .data()
            .lock()
            .unwrap()
            .insert("embedding.weight".to_string(), var);

        let lin = |name: &str, i: usize, o: usize| {
            seeded_linear_dt(&varmap, &vb, name, i, o, seed, dtype, &dev)
        };
        let wq = lin("wq", d, d)?;
        let wk = lin("wk", d, d)?;
        let wv = lin("wv", d, d)?;
        let wo = lin("wo", d, d)?;
        let w1 = lin("w1", d, d_ff)?;
        let w2 = lin("w2", d_ff, d)?;
        let readout = lin("readout", d, vocab)?;
        let mask = causal_attn_mask(seq_len, &dev)?;

        Ok(AutoregDenseLm {
            varmap,
            emb,
            wq,
            wk,
            wv,
            wo,
            w1,
            w2,
            readout,
            mask,
            vocab,
            d,
            device: dev,
        })
    }

    pub fn param_count(&self) -> usize {
        self.varmap
            .all_vars()
            .iter()
            .map(|v| v.as_tensor().elem_count())
            .sum()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Logits at every position: ids `(b, seq)` → `(b, seq, vocab)`.
    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let (b, seq) = ids.dims2()?;
        let emb = self
            .emb
            .index_select(&ids.flatten_all()?, 0)?
            .reshape((b, seq, self.d))?;
        // Causal self-attention block, residual.
        let q = self.wq.forward(&emb)?;
        let k = self.wk.forward(&emb)?;
        let v = self.wv.forward(&emb)?;
        let scale = 1.0 / (self.d as f64).sqrt();
        let scores = q.matmul(&k.transpose(1, 2)?)?.affine(scale, 0.0)?;
        // Mask-add + softmax in f32, then back to the compute dtype. No-op when f32.
        let scores = scores.to_dtype(DType::F32)?.broadcast_add(&self.mask)?;
        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?.to_dtype(emb.dtype())?;
        let ctx = attn.matmul(&v)?;
        let h = emb.add(&self.wo.forward(&ctx)?)?; // (b, seq, d)
                                                   // FFN block, residual.
        let ffn = self.w2.forward(&self.w1.forward(&h)?.relu()?)?;
        let h2 = h.add(&ffn)?;
        self.readout.forward(&h2) // (b, seq, vocab)
    }

    pub fn loss_on_batched(&self, ids: &Tensor, batch_size: usize) -> Result<f32> {
        let n = ids.dims2()?.0;
        if n == 0 {
            return Ok(0.0);
        }
        let bs = batch_size.clamp(1, n);
        let (mut weighted, mut rows) = (0.0f64, 0usize);
        let mut start = 0;
        while start < n {
            let len = (start + bs).min(n) - start;
            let xb = ids.narrow(0, start, len)?;
            let l = self.loss_tensor(&xb)?.to_scalar::<f32>()? as f64;
            weighted += l * len as f64;
            rows += len;
            start += len;
        }
        Ok((weighted / rows as f64) as f32)
    }

    fn loss_tensor(&self, ids: &Tensor) -> Result<Tensor> {
        let (b, seq) = ids.dims2()?;
        let logits = self.forward(ids)?;
        let pred = logits
            .narrow(1, 0, seq - 1)?
            .contiguous()?
            .reshape((b * (seq - 1), self.vocab))?;
        let tgt = ids
            .narrow(1, 1, seq - 1)?
            .contiguous()?
            .reshape((b * (seq - 1),))?;
        // Cross-entropy (log-softmax + nll) in f32 for stability. No-op when f32.
        loss::cross_entropy(&pred.to_dtype(DType::F32)?, &tgt)
    }

    /// Same mini-batch protocol as `AutoregLm::train_minibatched` (ADR-0005: both
    /// arms see identical batches in identical order for a given seed) — they share the
    /// exact same [`train_minibatched_impl`].
    pub fn train_minibatched(
        &mut self,
        ids: &Tensor,
        epochs: usize,
        batch_size: usize,
        lr: f64,
        shuffle_seed: u64,
    ) -> Result<()> {
        let vars = self.varmap.all_vars();
        train_minibatched_impl(
            &self.device,
            vars,
            ids,
            epochs,
            batch_size,
            lr,
            shuffle_seed,
            None,
            |xb| self.loss_tensor(xb),
        )
    }

    /// Checkpointing counterpart to [`Self::train_minibatched`] — see
    /// [`AutoregLm::train_minibatched_checkpointed`] for the (weight-level) semantics.
    pub fn train_minibatched_checkpointed(
        &mut self,
        ids: &Tensor,
        epochs: usize,
        batch_size: usize,
        lr: f64,
        shuffle_seed: u64,
        dir: &std::path::Path,
    ) -> Result<()> {
        std::fs::create_dir_all(dir).map_err(candle_core::Error::wrap)?;
        let vars = self.varmap.all_vars();
        train_minibatched_impl(
            &self.device,
            vars,
            ids,
            epochs,
            batch_size,
            lr,
            shuffle_seed,
            Some((&self.varmap, dir)),
            |xb| self.loss_tensor(xb),
        )
    }

    /// Persist the model weights to `dir/model.safetensors`.
    pub fn save(&self, dir: &std::path::Path) -> Result<()> {
        std::fs::create_dir_all(dir).map_err(candle_core::Error::wrap)?;
        self.varmap.save(dir.join("model.safetensors"))
    }

    /// Load weights previously written by [`Self::save`] (or a checkpoint) into this
    /// model, which must have the same config/shapes.
    pub fn load(&mut self, dir: &std::path::Path) -> Result<()> {
        self.varmap.load(dir.join("model.safetensors"))
    }
}

/// The shared mini-batch AdamW training loop for **both** H-01 arms. Keeping it in one
/// place is what makes ADR-0005's "identical training" literally true rather than a
/// promise kept by copy-paste. Two stabilizers beyond plain AdamW, both deterministic
/// and applied identically to NAT and dense so the comparison stays clean:
///
/// - **Linear LR warmup** over the first 5% of steps. Adam's second-moment estimate is
///   noisy in the first handful of steps, so a wide model (large `d`) can take a few
///   enormous steps and diverge at full `lr` — the failure mode that blew up one seed of
///   the 8M rung (`lr=0.003`, `d=476`). Warmup eases in the step size.
/// - **Global grad-norm clipping** at 1.0 (standard LM hygiene) — a second guard on the
///   same failure mode: if the total gradient norm spikes, the step is rescaled down.
#[allow(clippy::too_many_arguments)]
fn train_minibatched_impl(
    device: &Device,
    vars: Vec<Var>,
    ids: &Tensor,
    epochs: usize,
    batch_size: usize,
    lr: f64,
    shuffle_seed: u64,
    checkpoint: Option<(&VarMap, &std::path::Path)>,
    mut loss_of: impl FnMut(&Tensor) -> Result<Tensor>,
) -> Result<()> {
    let n = ids.dims2()?.0;
    let bs = batch_size.clamp(1, n.max(1));
    let total_steps = (epochs * n.div_ceil(bs)).max(1);
    let warmup = (total_steps / 20).max(1);
    const MAX_GRAD_NORM: f64 = 1.0;

    let mut opt = AdamW::new(
        vars.clone(),
        ParamsAdamW {
            lr,
            ..Default::default()
        },
    )?;
    let mut perm: Vec<u32> = (0..n as u32).collect();
    let mut step = 0usize;
    for epoch in 0..epochs {
        shuffle(
            &mut perm,
            shuffle_seed ^ (epoch as u64).wrapping_mul(0x9E37_79B9),
        );
        let mut start = 0;
        while start < n {
            let end = (start + bs).min(n);
            let idx = Tensor::from_vec(perm[start..end].to_vec(), (end - start,), device)?;
            let xb = ids.index_select(&idx, 0)?;

            // Linear warmup, then hold flat at `lr`.
            let cur_lr = if step < warmup {
                lr * (step as f64 + 1.0) / warmup as f64
            } else {
                lr
            };
            opt.set_learning_rate(cur_lr);

            let l = loss_of(&xb)?;
            let mut grads = l.backward()?;
            clip_grad_norm(&mut grads, &vars, MAX_GRAD_NORM)?;
            opt.step(&grads)?;

            start = end;
            step += 1;
        }
        // End-of-epoch checkpoint: a crashed multi-day run loses at most one epoch.
        if let Some((varmap, dir)) = checkpoint {
            varmap.save(dir.join("model.safetensors"))?;
            std::fs::write(
                dir.join("meta.json"),
                format!("{{\"epochs_completed\":{}}}\n", epoch + 1),
            )
            .map_err(candle_core::Error::wrap)?;
        }
    }
    Ok(())
}

/// Global L2 grad-norm clip: if the total gradient norm across `vars` exceeds
/// `max_norm`, scale every gradient by `max_norm / norm`; a no-op when already under.
/// Deterministic, so it preserves seed-reproducibility and the ADR-0005 equal-training
/// guarantee.
fn clip_grad_norm(grads: &mut GradStore, vars: &[Var], max_norm: f64) -> Result<()> {
    let mut sq = 0f64;
    for v in vars {
        if let Some(g) = grads.get(v.as_tensor()) {
            // Accumulate the norm in f32 even when the grad is bf16/f16.
            sq += g
                .sqr()?
                .sum_all()?
                .to_dtype(DType::F32)?
                .to_scalar::<f32>()? as f64;
        }
    }
    let norm = sq.sqrt();
    if norm <= max_norm {
        return Ok(());
    }
    let scale = max_norm / (norm + 1e-6);
    for v in vars {
        // Read the grad, drop the borrow, then write the scaled grad back.
        let scaled = match grads.get(v.as_tensor()) {
            Some(g) => g.affine(scale, 0.0)?,
            None => continue,
        };
        grads.insert(v.as_tensor(), scaled);
    }
    Ok(())
}

fn shuffle(perm: &mut [u32], seed: u64) {
    let mut rng = SplitMix64::new(seed);
    for i in (1..perm.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        perm.swap(i, j);
    }
}

/// `(seq, seq)` additive causal mask: 0 on/below the diagonal, large-negative above.
fn causal_attn_mask(seq: usize, dev: &Device) -> Result<Tensor> {
    let mut m = vec![0f32; seq * seq];
    for t in 0..seq {
        for k in (t + 1)..seq {
            m[t * seq + k] = -1e9;
        }
    }
    Tensor::from_vec(m, (seq, seq), dev)
}

/// The SSM constant matrices: `(t-k)` on/below the diagonal, and the lower-triangular
/// ones mask.
fn ssm_matrices(seq: usize, dev: &Device) -> Result<(Tensor, Tensor)> {
    let mut tkv = vec![0f32; seq * seq];
    for t in 0..seq {
        for k in 0..=t {
            tkv[t * seq + k] = (t - k) as f32;
        }
    }
    let tk = Tensor::from_vec(tkv, (seq, seq), dev)?;
    let tri = Tensor::tril2(seq, DType::F32, dev)?;
    Ok((tk, tri))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_shape_and_param_count() {
        let cfg = AutoregConfig {
            seq_len: 16,
            d: 24,
            ..AutoregConfig::byte_3zone()
        };
        let m = AutoregLm::new(&cfg).unwrap();
        let ids = Tensor::from_vec(
            (0..(2 * cfg.seq_len) as u32)
                .map(|i| i % 256)
                .collect::<Vec<_>>(),
            (2, cfg.seq_len),
            m.device(),
        )
        .unwrap();
        let out = m.forward(&ids).unwrap();
        assert_eq!(out.dims3().unwrap(), (2, cfg.seq_len, cfg.vocab));
        assert!(m.param_count() > 0);
    }

    #[test]
    fn batched_eval_matches_single_shot() {
        // loss_on_batched must equal loss_on regardless of batch size (sequences are
        // equal-length, so the row-weighted mean is exact). Guards the OOM fix.
        let cfg = AutoregConfig {
            seq_len: 16,
            d: 24,
            ..AutoregConfig::byte_3zone()
        };
        let m = AutoregLm::new(&cfg).unwrap();
        let n = 7u32; // deliberately not a multiple of any batch size below
        let ids = Tensor::from_vec(
            (0..(n * cfg.seq_len as u32))
                .map(|i| (i * 31 + 7) % 256)
                .collect::<Vec<_>>(),
            (n as usize, cfg.seq_len),
            m.device(),
        )
        .unwrap();
        let full = m.loss_on(&ids).unwrap();
        for bs in [1usize, 2, 3, 7, 100] {
            let batched = m.loss_on_batched(&ids, bs).unwrap();
            assert!(
                (full - batched).abs() < 1e-4,
                "bs={bs}: {batched} != {full}"
            );
        }
    }

    fn small_cfg() -> AutoregConfig {
        AutoregConfig {
            seq_len: 16,
            d: 24,
            ..AutoregConfig::byte_3zone()
        }
    }

    fn synthetic_ids(cfg: &AutoregConfig, dev: &Device) -> Tensor {
        let n = 12u32;
        Tensor::from_vec(
            (0..(n * cfg.seq_len as u32))
                .map(|i| (i * 17 + 3) % 256)
                .collect::<Vec<_>>(),
            (n as usize, cfg.seq_len),
            dev,
        )
        .unwrap()
    }

    #[test]
    fn checkpoint_save_load_round_trips() {
        // Save → load into a FRESH, differently-seeded model → identical loss. A match
        // can only come from load (the seeds differ), so this proves weights round-trip.
        let cfg = small_cfg();
        let mut a = AutoregLm::new(&cfg).unwrap();
        let ids = synthetic_ids(&cfg, a.device());
        a.train_minibatched(&ids, 2, 4, 0.01, 13).unwrap();
        let loss_a = a.loss_on_batched(&ids, 4).unwrap();

        let dir = std::env::temp_dir().join(format!("nat_autoreg_ckpt_{}", cfg.seed));
        a.save(&dir).unwrap();

        let mut b = AutoregLm::new(&AutoregConfig {
            seed: cfg.seed ^ 0xFFFF, // different init: a match must come from load
            ..cfg.clone()
        })
        .unwrap();
        let pre = b.loss_on_batched(&ids, 4).unwrap();
        b.load(&dir).unwrap();
        let post = b.loss_on_batched(&ids, 4).unwrap();
        let _ = std::fs::remove_dir_all(&dir);

        assert!(
            (pre - loss_a).abs() > 1e-6,
            "fresh model should differ before load"
        );
        assert!(
            (post - loss_a).abs() < 1e-6,
            "loaded model must reproduce the saved loss: {post} vs {loss_a}"
        );
    }

    #[test]
    fn checkpointed_training_writes_loadable_checkpoint() {
        // train_minibatched_checkpointed writes model.safetensors + meta.json each epoch;
        // the checkpoint loads into a fresh model and reproduces the trained loss.
        let cfg = small_cfg();
        let mut a = AutoregLm::new(&cfg).unwrap();
        let ids = synthetic_ids(&cfg, a.device());
        let dir = std::env::temp_dir().join(format!("nat_autoreg_train_ckpt_{}", cfg.seed));
        a.train_minibatched_checkpointed(&ids, 2, 4, 0.01, 13, &dir)
            .unwrap();
        let loss_a = a.loss_on_batched(&ids, 4).unwrap();

        let meta = std::fs::read_to_string(dir.join("meta.json")).unwrap();
        assert!(
            meta.contains("\"epochs_completed\":2"),
            "meta records epochs: {meta}"
        );

        let mut b = AutoregLm::new(&AutoregConfig {
            seed: cfg.seed ^ 0xFFFF,
            ..cfg.clone()
        })
        .unwrap();
        b.load(&dir).unwrap();
        let post = b.loss_on_batched(&ids, 4).unwrap();
        let _ = std::fs::remove_dir_all(&dir);

        assert!(
            (post - loss_a).abs() < 1e-6,
            "checkpoint must reproduce the trained loss: {post} vs {loss_a}"
        );
    }

    fn five_zone_small() -> AutoregConfig {
        AutoregConfig {
            zones: ZoneId::LEARNED.to_vec(),
            seq_len: 16,
            d: 24,
            ..AutoregConfig::byte_3zone()
        }
    }

    #[test]
    fn bf16_model_constructs_and_matches_param_count() {
        // The bf16 init plumbing (dtype-aware seeded weights) must build a 5-zone model
        // — SM/CB SSM + HP/PF/CX attention — with the SAME parameter count as f32 (same
        // shapes, different storage precision). bf16 *compute* (matmul) is GPU-only in
        // this candle build, so the forward/train comparison is the cuda-gated test below.
        let cfg = five_zone_small();
        let f = AutoregLm::new_with_dtype(&cfg, DType::F32).unwrap();
        let b = AutoregLm::new_with_dtype(&cfg, DType::BF16).unwrap();
        assert!(b.param_count() > 0);
        assert_eq!(f.param_count(), b.param_count());
    }

    // candle's CPU backend has no bf16 matmul ("unsupported dtype BF16 for op matmul"),
    // so the bf16 forward/train path only runs on CUDA. This validates that the
    // f32-protected softmax/SSM-decay/cross-entropy keep bf16 training stable and in the
    // f32 ballpark. Runs under `scripts/dgx-gpu.sh test`.
    #[cfg(feature = "cuda")]
    #[test]
    fn bf16_trains_and_tracks_f32() {
        let cfg = five_zone_small();
        let ids = synthetic_ids(&cfg, AutoregLm::new(&cfg).unwrap().device());

        let mut f = AutoregLm::new_with_dtype(&cfg, DType::F32).unwrap();
        f.train_minibatched(&ids, 2, 4, 0.01, 5).unwrap();
        let lf = f.loss_on_batched(&ids, 4).unwrap();

        let mut b = AutoregLm::new_with_dtype(&cfg, DType::BF16).unwrap();
        b.train_minibatched(&ids, 2, 4, 0.01, 5).unwrap();
        let lb = b.loss_on_batched(&ids, 4).unwrap();

        let uniform = (cfg.vocab as f32).ln();
        assert!(lb.is_finite() && lb > 0.0, "bf16 loss not finite: {lb}");
        assert!(
            lb < uniform * 1.1,
            "bf16 loss {lb} not ~below uniform {uniform}"
        );
        assert!(
            (lb - lf).abs() < 0.5,
            "bf16 loss {lb} should track f32 {lf} within 0.5"
        );
    }

    #[test]
    fn next_token_loss_drops_below_uniform() {
        // A short, fast run on real seed text: per-position next-token loss falls
        // below the uniform-byte baseline (ln 256). Proves the autoregressive
        // objective + causal cores train.
        let cfg = AutoregConfig {
            seq_len: 32,
            d: 32,
            ..AutoregConfig::byte_3zone()
        };
        let mut m = AutoregLm::new(&cfg).unwrap();
        let out = nat_data::run_pipeline(
            nat_data::seed::seed_corpus(),
            &nat_data::PipelineConfig::default(),
        );
        let ids =
            crate::corpus::sequence_windows(&out.shards, cfg.seq_len, 200, m.device()).unwrap();
        let before = m.loss_on(&ids).unwrap();
        m.train_minibatched(&ids, 5, 32, 0.003, 7).unwrap();
        let after = m.loss_on(&ids).unwrap();
        assert!(after < before, "did not learn: {before} -> {after}");
        assert!(
            after < (cfg.vocab as f32).ln(),
            "no better than uniform: {after}"
        );
    }

    #[test]
    fn dense_arm_per_position_shape_and_learns() {
        // The H-01 dense baseline: per-position logits (b, seq, vocab), and the
        // next-token loss drops below uniform — same shape + objective as AutoregLm.
        let (vocab, seq, d, d_ff) = (256usize, 32usize, 32usize, 48usize);
        let mut m = AutoregDenseLm::new(vocab, seq, d, d_ff, 7).unwrap();
        let out = nat_data::run_pipeline(
            nat_data::seed::seed_corpus(),
            &nat_data::PipelineConfig::default(),
        );
        let ids = crate::corpus::sequence_windows(&out.shards, seq, 200, m.device()).unwrap();
        let logits = m.forward(&ids).unwrap();
        let (b, s, v) = logits.dims3().unwrap();
        assert_eq!((s, v), (seq, vocab));
        assert_eq!(b, ids.dims2().unwrap().0);
        let before = m.loss_on_batched(&ids, 64).unwrap();
        m.train_minibatched(&ids, 5, 32, 0.003, 7).unwrap();
        let after = m.loss_on_batched(&ids, 64).unwrap();
        assert!(
            after < before,
            "dense arm did not learn: {before} -> {after}"
        );
        assert!(
            after < (vocab as f32).ln(),
            "no better than uniform: {after}"
        );
    }

    #[test]
    fn gguf_export_round_trips() {
        // g3-gguf: export to GGUF, read back via candle's GGUF reader; every weight
        // is present and its values match (lossless F32).
        let cfg = AutoregConfig {
            seq_len: 16,
            d: 24,
            ..AutoregConfig::byte_3zone()
        };
        let m = AutoregLm::new(&cfg).unwrap();
        let path = std::env::temp_dir().join("nat_autoreg.gguf");
        m.export_gguf(&path).unwrap();

        let names = crate::gguf::tensor_names(&path).unwrap();
        let want: std::collections::BTreeSet<String> =
            m.varmap.data().lock().unwrap().keys().cloned().collect();
        let got: std::collections::BTreeSet<String> = names.into_iter().collect();
        assert_eq!(got, want, "GGUF tensor set != model varmap");

        // A sampled weight round-trips exactly.
        let orig = m
            .varmap
            .data()
            .lock()
            .unwrap()
            .get("embedding.weight")
            .unwrap()
            .as_tensor()
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let back = crate::gguf::read_tensor(&path, "embedding.weight")
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(orig, back, "embedding did not round-trip");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn five_zone_autoreg_constructs_and_runs() {
        // The 5-zone config exercises the causal SSM cores (SM/CB) too.
        let cfg = AutoregConfig {
            zones: ZoneId::LEARNED.to_vec(),
            seq_len: 16,
            d: 24,
            ..AutoregConfig::byte_3zone()
        };
        let m = AutoregLm::new(&cfg).unwrap();
        let ids =
            Tensor::from_vec(vec![1u32; 2 * cfg.seq_len], (2, cfg.seq_len), m.device()).unwrap();
        assert_eq!(m.forward(&ids).unwrap().dims3().unwrap().2, cfg.vocab);
    }
}
