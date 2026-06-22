//! WP-1/WP-2 — the tensor-native trainable spine + differentiable merge (NAT-S2).
//!
//! The inference cores ([`crate::cores`]) implement `ZoneCore::forward(&[f32]) ->
//! CoreOutput([f32; D_OUT])`, which **drops the autodiff graph** at the array
//! boundary — fine for emitting a provenance trace, useless for training. This
//! module is the parallel *trainable* forward: every zone core is tensor-native
//! ([`TensorCore::forward_t`] returns a `Tensor`) and a loss at the output
//! backpropagates to **every zone's parameters**. That continuity (WP-1) is the
//! seam the rest of NAT-S2 builds on (WP-5 uses this as the real partitioned arm
//! of the H-01 ablation).
//!
//! Shapes mirror the model: input is sliced per zone (the partitioning), each
//! slice is read as a short token sequence `(seq, d_model)`, the core is a
//! sequence operator (attention for HP/PF/CX, a learned linear-recurrence SSM for
//! SM/CB), pooled to a fixed `d_out` summary. The summaries are composed by the
//! **differentiable merge** (WP-2, [`crate::merge_train`]): each zone also emits a
//! scalar score, `softmax(scores / τ)` weights the summaries, and a readout maps
//! the composed vector to the output. The scores are the same signal the canonical
//! `prune_and_reweight` hardens at inference — see `merge_train` for the
//! reconciliation that keeps training and the recorded decision in agreement.

use crate::seed::{seeded_linear, seeded_scalar_var};
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::{loss, Linear, Module, Optimizer, VarBuilder, VarMap};
use nat_types::{CoreType, ZoneId};

/// A trainable, tensor-native zone core. `forward_t` maps a per-zone input slice
/// `(batch, slice_w)` to a summary `(batch, d_out)` while **retaining the graph**.
pub trait TensorCore {
    fn forward_t(&self, x: &Tensor) -> Result<Tensor>;
    fn core_type(&self) -> CoreType;
}

/// Single-head self-attention zone core (HP/PF/CX). Learns Q/K/V and an output
/// projection; the slice is read as `seq = slice_w / d_model` tokens.
pub struct AttnCore {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    seq: usize,
    d_model: usize,
}

impl AttnCore {
    #[allow(clippy::too_many_arguments)]
    fn new(
        varmap: &VarMap,
        vb: &VarBuilder,
        prefix: &str,
        slice_w: usize,
        d_model: usize,
        d_out: usize,
        seed: u64,
        dev: &Device,
    ) -> Result<Self> {
        let seq = seq_len(slice_w, d_model)?;
        let wq = seeded_linear(varmap, vb, &p(prefix, "wq"), d_model, d_model, seed, dev)?;
        let wk = seeded_linear(varmap, vb, &p(prefix, "wk"), d_model, d_model, seed, dev)?;
        let wv = seeded_linear(varmap, vb, &p(prefix, "wv"), d_model, d_model, seed, dev)?;
        let wo = seeded_linear(varmap, vb, &p(prefix, "wo"), d_model, d_out, seed, dev)?;
        Ok(AttnCore {
            wq,
            wk,
            wv,
            wo,
            seq,
            d_model,
        })
    }
}

impl TensorCore for AttnCore {
    fn forward_t(&self, x: &Tensor) -> Result<Tensor> {
        let (b, _w) = x.dims2()?;
        let h = x.reshape((b, self.seq, self.d_model))?;
        let q = self.wq.forward(&h)?; // (b, seq, d)
        let k = self.wk.forward(&h)?;
        let v = self.wv.forward(&h)?;
        let scale = 1.0 / (self.d_model as f64).sqrt();
        let scores = q.matmul(&k.transpose(1, 2)?)?.affine(scale, 0.0)?; // (b, seq, seq)
        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let ctx = attn.matmul(&v)?; // (b, seq, d)
        let pooled = ctx.mean(1)?; // (b, d) — mean over the sequence
        self.wo.forward(&pooled) // (b, d_out)
    }
    fn core_type(&self) -> CoreType {
        CoreType::Attention
    }
}

/// State-space zone core (SM/CB): a learned causal linear recurrence over the
/// token sequence. The recurrence `y = D·(x·Wb)` uses a lower-triangular decay
/// matrix `D[t,k] = exp(-softplus(log_a)·(t-k))` for `k ≤ t` — the same
/// vectorized SSM-as-matmul shape as the inference core, but with a **learnable**
/// decay and projections, so gradients reach `log_a`, `Wb`, `Wc`, `Wo`.
pub struct SsmCore {
    wb: Linear,
    wc: Linear,
    wo: Linear,
    log_a: Tensor,
    tk: Tensor,   // constant (seq, seq): (t-k) on/below the diagonal, 0 above
    mask: Tensor, // constant (seq, seq): lower-triangular ones (incl. diagonal)
    seq: usize,
    d_model: usize,
}

impl SsmCore {
    #[allow(clippy::too_many_arguments)]
    fn new(
        varmap: &VarMap,
        vb: &VarBuilder,
        prefix: &str,
        slice_w: usize,
        d_model: usize,
        d_out: usize,
        seed: u64,
        dev: &Device,
    ) -> Result<Self> {
        let seq = seq_len(slice_w, d_model)?;
        let wb = seeded_linear(varmap, vb, &p(prefix, "wb"), d_model, d_model, seed, dev)?;
        let wc = seeded_linear(varmap, vb, &p(prefix, "wc"), d_model, d_model, seed, dev)?;
        let wo = seeded_linear(varmap, vb, &p(prefix, "wo"), d_model, d_out, seed, dev)?;
        // log_a init 0 → softplus(0)=ln2 → ~0.5 decay per step, a sane start.
        let log_a = seeded_scalar_var(varmap, &p(prefix, "log_a"), 0.0, dev)?;

        let mut tkv = vec![0f32; seq * seq];
        for t in 0..seq {
            for k in 0..=t {
                tkv[t * seq + k] = (t - k) as f32;
            }
        }
        let tk = Tensor::from_vec(tkv, (seq, seq), dev)?;
        let mask = Tensor::tril2(seq, DType::F32, dev)?;
        Ok(SsmCore {
            wb,
            wc,
            wo,
            log_a,
            tk,
            mask,
            seq,
            d_model,
        })
    }
}

impl TensorCore for SsmCore {
    fn forward_t(&self, x: &Tensor) -> Result<Tensor> {
        let (b, _w) = x.dims2()?;
        let h = x.reshape((b, self.seq, self.d_model))?;
        let proj = self.wb.forward(&h)?; // (b, seq, d)

        // decay_rate = -softplus(log_a) = -ln(1 + e^{log_a})  (a learnable scalar ≤ 0)
        let decay_rate = self
            .log_a
            .exp()?
            .affine(1.0, 1.0)?
            .log()?
            .affine(-1.0, 0.0)?;
        // D = exp((t-k)·decay_rate) masked to the lower triangle.
        let decay = self.tk.broadcast_mul(&decay_rate)?.exp()?.mul(&self.mask)?; // (seq, seq)
        let decay = decay.unsqueeze(0)?.broadcast_as((b, self.seq, self.seq))?;
        let y = decay.matmul(&proj)?; // (b, seq, d)
        let read = self.wc.forward(&y)?;
        let pooled = read.mean(1)?; // (b, d)
        self.wo.forward(&pooled) // (b, d_out)
    }
    fn core_type(&self) -> CoreType {
        CoreType::Ssm
    }
}

/// Configuration for a trainable zone pass.
#[derive(Debug, Clone)]
pub struct ZonePassConfig {
    /// The learned zones to include — e.g. the 3-zone subset `{HP, PF, CX}`
    /// (ADR-0008) or all of `ZoneId::LEARNED`.
    pub zones: Vec<ZoneId>,
    /// Total input width; split evenly across the zones (the partitioning).
    pub in_dim: usize,
    /// Token width each zone reads its slice as (`slice_w` must be a multiple).
    pub d_model: usize,
    /// Per-zone summary width handed to the merge.
    pub d_out: usize,
    /// Final output width.
    pub out_dim: usize,
    /// Softmax temperature for the differentiable merge (WP-2). Large → uniform;
    /// → 0 anneals toward the hard top-k decision. `1.0` is a sane default.
    pub tau: f64,
    pub seed: u64,
}

/// The trainable spine: per-zone tensor-native cores over input slices, composed
/// by the **differentiable merge** (WP-2) and a learned readout. One `VarMap`
/// holds every parameter, so a single optimizer trains the whole pass and
/// gradients reach every zone, every score head, and the readout.
///
/// Compose (WP-2): each zone emits a summary and a scalar score; the soft merge
/// `softmax(scores / τ)` weights the summaries into one composed vector, then the
/// readout maps it to `out_dim`. The scores are the same signal the canonical
/// `nat_provenance::prune_and_reweight` hardens at inference — see
/// [`crate::merge_train`] for the reconciliation.
pub struct TrainableZonePass {
    varmap: VarMap,
    cores: Vec<(ZoneId, Box<dyn TensorCore>)>,
    /// One scalar-score head per zone (`d_out → 1`), in `cores` order.
    score_heads: Vec<Linear>,
    /// Readout from the composed summary (`d_out → out_dim`).
    head: Linear,
    slice_w: usize,
    tau: f64,
    device: Device,
}

impl TrainableZonePass {
    pub fn new(cfg: &ZonePassConfig) -> Result<Self> {
        let dev = crate::device::device();
        let n = cfg.zones.len();
        assert!(n > 0, "a zone pass needs at least one zone");
        if !cfg.in_dim.is_multiple_of(n) {
            candle_core::bail!("in_dim {} not divisible by n_zones {}", cfg.in_dim, n);
        }
        let slice_w = cfg.in_dim / n;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);

        let mut cores: Vec<(ZoneId, Box<dyn TensorCore>)> = Vec::with_capacity(n);
        for &z in &cfg.zones {
            let prefix = format!("zone_{}", z.as_str());
            let core: Box<dyn TensorCore> = match z.default_core() {
                CoreType::Attention => Box::new(AttnCore::new(
                    &varmap,
                    &vb,
                    &prefix,
                    slice_w,
                    cfg.d_model,
                    cfg.d_out,
                    cfg.seed,
                    &dev,
                )?),
                CoreType::Ssm => Box::new(SsmCore::new(
                    &varmap,
                    &vb,
                    &prefix,
                    slice_w,
                    cfg.d_model,
                    cfg.d_out,
                    cfg.seed,
                    &dev,
                )?),
                CoreType::None => {
                    candle_core::bail!("zone {:?} has no learned core (MX is non-learned)", z)
                }
            };
            cores.push((z, core));
        }

        // One scalar-score head per zone (in cores order), then the readout.
        let mut score_heads = Vec::with_capacity(n);
        for (z, _) in &cores {
            let name = format!("score_{}", z.as_str());
            score_heads.push(seeded_linear(
                &varmap, &vb, &name, cfg.d_out, 1, cfg.seed, &dev,
            )?);
        }
        let head = seeded_linear(&varmap, &vb, "head", cfg.d_out, cfg.out_dim, cfg.seed, &dev)?;

        Ok(TrainableZonePass {
            varmap,
            cores,
            score_heads,
            head,
            slice_w,
            tau: cfg.tau,
            device: dev,
        })
    }

    /// The honest backend label (candle-cpu / candle-cuda) this pass runs on.
    pub fn backend(&self) -> &'static str {
        crate::device::backend_label()
    }

    /// Total trainable parameter count.
    pub fn param_count(&self) -> usize {
        self.varmap
            .all_vars()
            .iter()
            .map(|v| v.as_tensor().elem_count())
            .sum()
    }

    /// Run the zone cores and score heads. Returns `(summaries, scores)` where
    /// `summaries` is `(batch, n_zones, d_out)` and `scores` is `(batch, n_zones)`.
    fn cores_forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut summaries = Vec::with_capacity(self.cores.len());
        let mut scores = Vec::with_capacity(self.cores.len());
        for (i, (_z, core)) in self.cores.iter().enumerate() {
            let slice = x.narrow(1, i * self.slice_w, self.slice_w)?;
            let s = core.forward_t(&slice)?; // (b, d_out)
            scores.push(self.score_heads[i].forward(&s)?); // (b, 1)
            summaries.push(s);
        }
        let summary_refs: Vec<&Tensor> = summaries.iter().collect();
        let score_refs: Vec<&Tensor> = scores.iter().collect();
        let summaries = Tensor::stack(&summary_refs, 1)?; // (b, n, d_out)
        let scores = Tensor::cat(&score_refs, 1)?; // (b, n)
        Ok((summaries, scores))
    }

    /// Forward pass: slice the input per zone, run each core, compose the
    /// summaries by the differentiable soft merge (WP-2), then read out.
    /// Input `(batch, in_dim)` → `(batch, out_dim)`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (summaries, scores) = self.cores_forward(x)?;
        let weights = crate::merge_train::soft_weights(&scores, self.tau)?;
        let composed = crate::merge_train::compose(&weights, &summaries)?; // (b, d_out)
        self.head.forward(&composed)
    }

    /// The per-zone scores `(batch, n_zones)` for an input — the signal the
    /// canonical `prune_and_reweight` hardens at inference. Zones are in
    /// [`Self::zones`] order.
    pub fn zone_scores(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.cores_forward(x)?.1)
    }

    /// The zones in this pass, in the order scores/summaries are emitted.
    pub fn zones(&self) -> Vec<ZoneId> {
        self.cores.iter().map(|(z, _)| *z).collect()
    }

    /// Forward with externally-supplied per-zone activations (the learned
    /// router's output, WP-3). The merge score is `activation × confidence`, where
    /// `confidence = sigmoid(score_head(summary))` — the architecture's
    /// "activation × confidence" merge score (§6). `activations`: `(batch,
    /// n_zones)` in [`Self::zones`] order.
    pub fn forward_modulated(&self, x: &Tensor, activations: &Tensor) -> Result<Tensor> {
        let (summaries, raw_scores) = self.cores_forward(x)?;
        let confidence = candle_nn::ops::sigmoid(&raw_scores)?; // (b, n) in [0,1]
        let combined = activations.mul(&confidence)?; // activation × confidence
        let weights = crate::merge_train::soft_weights(&combined, self.tau)?;
        let composed = crate::merge_train::compose(&weights, &summaries)?;
        self.head.forward(&composed)
    }

    /// Set the soft-merge temperature (annealing toward the hard decision).
    pub fn set_tau(&mut self, tau: f64) {
        self.tau = tau;
    }

    /// The pass's parameter map (for the optimizer / checkpointing).
    pub fn varmap(&self) -> &VarMap {
        &self.varmap
    }

    /// Mutable parameter map (for loading a checkpoint).
    pub fn varmap_mut(&mut self) -> &mut VarMap {
        &mut self.varmap
    }

    /// Train to fit `(x, y)` for `steps` of AdamW, returning `(initial, final)` MSE.
    pub fn train(&mut self, x: &Tensor, y: &Tensor, steps: usize, lr: f64) -> Result<(f32, f32)> {
        let mut opt = AdamW::new(
            self.varmap.all_vars(),
            ParamsAdamW {
                lr,
                ..Default::default()
            },
        )?;
        let initial = loss::mse(&self.forward(x)?, y)?.to_scalar::<f32>()?;
        let mut final_loss = initial;
        for _ in 0..steps {
            let l = loss::mse(&self.forward(x)?, y)?;
            opt.backward_step(&l)?;
            final_loss = l.to_scalar::<f32>()?;
        }
        Ok((initial, final_loss))
    }

    /// The device this pass lives on (for building matching input tensors).
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Test/diagnostic: does a loss at the output produce a non-vanishing gradient
    /// on **every** trainable variable? This is the WP-1 acceptance — the graph is
    /// continuous from the output back to every zone's parameters.
    pub fn every_param_has_gradient(&self, x: &Tensor, y: &Tensor) -> Result<bool> {
        let l = loss::mse(&self.forward(x)?, y)?;
        let grads = l.backward()?;
        for v in self.varmap.all_vars() {
            match grads.get(v.as_tensor()) {
                Some(g) => {
                    let s = g.abs()?.sum_all()?.to_scalar::<f32>()?;
                    if !(s.is_finite() && s > 0.0) {
                        return Ok(false);
                    }
                }
                None => return Ok(false),
            }
        }
        Ok(true)
    }
}

fn p(prefix: &str, leaf: &str) -> String {
    format!("{prefix}.{leaf}")
}

fn seq_len(slice_w: usize, d_model: usize) -> Result<usize> {
    if d_model == 0 || !slice_w.is_multiple_of(d_model) {
        candle_core::bail!("slice_w {slice_w} must be a positive multiple of d_model {d_model}");
    }
    Ok(slice_w / d_model)
}

#[cfg(test)]
mod tests {
    use super::*;

    // The 3-zone {HP, PF, CX} subset (ADR-0008): all attention zones.
    fn cfg_3zone() -> ZonePassConfig {
        ZonePassConfig {
            zones: vec![ZoneId::HP, ZoneId::PF, ZoneId::CX],
            in_dim: 96, // 3 zones × slice 32; slice 32 = seq 4 × d_model 8
            d_model: 8,
            d_out: 8,
            out_dim: 4,
            tau: 1.0,
            seed: 2026,
        }
    }

    // A config that exercises an SSM zone too (SM is an SSM zone).
    fn cfg_mixed() -> ZonePassConfig {
        ZonePassConfig {
            zones: vec![ZoneId::SM, ZoneId::PF],
            in_dim: 64,
            d_model: 8,
            d_out: 8,
            out_dim: 4,
            tau: 1.0,
            seed: 7,
        }
    }

    fn synth(n: usize, in_dim: usize, out_dim: usize, dev: &Device) -> (Tensor, Tensor) {
        let xv: Vec<f32> = (0..n * in_dim)
            .map(|i| ((i % 17) as f32 - 8.0) / 8.0)
            .collect();
        let x = Tensor::from_vec(xv, (n, in_dim), dev).unwrap();
        let wv: Vec<f32> = (0..in_dim * out_dim)
            .map(|i| ((i % 7) as f32 - 3.0) / 5.0)
            .collect();
        let w = Tensor::from_vec(wv, (in_dim, out_dim), dev).unwrap();
        let y = x.matmul(&w).unwrap().tanh().unwrap();
        (x, y)
    }

    #[test]
    fn forward_shape_is_correct() {
        let cfg = cfg_3zone();
        let pass = TrainableZonePass::new(&cfg).unwrap();
        let (x, _) = synth(5, cfg.in_dim, cfg.out_dim, pass.device());
        let out = pass.forward(&x).unwrap();
        assert_eq!(out.dims2().unwrap(), (5, cfg.out_dim));
    }

    #[test]
    fn gradient_reaches_every_zone_param_attention() {
        // WP-1 acceptance: the graph is continuous from the loss to every param.
        let cfg = cfg_3zone();
        let pass = TrainableZonePass::new(&cfg).unwrap();
        let (x, y) = synth(8, cfg.in_dim, cfg.out_dim, pass.device());
        assert!(pass.every_param_has_gradient(&x, &y).unwrap());
        // Sanity: the param count is non-trivial (zones + head all present).
        assert!(pass.param_count() > 0);
    }

    #[test]
    fn gradient_reaches_every_zone_param_mixed_ssm_and_attn() {
        // The SSM core's learnable decay (log_a) and projections must get gradient.
        let cfg = cfg_mixed();
        let pass = TrainableZonePass::new(&cfg).unwrap();
        let (x, y) = synth(8, cfg.in_dim, cfg.out_dim, pass.device());
        assert!(pass.every_param_has_gradient(&x, &y).unwrap());
    }

    #[test]
    fn training_reduces_loss() {
        // An AdamW run drives the loss down — forward + autodiff + step all work
        // through the whole zone pass, not just a detached head.
        let cfg = cfg_3zone();
        let mut pass = TrainableZonePass::new(&cfg).unwrap();
        let (x, y) = synth(32, cfg.in_dim, cfg.out_dim, pass.device());
        let (initial, final_loss) = pass.train(&x, &y, 300, 0.02).unwrap();
        assert!(
            final_loss < initial * 0.5,
            "loss barely moved: {initial} -> {final_loss}"
        );
    }

    #[test]
    fn spine_scores_harden_to_the_canonical_decision() {
        // End-to-end reconciliation (ADR-0006): hardening the spine's own per-zone
        // scores reproduces what `prune_and_reweight` would record — the training
        // merge and the inference decision agree on which zones survive.
        use nat_provenance::prune_and_reweight;
        use nat_types::Q16;
        let cfg = cfg_3zone();
        let pass = TrainableZonePass::new(&cfg).unwrap();
        let zones = pass.zones();
        let (x, _) = synth(4, cfg.in_dim, cfg.out_dim, pass.device());
        let scores = pass.zone_scores(&x).unwrap(); // (batch, n_zones)
        let rows = scores.to_vec2::<f32>().unwrap();
        for thr in [0.0f32, 0.5] {
            for row in &rows {
                let scored: Vec<(nat_types::ZoneId, Q16)> = zones
                    .iter()
                    .zip(row.iter())
                    .map(|(&z, &s)| (z, Q16::from_f32(s)))
                    .collect();
                let decision = prune_and_reweight(&scored, Q16::from_f32(thr));
                let w = crate::merge_train::soft_weights(
                    &Tensor::from_vec(row.clone(), (1, row.len()), pass.device()).unwrap(),
                    cfg.tau,
                )
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
                let hardened: Vec<nat_types::ZoneId> =
                    crate::merge_train::argtopk(&w, decision.survivors.len())
                        .iter()
                        .map(|&i| zones[i])
                        .collect();
                assert_eq!(hardened, decision.survivors);
            }
        }
    }

    #[test]
    fn init_is_reproducible() {
        // Same seed → identical initial loss (deterministic seeded init), the
        // reproducibility floor for the training stack.
        let cfg = cfg_3zone();
        let a = TrainableZonePass::new(&cfg).unwrap();
        let b = TrainableZonePass::new(&cfg).unwrap();
        let (x, y) = synth(8, cfg.in_dim, cfg.out_dim, a.device());
        let la = loss::mse(&a.forward(&x).unwrap(), &y)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let lb = loss::mse(&b.forward(&x).unwrap(), &y)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(la, lb);
    }
}
