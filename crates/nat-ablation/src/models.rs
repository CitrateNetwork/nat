//! The two arms of the H-01 ablation, as small trainable Candle models.
//!
//! - [`PartitionedArm`] — the NAT-shaped arm: the first layer is split into
//!   `n_zones` independent blocks (each zone projects the input on its own), then
//!   a head merges the concatenated zone outputs. This is the structural analog of
//!   zone partitioning.
//! - [`DenseArm`] — the baseline: a fully-connected trunk of the same depth.
//!
//! At L1/DGX these scale up (more zones, real widths, real data) and the
//! partitioned arm becomes the full `NatModel` with Candle cores. The harness
//! does not change — only the models under it do.

use candle_core::{DType, Device, Result, Tensor, Var};
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::{linear, loss, Linear, Module, Optimizer, VarBuilder, VarMap};

/// A tiny deterministic PRNG (SplitMix64) for reproducible weight init. candle's
/// own init draws from the device RNG, which the CPU backend cannot seed
/// (`set_seed` errors), so we seed the weights ourselves — giving bit-identical
/// init on both CPU and GPU and honouring the reproducibility floor (PLANSET/01).
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// A float in `[lo, hi)`, deterministic from the seed stream.
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        let u = (self.next_u64() >> 11) as f32 / (1u64 << 53) as f32; // [0,1)
        lo + (hi - lo) * u
    }
}

/// FNV-1a of a layer name, so each layer gets a distinct sub-seed (no init
/// symmetry across layers) while staying a pure function of `(seed, name)`.
fn name_seed(seed: u64, name: &str) -> u64 {
    let mut h = 0xCBF2_9CE4_8422_2325u64 ^ seed;
    for b in name.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

/// A `Linear` whose weights are deterministically seeded. We pre-insert the
/// variables into the `VarMap` under the names `linear` looks up (`weight`,
/// `bias`); `VarMap::get` then reuses them, so the optimizer trains the same
/// tensors while init stays reproducible. Uniform `[-k, k]`, `k = 1/sqrt(in_dim)`
/// — the standard fan-in scale.
fn seeded_linear(
    varmap: &VarMap,
    vb: &VarBuilder,
    prefix: &str,
    in_dim: usize,
    out_dim: usize,
    seed: u64,
    dev: &Device,
) -> Result<Linear> {
    let k = 1.0 / (in_dim as f32).sqrt();
    let mut rng = SplitMix64::new(name_seed(seed, prefix));
    let wv: Vec<f32> = (0..out_dim * in_dim).map(|_| rng.uniform(-k, k)).collect();
    let bv: Vec<f32> = (0..out_dim).map(|_| rng.uniform(-k, k)).collect();
    let w = Var::from_tensor(&Tensor::from_vec(wv, (out_dim, in_dim), dev)?)?;
    let b = Var::from_tensor(&Tensor::from_vec(bv, (out_dim,), dev)?)?;
    {
        let mut data = varmap.data().lock().unwrap();
        data.insert(format!("{prefix}.weight"), w);
        data.insert(format!("{prefix}.bias"), b);
    }
    // Reuses the vars just inserted (VarMap::get returns the existing entry).
    linear(in_dim, out_dim, vb.pp(prefix))
}

/// A fixed synthetic regression task both arms train on (same data, same target).
pub struct TrainData {
    pub x: Tensor,
    pub y: Tensor,
}

/// Deterministic synthetic data: `y = tanh(x · W)`, a bounded smooth map a small
/// MLP can fit, so the comparison reflects structure, not luck.
///
/// `seed` deterministically shifts both the inputs and the target map, so each
/// seed is a *different* task drawn from the same family. Averaging the verdict
/// across seeds (ADR-0005 / §5.2 step 4) then measures structure, not one lucky
/// draw — while staying fully reproducible (same seed → same task, byte for byte).
pub fn synthetic_data(
    n: usize,
    in_dim: usize,
    out_dim: usize,
    seed: u64,
    dev: &Device,
) -> Result<TrainData> {
    let s = (seed % 19) as usize; // small deterministic phase offset per seed
    let xv: Vec<f32> = (0..n * in_dim)
        .map(|i| (((i + s) % 17) as f32 - 8.0) / 8.0)
        .collect();
    let x = Tensor::from_vec(xv, (n, in_dim), dev)?;
    // Fixed affine-plus-mild-nonlinear target, phase-shifted by the seed.
    let wv: Vec<f32> = (0..in_dim * out_dim)
        .map(|i| (((i + s) % 7) as f32 - 3.0) / 5.0)
        .collect();
    let w = Tensor::from_vec(wv, (in_dim, out_dim), dev)?;
    let y = x.matmul(&w)?.tanh()?; // bounded, learnable
    Ok(TrainData { x, y })
}

/// What an ablation arm must provide: a parameter count, a name, and a train step.
pub trait AblationArm {
    fn param_count(&self) -> usize;
    fn name(&self) -> &str;
    /// Train on the data for `steps`, returning `(initial_loss, final_loss)`.
    fn train(&mut self, data: &TrainData, steps: usize, lr: f64) -> Result<(f32, f32)>;
}

fn count_params(varmap: &VarMap) -> usize {
    varmap
        .all_vars()
        .iter()
        .map(|v| v.as_tensor().elem_count())
        .sum()
}

/// Analytic parameter count for the partitioned arm — used to size it to match
/// the dense arm before any tensors are built (ADR-0005 equal-params).
pub fn partitioned_params(
    in_dim: usize,
    n_zones: usize,
    zone_hidden: usize,
    out_dim: usize,
) -> usize {
    let merged = n_zones * zone_hidden;
    n_zones * (in_dim * zone_hidden + zone_hidden) + (merged * out_dim + out_dim)
}

/// Analytic parameter count for the dense arm.
pub fn dense_params(in_dim: usize, hidden: usize, out_dim: usize) -> usize {
    in_dim * hidden + hidden + hidden * out_dim + out_dim
}

/// The NAT-shaped arm: `n_zones` independent input projections → concat → head.
pub struct PartitionedArm {
    varmap: VarMap,
    zones: Vec<Linear>,
    head: Linear,
    params: usize,
}

impl PartitionedArm {
    pub fn new(
        in_dim: usize,
        n_zones: usize,
        zone_hidden: usize,
        out_dim: usize,
        seed: u64,
        dev: &Device,
    ) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, dev);
        let mut zones = Vec::with_capacity(n_zones);
        for i in 0..n_zones {
            let name = format!("zone{i}");
            zones.push(seeded_linear(
                &varmap,
                &vb,
                &name,
                in_dim,
                zone_hidden,
                seed,
                dev,
            )?);
        }
        let head = seeded_linear(
            &varmap,
            &vb,
            "head",
            n_zones * zone_hidden,
            out_dim,
            seed,
            dev,
        )?;
        let params = count_params(&varmap);
        Ok(PartitionedArm {
            varmap,
            zones,
            head,
            params,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let zone_outs: Vec<Tensor> = self
            .zones
            .iter()
            .map(|z| z.forward(x)?.relu())
            .collect::<Result<_>>()?;
        let merged = Tensor::cat(&zone_outs, 1)?; // (n, n_zones*zone_hidden)
        self.head.forward(&merged)
    }
}

impl AblationArm for PartitionedArm {
    fn param_count(&self) -> usize {
        self.params
    }
    fn name(&self) -> &str {
        "partitioned"
    }
    fn train(&mut self, data: &TrainData, steps: usize, lr: f64) -> Result<(f32, f32)> {
        let mut opt = AdamW::new(
            self.varmap.all_vars(),
            ParamsAdamW {
                lr,
                ..Default::default()
            },
        )?;
        let initial = loss::mse(&self.forward(&data.x)?, &data.y)?.to_scalar::<f32>()?;
        let mut final_loss = initial;
        for _ in 0..steps {
            let l = loss::mse(&self.forward(&data.x)?, &data.y)?;
            opt.backward_step(&l)?;
            final_loss = l.to_scalar::<f32>()?;
        }
        Ok((initial, final_loss))
    }
}

/// The dense baseline arm: a fully-connected two-layer trunk.
pub struct DenseArm {
    varmap: VarMap,
    l1: Linear,
    l2: Linear,
    params: usize,
}

impl DenseArm {
    pub fn new(
        in_dim: usize,
        hidden: usize,
        out_dim: usize,
        seed: u64,
        dev: &Device,
    ) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, dev);
        let l1 = seeded_linear(&varmap, &vb, "l1", in_dim, hidden, seed, dev)?;
        let l2 = seeded_linear(&varmap, &vb, "l2", hidden, out_dim, seed, dev)?;
        let params = count_params(&varmap);
        Ok(DenseArm {
            varmap,
            l1,
            l2,
            params,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.l2.forward(&self.l1.forward(x)?.relu()?)
    }
}

impl AblationArm for DenseArm {
    fn param_count(&self) -> usize {
        self.params
    }
    fn name(&self) -> &str {
        "dense"
    }
    fn train(&mut self, data: &TrainData, steps: usize, lr: f64) -> Result<(f32, f32)> {
        let mut opt = AdamW::new(
            self.varmap.all_vars(),
            ParamsAdamW {
                lr,
                ..Default::default()
            },
        )?;
        let initial = loss::mse(&self.forward(&data.x)?, &data.y)?.to_scalar::<f32>()?;
        let mut final_loss = initial;
        for _ in 0..steps {
            let l = loss::mse(&self.forward(&data.x)?, &data.y)?;
            opt.backward_step(&l)?;
            final_loss = l.to_scalar::<f32>()?;
        }
        Ok((initial, final_loss))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analytic_param_counts_match_the_built_models() {
        let dev = Device::Cpu;
        let dense = DenseArm::new(8, 16, 4, 0, &dev).unwrap();
        assert_eq!(dense.param_count(), dense_params(8, 16, 4));
        let part = PartitionedArm::new(8, 5, 4, 4, 0, &dev).unwrap();
        assert_eq!(part.param_count(), partitioned_params(8, 5, 4, 4));
    }

    #[test]
    fn both_arms_train_and_reduce_loss() {
        let dev = Device::Cpu;
        let data = synthetic_data(48, 8, 4, 0, &dev).unwrap();
        let mut dense = DenseArm::new(8, 16, 4, 0, &dev).unwrap();
        let (di, df) = dense.train(&data, 150, 0.05).unwrap();
        assert!(df < di, "dense did not learn: {di} -> {df}");
        let mut part = PartitionedArm::new(8, 5, 4, 4, 0, &dev).unwrap();
        let (pi, pf) = part.train(&data, 150, 0.05).unwrap();
        assert!(pf < pi, "partitioned did not learn: {pi} -> {pf}");
    }
}
