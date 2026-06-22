//! Deterministic, reproducible weight init for the trainable stack.
//!
//! candle's own init draws from the device RNG, which the CPU backend cannot seed
//! (`Device::set_seed` errors on CPU). So we seed the weights ourselves from a
//! small PRNG and pre-insert them into the `VarMap` under the names `linear` looks
//! up — `VarMap::get` then reuses them, and the optimizer still trains them. Same
//! seed → bit-identical init on both CPU and GPU, honouring the reproducibility
//! floor (PLANSET/01). (nat-ablation grew an equivalent helper first; this is the
//! shared home for the L1 training stack.)

use candle_core::{Device, Result, Tensor, Var};
use candle_nn::{linear, Linear, VarBuilder, VarMap};

/// SplitMix64 — a tiny deterministic PRNG.
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        SplitMix64(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// A float in `[lo, hi)`, deterministic from the seed stream.
    pub fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        let u = (self.next_u64() >> 11) as f32 / (1u64 << 53) as f32; // [0,1)
        lo + (hi - lo) * u
    }
}

/// FNV-1a of a layer name, so each layer gets a distinct sub-seed (no init
/// symmetry across layers) while staying a pure function of `(seed, name)`.
pub fn name_seed(seed: u64, name: &str) -> u64 {
    let mut h = 0xCBF2_9CE4_8422_2325u64 ^ seed;
    for b in name.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

/// A deterministic uniform `[-k, k]` tensor of the given shape.
pub fn seeded_uniform(shape: (usize, usize), k: f32, seed: u64, dev: &Device) -> Result<Tensor> {
    let (r, c) = shape;
    let mut rng = SplitMix64::new(seed);
    let v: Vec<f32> = (0..r * c).map(|_| rng.uniform(-k, k)).collect();
    Tensor::from_vec(v, (r, c), dev)
}

/// A `Linear` whose weights are deterministically seeded. The variables are
/// pre-inserted into the `VarMap` under the names `linear` looks up (`weight`,
/// `bias`); `VarMap::get` reuses them, so the optimizer trains the same tensors
/// while init stays reproducible. Uniform `[-k, k]`, `k = 1/sqrt(in_dim)`.
pub fn seeded_linear(
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
    linear(in_dim, out_dim, vb.pp(prefix))
}

/// A seeded trainable scalar variable, registered into the `VarMap` under `name`.
/// Used for per-core scalars (e.g. the SSM decay `log_a`). Returns the tensor
/// handle (shares id with the var, so the optimizer trains it).
pub fn seeded_scalar_var(varmap: &VarMap, name: &str, init: f32, dev: &Device) -> Result<Tensor> {
    let var = Var::from_tensor(&Tensor::from_vec(vec![init], (1,), dev)?)?;
    let t = var.as_tensor().clone();
    varmap.data().lock().unwrap().insert(name.to_string(), var);
    Ok(t)
}
