//! Candle-backed zone cores (CPU). These implement the same
//! [`nat_core::cores::ZoneCore`] trait the L0 toy cores do (ADR-0009), so they
//! drop in behind the trait with nothing above `cores` needing to change. The
//! difference: the math runs on Candle tensors (matmul, softmax), so the exact
//! same code path moves to a CUDA device at L1 by swapping `Device::Cpu` for a
//! GPU device — that is the de-risk of critique #6 (the Rust training stack).
//!
//! The cores here use fixed parameters, so a forward pass is deterministic
//! (same slice → same output). The trainable side — proving forward + backward +
//! optimizer all work on Candle — lives in [`crate::train`].

use candle_core::{Device, Tensor};
use nat_core::cores::{CoreOutput, ZoneCore, D_OUT};
use nat_types::CoreType;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Average a sequence into `D_OUT` contiguous buckets (matches the L0 cores so
/// the summary contract is identical).
fn bucketize(seq: &[f32]) -> [f32; D_OUT] {
    let mut out = [0.0f32; D_OUT];
    if seq.is_empty() {
        return out;
    }
    let chunk = seq.len().div_ceil(D_OUT).max(1);
    for (j, slot) in out.iter_mut().enumerate() {
        let start = (j * chunk).min(seq.len());
        let end = (start + chunk).min(seq.len());
        if start < end {
            *slot = seq[start..end].iter().sum::<f32>() / (end - start) as f32;
        }
    }
    out
}

fn confidence_from(summary: &[f32; D_OUT]) -> f32 {
    let mean_abs = summary.iter().map(|v| v.abs()).sum::<f32>() / D_OUT as f32;
    sigmoid(mean_abs)
}

/// State-Space Model core on Candle. The linear recurrence
/// `h_t = a·h_{t-1} + b·x_t`, `y_t = c·h_t` unrolls to `y = M·x` where `M` is the
/// lower-triangular matrix `M[t,k] = c·b·a^(t-k)` for `k ≤ t`. Expressing it as a
/// single matmul makes it vectorized and GPU-ready (no Python-style time loop).
pub struct CandleSsmCore {
    a: f32,
    b: f32,
    c: f32,
    device: Device,
}

impl Default for CandleSsmCore {
    fn default() -> Self {
        CandleSsmCore {
            a: 0.9,
            b: 0.3,
            c: 1.0,
            device: Device::Cpu,
        }
    }
}

impl CandleSsmCore {
    /// Fallible forward (Candle ops return `Result`); [`ZoneCore::forward`] wraps this.
    pub fn try_forward(&self, slice: &[f32]) -> candle_core::Result<CoreOutput> {
        let t = slice.len();
        if t == 0 {
            return Ok(CoreOutput {
                summary: [0.0; D_OUT],
                confidence: 0.5,
            });
        }
        // Lower-triangular kernel M[t,k] = c·b·a^(t-k), k ≤ t.
        let mut m = vec![0f32; t * t];
        for i in 0..t {
            for k in 0..=i {
                m[i * t + k] = self.c * self.b * self.a.powi((i - k) as i32);
            }
        }
        let m = Tensor::from_vec(m, (t, t), &self.device)?;
        let x = Tensor::from_vec(slice.to_vec(), (t, 1), &self.device)?;
        let y = m.matmul(&x)?; // (t, 1)
        let yv = y.flatten_all()?.to_vec1::<f32>()?;
        let summary = bucketize(&yv);
        Ok(CoreOutput {
            confidence: confidence_from(&summary),
            summary,
        })
    }
}

impl ZoneCore for CandleSsmCore {
    fn forward(&self, slice: &[f32]) -> CoreOutput {
        self.try_forward(slice).expect("candle ssm forward")
    }
    fn core_type(&self) -> CoreType {
        CoreType::Ssm
    }
}

/// Single-head self-attention core on Candle. The slice is a length-`T` sequence
/// of scalar tokens; `scores = x·xᵀ`, `attn = softmax(scores)`, `out = attn·x`.
/// Real Candle `matmul` + `softmax`, so it is the same op graph a trained,
/// multi-head attention zone uses at L1.
pub struct CandleAttentionCore {
    device: Device,
}

impl Default for CandleAttentionCore {
    fn default() -> Self {
        CandleAttentionCore {
            device: Device::Cpu,
        }
    }
}

impl CandleAttentionCore {
    pub fn try_forward(&self, slice: &[f32]) -> candle_core::Result<CoreOutput> {
        let t = slice.len();
        if t == 0 {
            return Ok(CoreOutput {
                summary: [0.0; D_OUT],
                confidence: 0.5,
            });
        }
        let x = Tensor::from_vec(slice.to_vec(), (t, 1), &self.device)?;
        let scores = x.matmul(&x.t()?)?; // (t, t)
        let attn = candle_nn::ops::softmax(&scores, candle_core::D::Minus1)?;
        let out = attn.matmul(&x)?; // (t, 1)
        let ov = out.flatten_all()?.to_vec1::<f32>()?;
        let summary = bucketize(&ov);
        Ok(CoreOutput {
            confidence: confidence_from(&summary),
            summary,
        })
    }
}

impl ZoneCore for CandleAttentionCore {
    fn forward(&self, slice: &[f32]) -> CoreOutput {
        self.try_forward(slice).expect("candle attention forward")
    }
    fn core_type(&self) -> CoreType {
        CoreType::Attention
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slice() -> Vec<f32> {
        (0..16).map(|i| i as f32 / 16.0 - 0.5).collect()
    }

    #[test]
    fn ssm_core_is_deterministic_and_well_formed() {
        let core = CandleSsmCore::default();
        let a = core.forward(&slice());
        let b = core.forward(&slice());
        assert_eq!(a.summary, b.summary); // fixed params → deterministic
        assert!(a.summary.iter().all(|v| v.is_finite()));
        assert!((0.0..=1.0).contains(&a.confidence));
    }

    #[test]
    fn attention_core_is_deterministic_and_well_formed() {
        let core = CandleAttentionCore::default();
        let a = core.forward(&slice());
        let b = core.forward(&slice());
        assert_eq!(a.summary, b.summary);
        assert!(a.summary.iter().all(|v| v.is_finite()));
        assert!((0.0..=1.0).contains(&a.confidence));
    }

    #[test]
    fn candle_cores_satisfy_the_zonecore_trait() {
        // The whole point: they are drop-in for the toy cores behind the trait.
        let cores: Vec<Box<dyn ZoneCore>> = vec![
            Box::new(CandleSsmCore::default()),
            Box::new(CandleAttentionCore::default()),
        ];
        for c in &cores {
            let out = c.forward(&slice());
            assert_eq!(out.summary.len(), D_OUT);
        }
        assert_eq!(cores[0].core_type(), CoreType::Ssm);
        assert_eq!(cores[1].core_type(), CoreType::Attention);
    }

    #[test]
    fn empty_slice_is_handled() {
        assert_eq!(CandleSsmCore::default().forward(&[]).summary, [0.0; D_OUT]);
    }
}
