//! Toy zone cores for L0 (Architecture §4, §9).
//!
//! These are deliberately small but *real and deterministic*: an SSM core is a
//! linear recurrence (`h_t = a·h_{t-1} + b·x_t`, `y_t = c·h_t`), an attention
//! core is a softmax-weighted combine. L0's job is to wire up the pass and prove
//! the provenance log emits (Master Plan rung L0), not to be capable. The
//! `ZoneCore` trait is the seam where Burn/Candle-backed cores slot in at L1
//! (ADR-0006); nothing above this module knows whether a core is toy or trained.

use nat_types::CoreType;

/// Length of a zone's output summary vector. Every zone emits this width so the
/// merge can compose survivors by weighted sum without shape negotiation.
pub const D_OUT: usize = 8;

/// What a zone core returns for one input slice.
#[derive(Debug, Clone)]
pub struct CoreOutput {
    /// Fixed-width summary handed to the merge (the cross-zone head, simplified).
    pub summary: [f32; D_OUT],
    /// The zone's self-reported confidence in [0,1] (Architecture §5.3).
    pub confidence: f32,
}

/// The per-zone sequence operator. Deterministic: same slice → same output.
pub trait ZoneCore {
    fn forward(&self, slice: &[f32]) -> CoreOutput;
    fn core_type(&self) -> CoreType;
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Reduce a slice to `D_OUT` buckets by averaging contiguous chunks. Keeps the
/// summary width fixed regardless of slice width.
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
            let s: f32 = seq[start..end].iter().sum();
            *slot = s / (end - start) as f32;
        }
    }
    out
}

fn confidence_from(summary: &[f32; D_OUT]) -> f32 {
    let mean_abs = summary.iter().map(|v| v.abs()).sum::<f32>() / D_OUT as f32;
    sigmoid(mean_abs)
}

/// State-Space Model core: linear-time recurrence over the slice (ADR-0002).
/// Fixed params at L0; learned at L1.
pub struct SsmCore {
    a: f32,
    b: f32,
    c: f32,
}

impl Default for SsmCore {
    fn default() -> Self {
        SsmCore {
            a: 0.9,
            b: 0.3,
            c: 1.0,
        }
    }
}

impl ZoneCore for SsmCore {
    fn forward(&self, slice: &[f32]) -> CoreOutput {
        let mut h = 0.0f32;
        let mut seq = Vec::with_capacity(slice.len());
        for &x in slice {
            h = self.a * h + self.b * x; // state evolution
            seq.push(self.c * h); // readout
        }
        let summary = bucketize(&seq);
        let confidence = confidence_from(&summary);
        CoreOutput {
            summary,
            confidence,
        }
    }
    fn core_type(&self) -> CoreType {
        CoreType::Ssm
    }
}

/// Attention core: a softmax-weighted combine over pairs of the slice. Toy
/// single-pass attention for L0; multi-head trained attention at L1.
pub struct AttentionCore;

impl ZoneCore for AttentionCore {
    fn forward(&self, slice: &[f32]) -> CoreOutput {
        // Pair up the slice; within each pair, softmax over the two values and
        // take the weighted combination. Produces one output per pair.
        let mut combined = Vec::with_capacity(slice.len().div_ceil(2));
        let mut i = 0;
        while i < slice.len() {
            let a = slice[i];
            let b = if i + 1 < slice.len() { slice[i + 1] } else { a };
            let (wa, wb) = softmax2(a, b);
            combined.push(wa * a + wb * b);
            i += 2;
        }
        let summary = bucketize(&combined);
        let confidence = confidence_from(&summary);
        CoreOutput {
            summary,
            confidence,
        }
    }
    fn core_type(&self) -> CoreType {
        CoreType::Attention
    }
}

fn softmax2(a: f32, b: f32) -> (f32, f32) {
    let m = a.max(b);
    let ea = (a - m).exp();
    let eb = (b - m).exp();
    let z = ea + eb;
    (ea / z, eb / z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cores_are_deterministic() {
        let slice: Vec<f32> = (0..16).map(|i| i as f32 / 16.0 - 0.5).collect();
        let ssm = SsmCore::default();
        assert_eq!(ssm.forward(&slice).summary, ssm.forward(&slice).summary);
        let attn = AttentionCore;
        assert_eq!(attn.forward(&slice).summary, attn.forward(&slice).summary);
    }

    #[test]
    fn confidence_in_unit_interval() {
        let slice: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let c = SsmCore::default().forward(&slice).confidence;
        assert!((0.0..=1.0).contains(&c));
    }

    #[test]
    fn summary_is_fixed_width() {
        let slice = vec![1.0; 16];
        assert_eq!(AttentionCore.forward(&slice).summary.len(), D_OUT);
    }
}
