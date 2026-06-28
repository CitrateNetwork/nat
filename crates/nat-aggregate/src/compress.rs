//! int8 pseudo-gradient compression (AGG-S1 WP-6 — optimization, first-class).
//!
//! DiLoCo over the real internet is bandwidth-bound, so the outer-step delta is the
//! thing to shrink. A pseudo-gradient ships as `i64` Q16 coordinates; this module
//! quantizes each round's delta to **int8 + one shared scale** (an ~8× wire shrink
//! per coordinate, the first rung of the int8→int4 ladder INTELLECT-2 rode to 400×).
//!
//! The non-negotiable constraint: compression must stay on the **deterministic
//! integer path**, so two nodes that compress, ship, decompress, and aggregate the
//! *same* deltas still reconcile bit-for-bit (the whole point of WS-1). Every step
//! here is integer arithmetic on the raw `i64`s — no float, no platform-dependent
//! rounding — so `aggregate(decompress(compress(g)))` is reproducible and frozen
//! below, exactly as the uncompressed path is.

use nat_types::Q16;

/// A pseudo-gradient quantized to int8 with a single per-vector scale. The wire form
/// is `scale_raw` (8 bytes, once) + `values` (1 byte each), versus 8 bytes/coordinate
/// uncompressed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressedGradient {
    /// The Q16 raw quantization step: coordinate ≈ `value * scale_raw`.
    pub scale_raw: i64,
    /// The int8 quantized coordinates.
    pub values: Vec<i8>,
}

impl CompressedGradient {
    /// Wire size in bytes (8 for the scale + 1 per coordinate).
    pub fn wire_bytes(&self) -> usize {
        8 + self.values.len()
    }
}

/// Quantize a Q16 pseudo-gradient to int8 + scale. The scale is `max|coord| / 127`
/// (integer division of the raw values), so the largest-magnitude coordinate maps to
/// ±127 and the rest scale linearly. All-zero input maps to scale 1 / all-zero values
/// (no division by zero). Deterministic: integer ops only.
pub fn compress(coords: &[Q16]) -> CompressedGradient {
    let max_abs = coords
        .iter()
        .map(|c| c.raw().unsigned_abs())
        .max()
        .unwrap_or(0);
    // scale_raw = ceil(max_abs / 127) so the max coordinate quantizes within ±127.
    let scale_raw = if max_abs == 0 {
        1
    } else {
        max_abs.div_ceil(127) as i64
    };
    let values = coords
        .iter()
        .map(|c| {
            // Round-to-nearest integer division on the raw, then clamp into i8 range.
            let q = div_round_nearest(c.raw(), scale_raw);
            q.clamp(-127, 127) as i8
        })
        .collect();
    CompressedGradient { scale_raw, values }
}

/// Reconstruct a Q16 pseudo-gradient from its int8 form: `coord = value * scale_raw`.
/// Exact integer multiply — deterministic, lossy only by the quantization already
/// baked into [`compress`].
pub fn decompress(c: &CompressedGradient) -> Vec<Q16> {
    c.values
        .iter()
        .map(|&v| Q16::from_raw(v as i64 * c.scale_raw))
        .collect()
}

/// Round-to-nearest integer division (ties away from zero), branch-symmetric for
/// negative numerators so quantization is unbiased and sign-symmetric.
fn div_round_nearest(num: i64, den: i64) -> i64 {
    debug_assert!(den > 0);
    if num >= 0 {
        (num + den / 2) / den
    } else {
        -(((-num) + den / 2) / den)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{aggregate, PseudoGradient};

    fn qv(coords: &[f32]) -> Vec<Q16> {
        coords.iter().map(|&v| Q16::from_f32(v)).collect()
    }

    #[test]
    fn roundtrip_is_deterministic_and_close() {
        let g = qv(&[1.0, -2.0, 3.5, 0.25, -0.75]);
        let c1 = compress(&g);
        let c2 = compress(&g);
        assert_eq!(c1, c2, "compression is deterministic");
        let back = decompress(&c1);
        // Each reconstructed coordinate is within one quantization step of the input.
        for (orig, rec) in g.iter().zip(&back) {
            assert!((orig.raw() - rec.raw()).abs() <= c1.scale_raw);
        }
    }

    #[test]
    fn all_zero_gradient_is_safe() {
        let g = qv(&[0.0, 0.0, 0.0]);
        let c = compress(&g);
        assert_eq!(c.scale_raw, 1);
        assert_eq!(c.values, vec![0, 0, 0]);
        assert_eq!(decompress(&c), g);
    }

    #[test]
    fn wire_shrink_is_real() {
        let g = qv(&[1.0; 256]);
        let c = compress(&g);
        // uncompressed: 8 bytes/coord = 2048; compressed: 8 + 256 = 264 → ~7.8×.
        assert_eq!(c.wire_bytes(), 8 + 256);
        assert!(c.wire_bytes() * 7 < 256 * 8);
    }

    /// WP-6 acceptance: aggregating over the COMPRESSED path is itself deterministic
    /// and bit-reproducible — a frozen digest, like the uncompressed path. Two nodes
    /// on the compressed wire still reconcile.
    #[test]
    fn frozen_compressed_path_aggregate() {
        let raw = [
            PseudoGradient::new("alpha", qv(&[1.0, -2.0, 3.5])),
            PseudoGradient::new("beta", qv(&[2.0, -1.0, 3.0])),
            PseudoGradient::new("gamma", qv(&[1.5, -1.5, 3.25])),
            PseudoGradient::new("delta", qv(&[1.75, -1.25, 3.1])),
        ];
        // Compress → decompress each delta, then aggregate the compressed path.
        let compressed: Vec<PseudoGradient> = raw
            .iter()
            .map(|g| PseudoGradient::new(g.node_id.clone(), decompress(&compress(&g.coords))))
            .collect();
        let r = aggregate(&compressed, 1, 64, b"frozen-seed-v1").expect("aggregate");
        assert_eq!(
            r.digest, "014ee81a5ef2a076780689dd60b743c218f5e74788fa58678f3f51b2837c4f9c",
            "compressed-path Q16 aggregate digest drifted — review before re-freezing"
        );
    }
}
