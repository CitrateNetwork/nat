//! Q16.16 deterministic fixed-point.
//!
//! The merge and the tool-routing path run on this type, not on `f32`, so that
//! the same gathered set always composes to the same bits (Architecture §3
//! "determinism where it matters", §6). Integer arithmetic is bit-identical
//! across platforms and across federated nodes; IEEE-754 float is not. This is
//! the property that lets federated results reconcile and on-chain provenance
//! verify (MergeDeterminism `DeterminismTheorem`).
//!
//! Representation: a value `v` is stored as the integer `round(v * 2^16)` in an
//! `i64`. Multiplication uses an `i128` intermediate so the 32→64 bit growth of
//! a product cannot overflow before the right-shift.

use serde::{Deserialize, Serialize};

const FRAC_BITS: u32 = 16;
const ONE_RAW: i64 = 1 << FRAC_BITS; // 65536

/// A Q16.16 fixed-point number. Serializes as its raw integer so the encoding
/// is exact and platform-independent (no float ever touches the wire here).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Q16(i64);

impl Q16 {
    pub const ZERO: Q16 = Q16(0);
    pub const ONE: Q16 = Q16(ONE_RAW);

    /// Construct from a raw Q16.16 integer (value = raw / 2^16).
    pub const fn from_raw(raw: i64) -> Self {
        Q16(raw)
    }

    /// The underlying raw integer. This is what gets hashed and committed.
    pub const fn raw(self) -> i64 {
        self.0
    }

    /// Quantize an `f32` onto the Q16.16 grid. This is the *only* lossy boundary:
    /// once a value is on the grid, every operation below stays exact. Round to
    /// nearest, ties away from zero, deterministically.
    pub fn from_f32(v: f32) -> Self {
        let scaled = (v as f64) * (ONE_RAW as f64);
        Q16(scaled.round() as i64)
    }

    /// Dequantize back to `f32` for display / non-deterministic downstream use.
    pub fn to_f32(self) -> f32 {
        (self.0 as f64 / ONE_RAW as f64) as f32
    }

    // Inherent arithmetic methods (not the std `Add`/`Sub`/`Mul`/`Div` traits) so
    // every fixed-point operation is explicit and greppable on the deterministic
    // path — `a.mul(b)` reads as "the Q16.16 multiply", never an ambient `*`.
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: Q16) -> Q16 {
        Q16(self.0 + other.0)
    }

    #[allow(clippy::should_implement_trait)]
    pub fn sub(self, other: Q16) -> Q16 {
        Q16(self.0 - other.0)
    }

    /// Fixed-point multiply: (a * b) >> 16, via i128 to avoid mid-product overflow.
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, other: Q16) -> Q16 {
        let prod = (self.0 as i128) * (other.0 as i128);
        Q16((prod >> FRAC_BITS) as i64)
    }

    /// Fixed-point divide: (a << 16) / b, via i128. Caller guarantees `other != 0`.
    #[allow(clippy::should_implement_trait)]
    pub fn div(self, other: Q16) -> Q16 {
        debug_assert!(other.0 != 0, "Q16 division by zero");
        let num = (self.0 as i128) << FRAC_BITS;
        Q16((num / other.0 as i128) as i64)
    }
}

impl std::iter::Sum for Q16 {
    fn sum<I: Iterator<Item = Q16>>(iter: I) -> Self {
        iter.fold(Q16::ZERO, |acc, x| acc.add(x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_round_trips() {
        assert_eq!(Q16::ONE.raw(), 65536);
        assert_eq!(Q16::from_f32(1.0), Q16::ONE);
        assert_eq!(Q16::ONE.to_f32(), 1.0);
    }

    #[test]
    fn mul_is_exact_on_grid() {
        // 0.5 * 0.5 = 0.25, exactly representable.
        let half = Q16::from_f32(0.5);
        assert_eq!(half.mul(half), Q16::from_f32(0.25));
    }

    #[test]
    fn mul_does_not_overflow_for_reasonable_magnitudes() {
        // 1000 * 1000 = 1_000_000, well within i64 after the shift.
        let k = Q16::from_f32(1000.0);
        assert_eq!(k.mul(k).to_f32(), 1_000_000.0);
    }

    #[test]
    fn div_then_mul_reconstructs() {
        let a = Q16::from_f32(3.0);
        let b = Q16::from_f32(4.0);
        let q = a.div(b); // 0.75
        assert_eq!(q.mul(b), a); // 0.75 * 4 = 3 exactly
    }

    #[test]
    fn determinism_same_inputs_same_bits() {
        // The load-bearing property: identical inputs -> identical raw bits,
        // every time, with no float nondeterminism in the path.
        let xs = [0.1f32, 0.2, 0.3, 0.4];
        let run = || -> i64 {
            xs.iter()
                .map(|&x| Q16::from_f32(x))
                .sum::<Q16>()
                .mul(Q16::from_f32(7.0))
                .raw()
        };
        assert_eq!(run(), run());
    }
}
