//! The NAT training loop.
//!
//! **L0 status: intentionally a stub.** The scale ladder (Master Plan §4) trains
//! nothing at L0 — L0 wires the forward pass and proves the provenance log
//! emits. Training begins at L1 (~1–2B on the Spark), where this crate gains a
//! Burn/Candle-backed loop, the reproducibility floor (config hash, fixed seed,
//! recorded hardware — Research Strategy §8), and the data-quality scoring that
//! feeds the compute-pool settlement seam (`docs/SETTLEMENT_SEAM.md`).
//!
//! What is fixed here now is the *shape* of a training step's accounting (the
//! settlement seam) and the **reproducibility floor** (the [`repro`] module), so
//! both can be designed against stable types before L1.

pub mod repro;

use nat_types::Q16;

/// The accounting a training step contributes toward a participant's reward.
/// NAT computes these; `citrate-compute-pool` settles them (it owns reward math,
/// tokenomics, and payout). NAT does not reinvent settlement (decision: integrate
/// with compute-pool).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepContribution {
    /// Metered compute for this step (e.g. normalized FLOP-seconds), Q16.16.
    pub compute_metered: Q16,
    /// Data-quality score of the shard trained on, in [0,1], Q16.16. Produced by
    /// the data pipeline's QUALITY_SCORE stage (Data Ops §4).
    pub data_quality: Q16,
    /// Number of tokens consumed this step.
    pub tokens: u64,
    /// The provenance trace hash for this step's forward passes.
    pub provenance_hash: String,
}

impl StepContribution {
    /// The reward *weight* NAT proposes for a step: compute × quality. This is
    /// the signal; compute-pool converts weight → payout under its tokenomics.
    /// Keeping the formula here (not the payout) is the seam boundary.
    pub fn reward_weight(&self) -> Q16 {
        reward_weight(self.compute_metered, self.data_quality)
    }
}

/// The canonical reward-weight formula, with the SAME clamping the on-chain
/// `PatronageLedger` / unified-settlement path applies (NAT-F2): compute is
/// floored at `0` and quality is clamped into `[0, 1]` before the product. The
/// legacy `Settlement::settle` path consumes this via
/// `AcceptedContribution.reward_weight`, and the unified `SettlementRow` path
/// delegates to it, so both seams now agree — previously the legacy path let a
/// `quality > 1` or negative quality flow straight through while the unified path
/// clamped. Honest inputs (`compute ≥ 0`, `quality ∈ [0, 1]`) are unaffected.
pub fn reward_weight(compute_metered: Q16, data_quality: Q16) -> Q16 {
    let compute = if compute_metered.raw() < 0 {
        Q16::ZERO
    } else {
        compute_metered
    };
    let quality = if data_quality.raw() < Q16::ZERO.raw() {
        Q16::ZERO
    } else if data_quality.raw() > Q16::ONE.raw() {
        Q16::ONE
    } else {
        data_quality
    };
    compute.mul(quality)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reward_weight_is_compute_times_quality() {
        let c = StepContribution {
            compute_metered: Q16::from_f32(4.0),
            data_quality: Q16::from_f32(0.5),
            tokens: 1024,
            provenance_hash: "abc".into(),
        };
        assert_eq!(c.reward_weight(), Q16::from_f32(2.0));
    }

    /// NAT-F2 tripwire. Reward weight must apply the on-chain clamping (compute ≥ 0,
    /// quality ∈ [0,1]) so the legacy `finalize_round`/`Settlement` path and the
    /// unified `SettlementRow` path cannot settle different amounts for the same
    /// contribution. The old formula let `quality > 1` inflate and a negative
    /// quality subtract on the legacy path.
    #[test]
    fn reward_weight_clamps_quality_and_compute_nat_f2() {
        // quality > 1 does not inflate above the compute value.
        assert_eq!(
            reward_weight(Q16::from_f32(4.0), Q16::from_f32(2.0)),
            Q16::from_f32(4.0),
        );
        // negative quality floors at 0 (no negative reward on the legacy path).
        assert_eq!(
            reward_weight(Q16::from_f32(4.0), Q16::from_f32(-0.5)),
            Q16::ZERO,
        );
        // negative compute floors at 0.
        assert_eq!(
            reward_weight(Q16::from_f32(-4.0), Q16::from_f32(0.5)),
            Q16::ZERO,
        );
        // in-range values are unaffected (honest path unchanged).
        assert_eq!(
            reward_weight(Q16::from_f32(4.0), Q16::from_f32(0.5)),
            Q16::from_f32(2.0),
        );
    }

    #[test]
    fn zero_quality_yields_zero_weight() {
        // A node that contributes compute on garbage data earns no reward weight.
        let c = StepContribution {
            compute_metered: Q16::from_f32(100.0),
            data_quality: Q16::ZERO,
            tokens: 1,
            provenance_hash: "x".into(),
        };
        assert_eq!(c.reward_weight(), Q16::ZERO);
    }
}
