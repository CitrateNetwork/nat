//! The merge (Architecture §6): score → prune → re-weight → compose.
//!
//! Steps 2–3 (prune + re-weight) are delegated to
//! [`nat_provenance::prune_and_reweight`] so the decision has exactly one
//! implementation — the one that produces the trace and the one that verifies it
//! are identical. Step 4 (compose) runs on the Q16.16 deterministic path: the
//! same gathered set always composes to the same bits, which is what federated
//! reconciliation and on-chain replay require (MergeDeterminism).

use crate::cores::D_OUT;
use nat_provenance::{prune_and_reweight, MergeDecision};
use nat_types::{ZoneId, Q16};

/// A gathered zone output entering the merge: its combined score and its summary.
#[derive(Debug, Clone)]
pub struct Gathered {
    pub zone: ZoneId,
    /// Combined score = activation × confidence (Architecture §6 step 1).
    pub score: Q16,
    /// The zone's fixed-width summary.
    pub summary: [f32; D_OUT],
}

#[derive(Debug, Clone)]
pub struct MergeOutput {
    pub decision: MergeDecision,
    /// The composed output vector, on the Q16.16 grid (raw integers).
    pub composed_q16: [Q16; D_OUT],
    /// The composed output as f32, for downstream non-deterministic use.
    pub composed_f32: [f32; D_OUT],
}

/// Run the full merge over the gathered set. Returns the decision (survivors +
/// weights) and the deterministically composed output.
pub fn merge(gathered: &[Gathered], prune_threshold: Q16) -> MergeOutput {
    let scores: Vec<(ZoneId, Q16)> = gathered.iter().map(|g| (g.zone, g.score)).collect();
    let decision = prune_and_reweight(&scores, prune_threshold);

    // Compose survivors by weighted sum, entirely on the Q16.16 path.
    let mut composed_q16 = [Q16::ZERO; D_OUT];
    for (zone, weight) in &decision.weights {
        let g = gathered
            .iter()
            .find(|g| g.zone == *zone)
            .expect("survivor must be in the gathered set");
        for (i, slot) in composed_q16.iter_mut().enumerate() {
            let contribution = weight.mul(Q16::from_f32(g.summary[i]));
            *slot = slot.add(contribution);
        }
    }

    let mut composed_f32 = [0.0f32; D_OUT];
    for i in 0..D_OUT {
        composed_f32[i] = composed_q16[i].to_f32();
    }

    MergeOutput {
        decision,
        composed_q16,
        composed_f32,
    }
}

/// Hash the composed output for the trace's `output_hash`. Hashes the Q16.16 raw
/// integers (little-endian), never the float — so the hash is bit-reproducible
/// across nodes (this is the bit-faithful surface).
pub fn output_hash(composed_q16: &[Q16; D_OUT]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    for q in composed_q16 {
        h.update(q.raw().to_le_bytes());
    }
    nat_provenance::hex(&h.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g(zone: ZoneId, score: f32, fill: f32) -> Gathered {
        Gathered {
            zone,
            score: Q16::from_f32(score),
            summary: [fill; D_OUT],
        }
    }

    #[test]
    fn same_gathered_set_composes_to_same_hash() {
        // The federation-critical property (DeterminismTheorem): identical input
        // → identical output bits.
        let set = vec![
            g(ZoneId::HP, 0.9, 0.5),
            g(ZoneId::PF, 0.5, 0.2),
            g(ZoneId::CX, 0.1, 0.9),
        ];
        let a = merge(&set, Q16::from_f32(0.5));
        let b = merge(&set, Q16::from_f32(0.5));
        assert_eq!(output_hash(&a.composed_q16), output_hash(&b.composed_q16));
    }

    #[test]
    fn single_survivor_composes_to_its_own_summary() {
        // Drop 80% of 5 → keep 1 (HP, the top scorer). Composed == HP summary.
        let set = vec![
            g(ZoneId::SM, 0.1, 0.1),
            g(ZoneId::CB, 0.2, 0.2),
            g(ZoneId::HP, 0.9, 0.7),
            g(ZoneId::PF, 0.3, 0.3),
            g(ZoneId::CX, 0.4, 0.4),
        ];
        let out = merge(&set, Q16::from_f32(0.8));
        assert_eq!(out.decision.survivors, vec![ZoneId::HP]);
        // weight 1.0 × 0.7 = 0.7 in every slot (within a Q16 ulp).
        assert!((out.composed_f32[0] - 0.7).abs() < 1e-3);
    }
}
