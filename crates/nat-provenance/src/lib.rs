//! The provenance trace (Architecture §7) and the canonical merge decision.
//!
//! Two things live here on purpose:
//!
//! 1. **The trace** — a structured, deterministically-serializable record of one
//!    forward pass. It hashes to a single digest that can be committed on-chain
//!    and replayed by a third party.
//! 2. **The merge decision** ([`prune_and_reweight`]) — the *pure* function that
//!    turns gathered scores into survivors and weights. It lives here, not in
//!    `nat-core`, so there is exactly ONE implementation: the one that produces
//!    the trace and the one that verifies it are the same code. That is what
//!    makes [`verify_decision_faithful`] meaningful rather than circular.
//!
//! ## Faithfulness, stated honestly (critique remediation #3)
//!
//! We distinguish two claims:
//!
//! - **Decision-faithful** — replaying the recorded scores reproduces the recorded
//!   survivor set and weights. This is a pure integer computation; it always
//!   holds and is what [`verify_decision_faithful`] checks. This is the product
//!   guarantee: "which zones fired, what got pruned, with what weights" is
//!   verifiable by anyone.
//! - **Bit-faithful** — re-running the full forward pass reproduces `output_hash`
//!   bit-for-bit. This holds only under a fully deterministic inference path
//!   (the Q16.16 merge composes deterministically, but the learned zone cores are
//!   float and only deterministic under a deterministic-inference mode). The
//!   model-level bit-faithful check lives in `nat-core`.

use nat_types::{CoreType, Verification, ZoneId, ZoneStatus, Q16};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The router's per-input output, as recorded (Architecture §5.2).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouterRecord {
    /// Zone-activation vector, in canonical `ZoneId::ALL` order (length 6).
    pub zone_activation: Vec<(ZoneId, Q16)>,
    /// Edge-modulation weights, one per *declared* topology edge. By
    /// construction there is no entry for an undeclared edge.
    pub edge_modulation: Vec<EdgeRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EdgeRecord {
    pub from: ZoneId,
    pub to: ZoneId,
    pub strength: Q16,
}

/// One zone's record for the pass (Architecture §7).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZoneRecord {
    pub id: ZoneId,
    pub core: CoreType,
    pub activated: bool,
    pub confidence: Q16,
    pub latency_ms: u64,
    pub status: ZoneStatus,
}

/// The merge record: scores in, decision out (Architecture §6).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MergeRecord {
    /// Combined score for each gathered zone (canonical order).
    pub scores: Vec<(ZoneId, Q16)>,
    /// Fraction dropped (e.g. 0.8 means keep the top 20%).
    pub prune_threshold: Q16,
    /// Zones that survived the prune (canonical order).
    pub survivors: Vec<ZoneId>,
    /// Normalized composition weight per survivor; these sum to ~`Q16::ONE`.
    pub weights: Vec<(ZoneId, Q16)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodecRecord {
    pub verification: Verification,
    pub artifact_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCallRecord {
    pub tool: String,
    pub args_hash: String,
    pub result_status: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpRecord {
    pub state_transitions: Vec<String>,
    pub tool_calls: Vec<ToolCallRecord>,
    /// A recorded refusal, if the harness failed closed (none on the happy path).
    pub refusal: Option<String>,
}

/// The full provenance trace emitted alongside the model output on every pass.
///
/// Field order is fixed: the deterministic hash depends on it. Every collection
/// here is an ordered `Vec` (never a `HashMap`) so serialization is reproducible.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Trace {
    pub input_hash: String,
    /// Identifier of the core backend that ran this pass (e.g. "toy-l0",
    /// "candle-cpu"). Recorded so an auditor — and the L1/DGX gate — can verify
    /// which implementation produced the trace, and in particular that a real
    /// run did NOT silently fall back to the toy L0 cores.
    pub backend: String,
    pub router: RouterRecord,
    pub zones: Vec<ZoneRecord>,
    pub inter_zone_flows: Vec<EdgeRecord>,
    pub merge: MergeRecord,
    pub codec: CodecRecord,
    pub mcp: McpRecord,
    pub output_hash: String,
}

impl Trace {
    /// Deterministic serialization → SHA-256 → hex. Serializing the same trace
    /// twice yields the same bytes (struct field order is stable, `Q16`
    /// serializes as a raw integer, no maps), so the hash is reproducible.
    /// This is the digest committed on-chain.
    pub fn trace_hash(&self) -> String {
        let mut bytes = Vec::new();
        ciborium::into_writer(self, &mut bytes).expect("trace is always serializable");
        hex(&Sha256::digest(&bytes))
    }

    /// The canonical bytes that `trace_hash` digests. Exposed for callers that
    /// want to commit the bytes themselves.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        ciborium::into_writer(self, &mut bytes).expect("trace is always serializable");
        bytes
    }
}

/// The result of the merge's prune+reweight step: a pure function of the inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeDecision {
    pub survivors: Vec<ZoneId>,
    pub weights: Vec<(ZoneId, Q16)>,
}

/// The canonical merge decision (Architecture §6 steps 2–3): prune the bottom
/// `prune_threshold` fraction by score, then normalize the survivors' scores
/// into composition weights summing to `Q16::ONE`.
///
/// Determinism is mandatory here — this runs on the Q16.16 path and its output
/// must be identical across nodes. Ties are broken by canonical `ZoneId` order,
/// never by hash-map iteration or float comparison.
///
/// `scores` is `(zone, combined_score)` for the gathered zones. At least one
/// zone always survives (you cannot prune everything).
pub fn prune_and_reweight(scores: &[(ZoneId, Q16)], prune_threshold: Q16) -> MergeDecision {
    assert!(!scores.is_empty(), "prune called on an empty gathered set");

    // Rank by score descending; tie-break by canonical ZoneId ascending so the
    // ranking is total and deterministic.
    let mut ranked: Vec<(ZoneId, Q16)> = scores.to_vec();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    // Keep the top (1 - prune_threshold) fraction, at least one zone.
    let n = ranked.len();
    let keep_frac = Q16::ONE.sub(prune_threshold); // fraction to keep
    let keep_raw = Q16::from_raw(n as i64 * Q16::ONE.raw()).mul(keep_frac); // n * keep_frac
                                                                            // ceil(n * keep_frac), clamped to [1, n].
    let mut keep = ((keep_raw.raw() + Q16::ONE.raw() - 1) / Q16::ONE.raw()) as usize;
    keep = keep.clamp(1, n);

    let survivors_ranked: Vec<(ZoneId, Q16)> = ranked.into_iter().take(keep).collect();

    // Normalize survivor scores into weights. If every survivor scored zero,
    // fall back to equal weights so the composition is still well-defined.
    let sum: Q16 = survivors_ranked.iter().map(|(_, s)| *s).sum();
    let mut weights: Vec<(ZoneId, Q16)> = if sum == Q16::ZERO {
        let equal = Q16::ONE.div(Q16::from_raw(keep as i64 * Q16::ONE.raw()));
        survivors_ranked.iter().map(|(z, _)| (*z, equal)).collect()
    } else {
        survivors_ranked
            .iter()
            .map(|(z, s)| (*z, s.div(sum)))
            .collect()
    };

    // Emit survivors and weights in canonical ZoneId order so the trace is
    // comparable across passes and nodes.
    weights.sort_by_key(|(z, _)| *z);
    let survivors: Vec<ZoneId> = weights.iter().map(|(z, _)| *z).collect();
    MergeDecision { survivors, weights }
}

/// Decision-faithful replay: recompute the merge decision from the trace's own
/// recorded scores and threshold, and confirm it matches the recorded survivors
/// and weights. If this returns `true`, an auditor knows the recorded "which
/// zones survived, with what weights" was not fabricated — it is exactly what
/// the deterministic rule produces from the recorded scores.
pub fn verify_decision_faithful(trace: &Trace) -> bool {
    if trace.merge.scores.is_empty() {
        return trace.merge.survivors.is_empty();
    }
    let recomputed = prune_and_reweight(&trace.merge.scores, trace.merge.prune_threshold);
    recomputed.survivors == trace.merge.survivors && recomputed.weights == trace.merge.weights
}

/// Lowercase hex encoding (no external dep needed for 32 bytes).
pub fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(char::from_digit((b >> 4) as u32, 16).unwrap());
        s.push(char::from_digit((b & 0xf) as u32, 16).unwrap());
    }
    s
}

/// SHA-256 → hex, for hashing inputs/outputs/artifacts into the trace.
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(v: f32) -> Q16 {
        Q16::from_f32(v)
    }

    #[test]
    fn prune_keeps_top_fraction_and_normalizes() {
        // Five zones, drop 80% → keep top 1 (ceil(5*0.2)=1).
        let scores = vec![
            (ZoneId::SM, q(0.1)),
            (ZoneId::CB, q(0.2)),
            (ZoneId::HP, q(0.9)),
            (ZoneId::PF, q(0.5)),
            (ZoneId::CX, q(0.3)),
        ];
        let d = prune_and_reweight(&scores, q(0.8));
        assert_eq!(d.survivors, vec![ZoneId::HP]);
        // Single survivor normalizes to 1.
        assert_eq!(d.weights, vec![(ZoneId::HP, Q16::ONE)]);
    }

    #[test]
    fn prune_70_percent_keeps_more_and_weights_sum_to_one() {
        let scores = vec![
            (ZoneId::SM, q(0.1)),
            (ZoneId::CB, q(0.2)),
            (ZoneId::HP, q(0.9)),
            (ZoneId::PF, q(0.5)),
            (ZoneId::CX, q(0.3)),
        ];
        // Drop 70% → keep ceil(5*0.3)=2 (HP=0.9, PF=0.5).
        let d = prune_and_reweight(&scores, q(0.7));
        assert_eq!(d.survivors, vec![ZoneId::HP, ZoneId::PF]);
        let sum: Q16 = d.weights.iter().map(|(_, w)| *w).sum();
        // Weights normalize to 1 within one Q16 ulp of rounding.
        assert!((sum.raw() - Q16::ONE.raw()).abs() <= 2);
    }

    #[test]
    fn prune_is_deterministic_under_ties() {
        // All equal scores: tie-break by canonical ZoneId order must be stable.
        let scores = vec![
            (ZoneId::PF, q(0.5)),
            (ZoneId::SM, q(0.5)),
            (ZoneId::HP, q(0.5)),
            (ZoneId::CB, q(0.5)),
        ];
        let a = prune_and_reweight(&scores, q(0.5));
        let b = prune_and_reweight(&scores, q(0.5));
        assert_eq!(a, b);
        // keep ceil(4*0.5)=2 → first two in canonical order: SM, CB.
        assert_eq!(a.survivors, vec![ZoneId::SM, ZoneId::CB]);
    }

    #[test]
    fn at_least_one_survivor_even_at_extreme_prune() {
        let scores = vec![(ZoneId::PF, q(0.5)), (ZoneId::HP, q(0.4))];
        let d = prune_and_reweight(&scores, q(0.99));
        assert_eq!(d.survivors.len(), 1);
    }
}
