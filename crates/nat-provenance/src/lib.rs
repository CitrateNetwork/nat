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

/// Clamp a prune fraction onto the meaningful `[0, Q16::ONE]` grid. A
/// `prune_threshold` outside this range is not a valid fraction; clamping keeps
/// the downstream integer arithmetic total (no overflow/saturation) and makes the
/// debug and release builds agree. In-range values pass through unchanged.
fn clamp_fraction(t: Q16) -> Q16 {
    if t.raw() < Q16::ZERO.raw() {
        Q16::ZERO
    } else if t.raw() > Q16::ONE.raw() {
        Q16::ONE
    } else {
        t
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
    //
    // `prune_threshold` arrives unvalidated from an untrusted `Trace` in
    // `verify_decision_faithful` (NAT2-B-020). Clamp it onto the meaningful
    // fraction grid `[0, 1]` before any arithmetic: an out-of-grid value (e.g.
    // `Q16::from_raw(i64::MIN)`) otherwise saturates `keep_frac`/`keep_raw` to the
    // Q16 limit and the ceiling `keep_raw.raw() + Q16::ONE.raw() - 1` overflows —
    // a debug panic and a release wrap that disagree on the survivor set, exactly
    // the build-profile fork the saturating-Q16 discipline exists to forbid. For an
    // honest in-grid threshold the clamp is a no-op, so decision-faithful replay is
    // unchanged. The ceiling is computed in `i128` so it cannot overflow.
    let n = ranked.len();
    let prune_threshold = clamp_fraction(prune_threshold);
    let keep_frac = Q16::ONE.sub(prune_threshold); // fraction to keep, in [0, 1]
    let keep_raw = Q16::from_raw(n as i64 * Q16::ONE.raw()).mul(keep_frac); // n * keep_frac
                                                                            // ceil(n * keep_frac) in i128 (cannot overflow), clamped to [1, n].
    let mut keep =
        ((keep_raw.raw() as i128 + Q16::ONE.raw() as i128 - 1) / Q16::ONE.raw() as i128) as usize;
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

    /// NAT2-B-020 tripwire. `prune_and_reweight` runs on a `prune_threshold` that
    /// `verify_decision_faithful` passes straight from an untrusted `Trace`. It
    /// must be TOTAL over the whole `Q16` domain and return the SAME survivor set
    /// in debug and release. The old body reached through `Q16` to raw `i64` for
    /// the ceiling and, with `prune_threshold = Q16::from_raw(i64::MIN)`, saturated
    /// `keep_raw` to the Q16 limit and overflowed `keep_raw.raw() + ONE - 1` —
    /// panicking in debug ("attempt to add with overflow") and wrapping in release
    /// to a different survivor set (the fork the saturating-Q16 discipline forbids).
    /// The fix clamps the threshold onto `[0, 1]` and computes the ceiling in i128.
    #[test]
    fn prune_is_total_over_untrusted_threshold_nat2_b_020() {
        let scores = vec![
            (ZoneId::HP, q(0.9)),
            (ZoneId::PF, q(0.5)),
            (ZoneId::SM, q(0.2)),
        ];
        // Adversarial out-of-grid thresholds must neither panic nor wrap.
        for bad in [
            Q16::from_raw(i64::MIN),
            Q16::from_raw(i64::MAX),
            Q16::from_raw(-1),
            Q16::from_f32(-5.0),
            Q16::from_f32(5.0),
        ] {
            let d = prune_and_reweight(&scores, bad);
            assert!(
                !d.survivors.is_empty() && d.survivors.len() <= scores.len(),
                "threshold {:?} produced an out-of-range survivor set",
                bad.raw()
            );
        }
        // A negative threshold clamps to 0 (keep all); a >1 threshold clamps to 1
        // (keep exactly one). Both agree in debug and release.
        assert_eq!(
            prune_and_reweight(&scores, Q16::from_f32(-1.0))
                .survivors
                .len(),
            3,
        );
        assert_eq!(
            prune_and_reweight(&scores, Q16::from_f32(2.0))
                .survivors
                .len(),
            1,
        );
        // In-grid thresholds are unaffected by the clamp (decision-faithful replay
        // is unchanged): keep ceil(3 * 0.5) = 2.
        assert_eq!(
            prune_and_reweight(&scores, q(0.5)).survivors,
            vec![ZoneId::HP, ZoneId::PF],
        );
    }

    /// NAT2-B-001 RED-witness (disposition: HELD/OWNER — the real fix is
    /// architectural). A `Trace` carries no signature, no model/weight
    /// commitment, and no round/nonce, and `verify_decision_faithful` only
    /// replays the merge against the trace's OWN recorded scores. So a trace
    /// whose scores, backend and output_hash are wholly invented — produced by a
    /// node that never ran the model — is "decision-faithful" and hashes cleanly,
    /// indistinguishable to any third party from a genuine trace.
    ///
    /// This test asserts the CURRENT (unsound) behavior on purpose, as a canary.
    /// When the trace is bound to a signer + model commitment (extend the
    /// `SignedContribution` signing message to cover `Trace::canonical_bytes`,
    /// add a `model_commitment`/`round`, and require verification at the consumer
    /// boundary), fabrication must stop being accepted — invert this test then.
    #[test]
    fn fabricated_trace_is_decision_faithful_ccp_nat2_b_001_witness() {
        // Scores are invented; survivors/weights are made self-consistent by the
        // same rule an honest producer would run — the ONLY thing the check binds.
        let scores = vec![(ZoneId::HP, q(0.9)), (ZoneId::PF, q(0.5))];
        let decision = prune_and_reweight(&scores, q(0.0));

        let forged = Trace {
            input_hash: "deadbeef".into(),
            // A GPU-run claim the node never made.
            backend: "candle-cuda".into(),
            router: RouterRecord {
                zone_activation: vec![],
                edge_modulation: vec![],
            },
            zones: vec![],
            inter_zone_flows: vec![],
            merge: MergeRecord {
                scores,
                prune_threshold: q(0.0),
                survivors: decision.survivors,
                weights: decision.weights,
            },
            codec: CodecRecord {
                verification: Verification::Pass,
                artifact_hash: "0000".into(),
            },
            mcp: McpRecord {
                state_transitions: vec![],
                tool_calls: vec![],
                refusal: None,
            },
            // An output the node never computed.
            output_hash: "not-a-real-output".into(),
        };

        // WITNESS: the wholly-fabricated trace passes and commits to a stable hash.
        assert!(verify_decision_faithful(&forged));
        assert_eq!(forged.trace_hash().len(), 64);
    }
}
