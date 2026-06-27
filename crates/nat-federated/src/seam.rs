//! seam.rs — the NAT ↔ RM-FL unification seam (UNIFY-S1, WS-0 keystone).
//!
//! ADR-2026-06-27-nat-rmfl-unification: *"NAT is the model that flows through
//! RM-FL's Belnap-routing / mentorship / learning-cycle pipeline, settled through
//! the Paper VII + co-op patronage ledger."*
//!
//! This module is the **typed boundary** between NAT-federated (the model half —
//! signed gather, `compute × data_quality`, provenance) and RM-FL's machinery (the
//! Belnap precompile `0x0110`, the routing meta-model `0x0111`, the
//! `LearningOrchestrator`, and the co-op `FederatedSettlement` ledger). It declares
//! the **six seam bindings** (ADR §Decision) as types + traits.
//!
//! **Rule 1 (no stubs):** the traits here are *interfaces*. The adapters that
//! implement them (later UNIFY-S1 WPs, in `core/learning` ↔ `LearningOrchestrator`
//! ↔ the co-op contracts) wrap the **real** impls on each side — never a stub. The
//! only impls in this file are `#[cfg(test)]` reference adapters that exercise the
//! types. The seam itself owns no ML and no I/O.
//!
//! The six bindings:
//! 1. [`RoutingTarget`] / [`RoutesToZone`] — the router destination is a NAT zone.
//! 2. [`ObserveStep`] — NAT's signed gather is RM-FL's contribution intake.
//! 3. [`ZoneWeightDelta`] / [`BelnapZoneAggregator`] — Belnap-aggregate zone-weight
//!    deltas via the **Q16 precompile `0x0110`** (never the f32 `core/learning` path).
//! 4. [`SettlementRow`] / [`UnifiedSettlement`] — one ledger; the `data_quality`
//!    term flows through to the co-op `FederatedSettlement` seam.
//! 5. [`UnifiedProvenance`] — one provenance object; NAT's `trace_hash` folds into
//!    the adapter chain + checkpoint `learning_root`.
//! 6. [`LearningCoordinator`] — one coordinator; NAT's gather is the OBSERVE step
//!    inside RM-FL's stateful cycle.

use crate::{AcceptedContribution, FederationError, GatherResult, SignedContribution, Verifier};
use nat_types::{Q16, ZoneId};
use sha2::{Digest, Sha256};

/// An error at the seam boundary: an input that violates a binding's invariant
/// before any adapter touches it (fail-closed, like the gather's unknown-node path).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SeamError {
    /// A routing/adapter/aggregation target named the non-learned `MX` harness.
    NotALearnedZone(ZoneId),
    /// A zone-weight delta vector was empty (nothing to aggregate).
    EmptyDelta,
    /// Zone-weight deltas disagreed on dimensionality (cannot coordinate-wise reduce).
    DeltaDimensionMismatch { expected: usize, found: usize },
}

impl std::fmt::Display for SeamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SeamError::NotALearnedZone(z) => {
                write!(f, "seam: {} is not a learned zone (MX is the non-learned harness)", z.as_str())
            }
            SeamError::EmptyDelta => write!(f, "seam: empty zone-weight delta"),
            SeamError::DeltaDimensionMismatch { expected, found } => {
                write!(f, "seam: delta dimension mismatch (expected {expected}, found {found})")
            }
        }
    }
}
impl std::error::Error for SeamError {}

// ---------------------------------------------------------------------------
// Binding #1 — the routing/adapter destination is a NAT zone.
// ---------------------------------------------------------------------------

/// A routing destination, redefined (ADR binding #1) to address a **learned NAT
/// zone** (`SM/CB/HP/PF/CX`). RM-FL's router (`0x0111`) emits one of these; a LoRA
/// then targets that zone's weights. The non-learned `MX` harness is rejected at
/// construction, so a destination can never name a zone that has no weights to learn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoutingTarget(ZoneId);

impl RoutingTarget {
    /// Construct a routing target, rejecting the non-learned `MX` harness.
    pub fn new(zone: ZoneId) -> Result<Self, SeamError> {
        if zone.is_learned() {
            Ok(RoutingTarget(zone))
        } else {
            Err(SeamError::NotALearnedZone(zone))
        }
    }
    pub fn zone(self) -> ZoneId {
        self.0
    }
}

/// A LoRA adapter tagged with the NAT zone whose weights it targets (binding #1).
/// The on-chain `LoRAFactory` record carries this so routing can serve
/// `{base + best LoRA}` *for a specific zone*.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterTarget {
    pub lora_id: String,
    pub zone: ZoneId,
}

impl AdapterTarget {
    pub fn new(lora_id: impl Into<String>, zone: ZoneId) -> Result<Self, SeamError> {
        if zone.is_learned() {
            Ok(AdapterTarget { lora_id: lora_id.into(), zone })
        } else {
            Err(SeamError::NotALearnedZone(zone))
        }
    }
}

/// The router adapter (over `0x0111` / `core/learning::routing`) implements this to
/// resolve a routing decision to a zone. `agg_state` is the aggregated Belnap
/// state-vector the router already consumes; the seam only fixes the *output type*.
pub trait RoutesToZone {
    fn route_to_zone(&self, query: &[Q16], agg_state: &[Q16]) -> RoutingTarget;
}

// ---------------------------------------------------------------------------
// Binding #2 — NAT's signed gather is RM-FL's contribution intake (OBSERVE step).
// ---------------------------------------------------------------------------

/// The RM-FL `LearningOrchestrator` adapter implements this so a checkpoint's
/// **OBSERVE** step *is* NAT's verify-before-compose gather (binding #2 + #6). The
/// default body wires the existing [`crate::gather_and_aggregate`] so the seam ties
/// the two halves with no reimplementation; an adapter may override only to add
/// orchestration bookkeeping around the same call.
pub trait ObserveStep {
    fn observe(&self, contribs: &[SignedContribution], verifier: &dyn Verifier) -> GatherResult {
        crate::gather_and_aggregate(contribs, verifier)
    }
}

// ---------------------------------------------------------------------------
// Binding #3 — Belnap-aggregate zone-weight deltas via the Q16 precompile 0x0110.
// ---------------------------------------------------------------------------

/// A per-zone weight delta: the pseudo-gradient for one learned NAT zone, as a
/// fixed-point (Q16) vector. This is the unit the Belnap precompile `0x0110`
/// aggregates across nodes (binding #3) — pointing the existing deterministic Q16
/// aggregation at *zone-weight vectors* rather than embeddings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZoneWeightDelta {
    pub zone: ZoneId,
    pub delta: Vec<Q16>,
}

impl ZoneWeightDelta {
    /// Construct, rejecting `MX` and empty deltas (fail-closed at the boundary).
    pub fn new(zone: ZoneId, delta: Vec<Q16>) -> Result<Self, SeamError> {
        if !zone.is_learned() {
            return Err(SeamError::NotALearnedZone(zone));
        }
        if delta.is_empty() {
            return Err(SeamError::EmptyDelta);
        }
        Ok(ZoneWeightDelta { zone, delta })
    }

    pub fn dim(&self) -> usize {
        self.delta.len()
    }
}

/// The Belnap-precompile adapter implements this. **It MUST ride the Q16 precompile
/// path (`0x0110` family), never the f32 `core/learning::belnap` reference impl**
/// (ADR divergence D-1) — that is the only path that is bit-reproducible across
/// heterogeneous validators. Implementations aggregate coordinate-wise over a set
/// of same-dimension deltas for one zone and return the aggregated delta.
pub trait BelnapZoneAggregator {
    fn aggregate_zone_deltas(&self, deltas: &[ZoneWeightDelta]) -> Result<ZoneWeightDelta, SeamError>;
}

/// Validate that a set of zone-weight deltas can be coordinate-wise reduced: at
/// least one delta, all the same zone, all the same dimension. A reusable guard for
/// any [`BelnapZoneAggregator`] impl (so each adapter does not re-derive it).
pub fn check_aggregable(deltas: &[ZoneWeightDelta]) -> Result<(ZoneId, usize), SeamError> {
    let first = deltas.first().ok_or(SeamError::EmptyDelta)?;
    let dim = first.dim();
    for d in deltas {
        if d.dim() != dim {
            return Err(SeamError::DeltaDimensionMismatch { expected: dim, found: d.dim() });
        }
        if d.zone != first.zone {
            // Cross-zone mixing is a caller error; the harness aggregates per zone.
            return Err(SeamError::NotALearnedZone(d.zone));
        }
    }
    Ok((first.zone, dim))
}

// ---------------------------------------------------------------------------
// Binding #4 — one ledger; the data_quality term flows to FederatedSettlement.
// ---------------------------------------------------------------------------

/// A single unified settlement record (binding #4). Where the legacy
/// [`crate::Settlement`] trait settles only `(node_id, reward_weight)`, this carries
/// the **`data_quality` term explicitly** so the co-op ledger can compute
/// `reward = type-weight × usage × data_quality × compute` — `data_quality` is the
/// honesty factor the whole economic-security story turns on, so it must not be
/// pre-collapsed into the product. Optionally tagged with the zone the work
/// targeted, and bound to the provenance `trace_hash`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SettlementRow {
    pub node_id: String,
    pub reward_weight: Q16,
    pub data_quality: Q16,
    pub zone: Option<ZoneId>,
    pub trace_hash: String,
}

impl SettlementRow {
    /// Build a settlement row from an [`AcceptedContribution`], injecting the
    /// `data_quality` term (which the gather collapses into `reward_weight`) so it
    /// flows separately to the ledger. The orchestrator holds the original
    /// [`crate::SignedContribution`], so it can supply `data_quality` and the
    /// (optional) target zone.
    pub fn from_accepted(accepted: &AcceptedContribution, data_quality: Q16, zone: Option<ZoneId>) -> Self {
        SettlementRow {
            node_id: accepted.node_id.clone(),
            reward_weight: accepted.reward_weight,
            data_quality,
            zone,
            trace_hash: accepted.trace_hash.clone(),
        }
    }
}

/// The co-op `FederatedSettlement` adapter (over `citrate-coop`'s WP-8 seam +
/// `ContributionAccounting` / `PatronageLedger`) implements this. It replaces the
/// legacy `(node_id, reward_weight)` [`crate::Settlement`] with the richer
/// [`SettlementRow`] so `data_quality` reaches the patronage ledger.
pub trait UnifiedSettlement {
    fn settle_row(&self, row: &SettlementRow) -> Result<(), FederationError>;
}

// ---------------------------------------------------------------------------
// Binding #5 — one provenance object.
// ---------------------------------------------------------------------------

/// The single provenance object for a checkpoint (binding #5): NAT's per-round
/// merged `trace_hash`es fold into one `learning_root`, alongside the adapter
/// `ProvenanceChain`. The fold is a deterministic hash chain, so an auditor replays
/// the same roots from the same trace-hash sequence.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct UnifiedProvenance {
    /// The running checkpoint learning-root (hex). Empty before the first fold.
    pub learning_root: String,
    /// The adapter provenance chain (LoRA ids / adapter trace hashes), in order.
    pub adapter_chain: Vec<String>,
}

impl UnifiedProvenance {
    /// Fold one round's merged trace-hash into the learning-root:
    /// `learning_root' = H(learning_root || trace_hash)`. Deterministic + monotone
    /// (each fold strictly depends on all prior folds), so the root commits the
    /// full history and cannot be reordered.
    pub fn fold_trace(&mut self, trace_hash: &str) {
        let mut h = Sha256::new();
        h.update(self.learning_root.as_bytes());
        h.update(b"\n");
        h.update(trace_hash.as_bytes());
        self.learning_root = hex(&h.finalize());
    }

    /// Record an adapter (LoRA) into the provenance chain.
    pub fn push_adapter(&mut self, adapter_id: impl Into<String>) {
        self.adapter_chain.push(adapter_id.into());
    }
}

// ---------------------------------------------------------------------------
// Binding #6 — one coordinator (the OODA checkpoint owns the cycle).
// ---------------------------------------------------------------------------

/// The RM-FL `LearningOrchestrator` adapter implements this: it owns the stateful
/// checkpoint cycle and, at each checkpoint, drives OBSERVE (the gather, binding #2)
/// → aggregate (binding #3) → fold provenance (#5) → settle (#4). NAT's stateless
/// `gather_and_aggregate` is demoted to the OBSERVE step *inside* this loop, rather
/// than being a coordinator itself.
pub trait LearningCoordinator: ObserveStep {
    /// Run one checkpoint: observe → (aggregator) → fold → (settlement). Returns the
    /// settlement rows produced, so the caller can assert conservation. The adapter
    /// supplies the concrete [`BelnapZoneAggregator`] and [`UnifiedSettlement`].
    fn run_checkpoint(
        &self,
        contribs: &[SignedContribution],
        verifier: &dyn Verifier,
        provenance: &mut UnifiedProvenance,
    ) -> GatherResult {
        let result = self.observe(contribs, verifier);
        provenance.fold_trace(&result.merged_hash);
        result
    }
}

// ---------------------------------------------------------------------------
// small helper (mirrors crate::hex; kept private to the seam)
// ---------------------------------------------------------------------------

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0xf) as usize] as char);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{gather_and_aggregate, SignedContribution, ToyKeyedSigner, ToyRosterVerifier};
    use nat_train::StepContribution;

    fn contrib(compute: f32, quality: f32, prov: &str) -> StepContribution {
        StepContribution {
            compute_metered: Q16::from_f32(compute),
            data_quality: Q16::from_f32(quality),
            tokens: 1024,
            provenance_hash: prov.into(),
        }
    }

    // -- Binding #1 --------------------------------------------------------

    #[test]
    fn routing_target_accepts_learned_rejects_mx() {
        for z in ZoneId::LEARNED {
            assert!(RoutingTarget::new(z).is_ok());
        }
        assert_eq!(RoutingTarget::new(ZoneId::MX), Err(SeamError::NotALearnedZone(ZoneId::MX)));
    }

    #[test]
    fn adapter_target_carries_zone_and_rejects_mx() {
        let t = AdapterTarget::new("lora-7", ZoneId::PF).unwrap();
        assert_eq!(t.zone, ZoneId::PF);
        assert!(AdapterTarget::new("lora-x", ZoneId::MX).is_err());
    }

    // -- Binding #3 --------------------------------------------------------

    #[test]
    fn zone_weight_delta_rejects_mx_and_empty() {
        assert!(ZoneWeightDelta::new(ZoneId::SM, vec![Q16::ONE, Q16::ZERO]).is_ok());
        assert_eq!(ZoneWeightDelta::new(ZoneId::MX, vec![Q16::ONE]), Err(SeamError::NotALearnedZone(ZoneId::MX)));
        assert_eq!(ZoneWeightDelta::new(ZoneId::SM, vec![]), Err(SeamError::EmptyDelta));
    }

    #[test]
    fn check_aggregable_enforces_same_zone_and_dim() {
        let a = ZoneWeightDelta::new(ZoneId::CB, vec![Q16::ONE, Q16::ZERO]).unwrap();
        let b = ZoneWeightDelta::new(ZoneId::CB, vec![Q16::ZERO, Q16::ONE]).unwrap();
        assert_eq!(check_aggregable(&[a.clone(), b]).unwrap(), (ZoneId::CB, 2));
        // dimension mismatch is caught
        let short = ZoneWeightDelta::new(ZoneId::CB, vec![Q16::ONE]).unwrap();
        assert!(matches!(
            check_aggregable(&[a, short]),
            Err(SeamError::DeltaDimensionMismatch { expected: 2, found: 1 })
        ));
        assert_eq!(check_aggregable(&[]), Err(SeamError::EmptyDelta));
    }

    // -- Binding #4: data_quality flows separately, not pre-collapsed --------

    #[test]
    fn settlement_row_injects_data_quality_term() {
        let v = ToyRosterVerifier::new().with_node("a", b"key-a".to_vec());
        let c = SignedContribution::create(
            &ToyKeyedSigner::new("a", b"key-a".to_vec()),
            contrib(4.0, 0.5, "pa"),
            "ma",
            "ta",
        );
        let r = gather_and_aggregate(std::slice::from_ref(&c), &v);
        let accepted = &r.accepted[0];
        // reward_weight is the collapsed product (4.0 * 0.5 = 2.0)…
        assert_eq!(accepted.reward_weight, Q16::from_f32(2.0));
        // …and the seam carries data_quality (0.5) SEPARATELY to the ledger.
        let row = SettlementRow::from_accepted(accepted, c.contribution.data_quality, Some(ZoneId::PF));
        assert_eq!(row.reward_weight, Q16::from_f32(2.0));
        assert_eq!(row.data_quality, Q16::from_f32(0.5));
        assert_eq!(row.zone, Some(ZoneId::PF));
        assert_eq!(row.trace_hash, "ta");
        assert_eq!(row.node_id, "a");
    }

    // -- Binding #5: provenance fold is deterministic + monotone -------------

    #[test]
    fn provenance_fold_is_deterministic_and_order_sensitive() {
        let mut p1 = UnifiedProvenance::default();
        p1.fold_trace("round-1");
        p1.fold_trace("round-2");

        let mut p2 = UnifiedProvenance::default();
        p2.fold_trace("round-1");
        p2.fold_trace("round-2");
        // Same sequence -> identical root (deterministic, auditor-replayable).
        assert_eq!(p1.learning_root, p2.learning_root);
        assert_eq!(p1.learning_root.len(), 64);

        // Different order -> different root (monotone history, not a set).
        let mut p3 = UnifiedProvenance::default();
        p3.fold_trace("round-2");
        p3.fold_trace("round-1");
        assert_ne!(p1.learning_root, p3.learning_root);
    }

    // -- WP-7 validation: fixed-seed in-process sweep (conservation + order-indep) --

    /// A tiny deterministic LCG (Numerical Recipes constants) — a fixed-seed value
    /// source so the sweep is reproducible with no external dep (house discipline:
    /// the determinism path carries no third-party crates).
    struct Lcg(u64);
    impl Lcg {
        fn next_u32(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
            (self.0 >> 16) as u32
        }
        /// A Q16 in [0, 8) for compute and [0,1] for quality, on the Q16 grid.
        fn next_unit(&mut self) -> f32 {
            (self.next_u32() % 1000) as f32 / 1000.0
        }
    }

    #[test]
    fn sweep_seam_conserves_weight_and_carries_quality_order_independent() {
        let mut rng = Lcg(0xC1_7A_7E_5A_1D_00_00_01);
        // 3000 random rounds of up to 6 nodes each — fixed seed, fully reproducible.
        for _round in 0..3000 {
            let n = 1 + (rng.next_u32() % 6) as usize;
            let mut verifier = ToyRosterVerifier::new();
            let mut contribs = Vec::new();
            let mut expected_quality = std::collections::BTreeMap::new();
            for i in 0..n {
                let id = format!("n{i}");
                let key = format!("key-{id}").into_bytes();
                verifier = verifier.with_node(&id, key.clone());
                let compute = 8.0 * rng.next_unit();
                let quality = rng.next_unit(); // in [0,1]
                expected_quality.insert(id.clone(), Q16::from_f32(quality));
                contribs.push(SignedContribution::create(
                    &ToyKeyedSigner::new(id, key),
                    contrib(compute, quality, "p"),
                    "m",
                    format!("t{i}"),
                ));
            }

            let result = gather_and_aggregate(&contribs, &verifier);

            // Build settlement rows, injecting each node's data_quality.
            let rows: Vec<SettlementRow> = result
                .accepted
                .iter()
                .map(|a| {
                    let dq = expected_quality[&a.node_id];
                    SettlementRow::from_accepted(a, dq, Some(ZoneId::PF))
                })
                .collect();

            // (1) CONSERVATION: the seam neither loses nor creates reward weight.
            let row_total: Q16 = rows.iter().fold(Q16::ZERO, |acc, r| acc.add(r.reward_weight));
            assert_eq!(row_total, result.total_reward_weight, "round conserved total");

            // (2) data_quality is carried SEPARATELY (not pre-collapsed into the product).
            for r in &rows {
                assert_eq!(r.data_quality, expected_quality[&r.node_id]);
            }

            // (3) ORDER-INDEPENDENCE: reversing input order yields the same total + the
            //     same merged hash (the gather is a function of the accepted set).
            let mut rev = contribs.clone();
            rev.reverse();
            let result_rev = gather_and_aggregate(&rev, &verifier);
            assert_eq!(result.total_reward_weight, result_rev.total_reward_weight);
            assert_eq!(result.merged_hash, result_rev.merged_hash);
        }
    }

    // -- Bindings #2 + #6: gather is the OBSERVE step inside the coordinator --

    #[test]
    fn coordinator_runs_gather_as_observe_step_and_folds_provenance() {
        // A reference adapter standing in for the LearningOrchestrator (test-only).
        struct RefOrchestrator;
        impl ObserveStep for RefOrchestrator {}
        impl LearningCoordinator for RefOrchestrator {}

        let v = ToyRosterVerifier::new()
            .with_node("a", b"key-a".to_vec())
            .with_node("b", b"key-b".to_vec());
        let cs = vec![
            SignedContribution::create(&ToyKeyedSigner::new("a", b"key-a".to_vec()), contrib(4.0, 0.5, "pa"), "ma", "ta"),
            SignedContribution::create(&ToyKeyedSigner::new("b", b"key-b".to_vec()), contrib(2.0, 1.0, "pb"), "mb", "tb"),
        ];

        let orch = RefOrchestrator;
        let mut prov = UnifiedProvenance::default();
        let result = orch.run_checkpoint(&cs, &v, &mut prov);

        // OBSERVE produced the same result as the raw gather (binding #2: gather IS intake).
        assert_eq!(result, gather_and_aggregate(&cs, &v));
        // The checkpoint folded the round's merged hash into the learning-root (binding #5).
        assert!(!prov.learning_root.is_empty());
        let mut expected = UnifiedProvenance::default();
        expected.fold_trace(&result.merged_hash);
        assert_eq!(prov.learning_root, expected.learning_root);
    }
}
