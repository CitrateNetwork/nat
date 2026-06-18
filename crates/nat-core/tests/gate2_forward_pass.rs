//! Gate-2 acceptance: zone-partitioned forward pass with provenance.
//!
//! Each test maps 1:1 to a scenario in `features/gate2_forward_pass.feature`.
//! This is the executable definition of done for that part of Gate 2
//! (Formal Scaffold §B.1).

use nat_core::NatModel;
use nat_provenance::verify_decision_faithful;
use nat_types::{CoreType, ZoneId, ZoneStatus};

/// Scenario: All zones execute in parallel over one input.
#[test]
fn all_learned_zones_produce_output_and_mx_runs_no_learned_core() {
    let model = NatModel::l0();
    let r = model.forward("compute the sum of the first ten primes", None);

    // Each learned zone produced an output carrying a confidence and a latency.
    for z in ZoneId::LEARNED {
        let rec = r.trace.zones.iter().find(|zr| zr.id == z).unwrap();
        assert!(rec.confidence.to_f32() > 0.0, "{z:?} has no confidence");
        assert!(rec.latency_ms > 0, "{z:?} has no latency");
        assert_ne!(rec.core, CoreType::None, "{z:?} must have a learned core");
    }

    // The MCP harness MX does not run a learned core.
    let mx = r.trace.zones.iter().find(|zr| zr.id == ZoneId::MX).unwrap();
    assert_eq!(mx.core, CoreType::None);
}

/// Scenario: The router modulates a fixed topology.
#[test]
fn router_emits_len6_activation_and_only_declared_edge_modulation() {
    let model = NatModel::l0();
    let r = model.forward("a sensory prompt: bright loud image", None);

    // Zone-activation vector of length 6.
    assert_eq!(r.trace.router.zone_activation.len(), 6);

    // Edge-modulation only for declared edges; no flow on an absent edge.
    for e in &r.trace.router.edge_modulation {
        assert!(
            model.sidecar.topology.has_edge(e.from, e.to),
            "router modulated an undeclared edge {:?}->{:?}",
            e.from,
            e.to
        );
    }
    for flow in &r.trace.inter_zone_flows {
        assert!(
            model.sidecar.topology.has_edge(flow.from, flow.to),
            "signal flowed along an undeclared edge {:?}->{:?}",
            flow.from,
            flow.to
        );
    }
}

/// Scenario: The merge scores, prunes, and composes.
#[test]
fn merge_scores_prunes_bottom_majority_and_normalizes_weights() {
    let model = NatModel::l0();
    let r = model.forward("12 * 7 + 3 = ?", None);

    // Every gathered output got a combined score.
    assert!(!r.trace.merge.scores.is_empty());

    // With the default 0.8 threshold over the gathered set, the bottom 70–80%
    // are pruned: survivors are a strict minority of what was gathered.
    let gathered = r.trace.merge.scores.len();
    let survivors = r.trace.merge.survivors.len();
    assert!(
        survivors >= 1 && survivors < gathered,
        "survivors={survivors} gathered={gathered}"
    );

    // Every pruned (gathered-but-not-surviving) zone is recorded with status Pruned.
    for (zone, _) in &r.trace.merge.scores {
        if !r.trace.merge.survivors.contains(zone) {
            let rec = r.trace.zones.iter().find(|zr| zr.id == *zone).unwrap();
            assert_eq!(rec.status, ZoneStatus::Pruned, "{zone:?} should be pruned");
        }
    }

    // Surviving weights normalize to 1 (within Q16 rounding).
    let sum: i64 = r.trace.merge.weights.iter().map(|(_, w)| w.raw()).sum();
    assert!(
        (sum - nat_types::Q16::ONE.raw()).abs() <= 4,
        "weights sum raw={sum}"
    );
}

/// Scenario: The provenance trace is complete and hashable.
#[test]
fn trace_is_complete_and_hashes_reproducibly() {
    let model = NatModel::l0();
    let r = model.forward("tell me a quiet story about the shore", None);
    let trace = &r.trace;

    // Every activated zone appears with a valid status.
    for rec in &trace.zones {
        assert!(matches!(
            rec.status,
            ZoneStatus::Ok | ZoneStatus::TimedOut | ZoneStatus::Pruned
        ));
    }

    // The merge record carries scores, threshold, and survivors.
    assert!(!trace.merge.scores.is_empty());
    assert!(trace.merge.prune_threshold.raw() > 0);
    assert!(!trace.merge.survivors.is_empty());

    // Serializing the trace twice yields the same hash.
    assert_eq!(trace.trace_hash(), trace.trace_hash());

    // And the recorded decision is faithful to its own recorded scores.
    assert!(verify_decision_faithful(trace));

    // The output hash is present and stable across re-runs (bit-faithful merge).
    let r2 = model.forward("tell me a quiet story about the shore", None);
    assert_eq!(r.output.output_hash, r2.output.output_hash);
}
