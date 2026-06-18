//! Gate-2 acceptance: async gather with a deadline.
//!
//! Maps to `features/gate2_async_gather.feature` (Formal Scaffold §B.2,
//! AsyncGather.tla). The merge must never block on a slow zone.

use nat_core::NatModel;
use nat_types::{ZoneId, ZoneStatus};

/// Scenario: Gather closes at the deadline.
#[test]
fn slow_zone_times_out_and_merge_composes_from_arrivals() {
    let mut model = NatModel::l0();
    // PF will not return before the deadline (default deadline is 100ms).
    model.set_latency(ZoneId::PF, 150);

    let r = model.forward("a reasoning-heavy prompt that leans on PF", None);

    // PF is recorded with status timed_out.
    let pf = r.trace.zones.iter().find(|z| z.id == ZoneId::PF).unwrap();
    assert_eq!(pf.status, ZoneStatus::TimedOut);

    // The merge composed from the zones that arrived — PF is not a survivor, and
    // an output was still produced.
    assert!(!r.trace.merge.survivors.contains(&ZoneId::PF));
    assert!(!r.output.output_hash.is_empty());
}

/// Scenario: Straggler completeness.
#[test]
fn every_zone_has_exactly_one_status_and_no_ok_timed_out_overlap() {
    let mut model = NatModel::l0();
    model.set_latency(ZoneId::PF, 150);
    model.set_latency(ZoneId::CX, 200);

    let r = model.forward("force two stragglers", None);

    // Each non-arrived zone is timed_out; none is both ok and timed_out (the
    // status is a single enum value, which makes the overlap structurally
    // impossible — assert the partition explicitly anyway).
    let timed_out: Vec<ZoneId> = r
        .trace
        .zones
        .iter()
        .filter(|z| z.status == ZoneStatus::TimedOut)
        .map(|z| z.id)
        .collect();
    assert!(timed_out.contains(&ZoneId::PF));
    assert!(timed_out.contains(&ZoneId::CX));

    for z in &r.trace.zones {
        // ok | timed_out | pruned — exactly one, never two.
        let states = [
            z.status == ZoneStatus::Ok,
            z.status == ZoneStatus::TimedOut,
            z.status == ZoneStatus::Pruned,
        ];
        assert_eq!(states.iter().filter(|b| **b).count(), 1, "{:?}", z.id);
    }
}
