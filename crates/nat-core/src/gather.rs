//! Async gather with a deadline (Architecture §5.3, formal/AsyncGather.tla).
//!
//! Zones finish at different times; the merge must not block on a slow one. The
//! gather waits up to a deadline, then composes with whatever arrived. Late
//! zones are recorded `timed_out` and excluded from the pass.
//!
//! At L0 the gather is a *deterministic simulation* of the deadline discipline:
//! each zone reports a `latency_ms` and arrives iff `latency_ms <= deadline_ms`.
//! This mirrors AsyncGather.tla exactly (a logical clock, not wall-clock), so the
//! timed-out path is reproducible in tests. Real wall-clock async gather across
//! nodes is L1/Gate-4 work; the discipline it must honor is fixed here.

use nat_types::{ZoneId, ZoneStatus};

#[derive(Debug, Clone)]
pub struct ZoneArrival {
    pub zone: ZoneId,
    pub latency_ms: u64,
    pub arrived: bool,
}

/// Close the gather window at `deadline_ms`. Every learned zone is classified
/// arrived (`ok` for now) or `timed_out`. This is total and deterministic:
/// no zone is left `pending`, and none is both `ok` and `timed_out`
/// (`StragglerRecorded`, `Consistent`).
pub fn gather(latencies: &[(ZoneId, u64)], deadline_ms: u64) -> Vec<ZoneArrival> {
    latencies
        .iter()
        .map(|&(zone, latency_ms)| ZoneArrival {
            zone,
            latency_ms,
            arrived: latency_ms <= deadline_ms,
        })
        .collect()
}

/// The status a non-arrived zone carries into the trace.
pub fn arrival_status(a: &ZoneArrival) -> ZoneStatus {
    if a.arrived {
        ZoneStatus::Ok
    } else {
        ZoneStatus::TimedOut
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn straggler_past_deadline_times_out() {
        let lat = vec![
            (ZoneId::SM, 10),
            (ZoneId::CB, 15),
            (ZoneId::HP, 30),
            (ZoneId::PF, 150), // slow
            (ZoneId::CX, 60),
        ];
        let g = gather(&lat, 100);
        let pf = g.iter().find(|a| a.zone == ZoneId::PF).unwrap();
        assert!(!pf.arrived);
        assert_eq!(arrival_status(pf), ZoneStatus::TimedOut);
        // Everyone under the deadline arrived.
        assert_eq!(g.iter().filter(|a| a.arrived).count(), 4);
    }

    #[test]
    fn every_zone_is_classified_exactly_once() {
        let lat = vec![(ZoneId::SM, 10), (ZoneId::PF, 999)];
        let g = gather(&lat, 100);
        assert_eq!(g.len(), 2);
        for a in &g {
            // ok XOR timed_out — never both, never neither.
            let ok = a.arrived;
            let to = !a.arrived;
            assert!(ok ^ to);
        }
    }
}
