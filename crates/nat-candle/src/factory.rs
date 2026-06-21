//! The Candle core backend, injectable into `NatModel`.
//!
//! `nat-core` cannot depend on `nat-candle` (it would cycle), so the Candle
//! [`CoreFactory`] lives here and is handed to the model via
//! [`nat_core::NatModel::with_cores`]. [`candle_model`] is the convenience
//! constructor a real run uses instead of the toy-core `NatModel::l0`.

use crate::cores::{CandleAttentionCore, CandleSsmCore};
use nat_core::cores::{CoreFactory, ZoneCore};
use nat_core::NatModel;
use nat_sidecar::Sidecar;
use nat_types::{CoreType, ZoneId};

/// The Candle (CPU) core backend. `is_toy()` is false, so a run using this
/// backend records `backend = "candle-cpu"` in its trace and passes the
/// non-toy assertion the L1/DGX path makes.
pub struct CandleCores;

impl CoreFactory for CandleCores {
    fn core_for(&self, zone: ZoneId) -> Box<dyn ZoneCore> {
        match zone.default_core() {
            CoreType::Ssm => Box::new(CandleSsmCore::default()),
            CoreType::Attention => Box::new(CandleAttentionCore::default()),
            CoreType::None => unreachable!("MX has no learned core; never built here"),
        }
    }
    fn backend(&self) -> &str {
        "candle-cpu"
    }
    fn is_toy(&self) -> bool {
        false
    }
}

/// A `NatModel` running the Candle backend over the given sidecar. This is the
/// real-core counterpart to `NatModel::l0` (which uses toy cores).
pub fn candle_model(sidecar: Sidecar) -> NatModel {
    NatModel::with_cores(sidecar, Box::new(CandleCores))
}

/// The default L0 sidecar wired to the Candle backend.
pub fn candle_model_l0() -> NatModel {
    candle_model(Sidecar::default_l0())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candle_model_runs_a_forward_pass_with_real_cores() {
        let model = candle_model_l0();
        // The guarantee the DGX path depends on: this is NOT the toy backend.
        assert!(!model.uses_toy_cores());
        assert_eq!(model.backend(), "candle-cpu");

        let r = model.forward("compute 12 * 7 + 3 and explain", None);
        // The trace records the real backend, so an auditor can verify no toy fallback.
        assert_eq!(r.trace.backend, "candle-cpu");
        // The pass still produces a complete, decision-faithful trace.
        assert!(nat_provenance::verify_decision_faithful(&r.trace));
        assert!(!r.output.output_hash.is_empty());
    }

    #[test]
    fn toy_and_candle_backends_are_distinguishable_in_the_trace() {
        let toy = NatModel::l0().forward("hello world", None);
        let candle = candle_model_l0().forward("hello world", None);
        assert_eq!(toy.trace.backend, "toy-l0");
        assert_eq!(candle.trace.backend, "candle-cpu");
        assert_ne!(toy.trace.backend, candle.trace.backend);
    }
}
