//! The NAT eval harness (Data Ops §7). Built fully at L1; the metrics that are
//! *measurable at L0* live here now so the harness exists before it is needed.
//!
//! The bet-deciding metric, capability-per-parameter (H-01), cannot be measured
//! at L0 (nothing is trained). What *is* measurable at L0 is the structural
//! plumbing: routing differentiation (H-02) and provenance faithfulness (H-03).

use nat_core::NatModel;
use nat_provenance::verify_decision_faithful;
use nat_types::ZoneId;

/// Routing-differentiation score between two prompt classes (H-02): how much the
/// dominant zone activation differs. 0 = identical mixes, higher = more
/// differentiated. At L0 this previews the metric; at L1 it runs over labeled
/// prompt-class batteries against a significance threshold.
pub fn routing_divergence(model: &NatModel, prompt_a: &str, prompt_b: &str) -> f32 {
    let a = model.forward(prompt_a, None).trace.router.zone_activation;
    let b = model.forward(prompt_b, None).trace.router.zone_activation;
    // L1 distance over the activation vectors (in f32, for a human-readable score).
    ZoneId::ALL
        .iter()
        .map(|z| {
            let av = a.iter().find(|(id, _)| id == z).unwrap().1.to_f32();
            let bv = b.iter().find(|(id, _)| id == z).unwrap().1.to_f32();
            (av - bv).abs()
        })
        .sum()
}

/// Provenance faithfulness (H-03), decision-faithful sense: every pass's trace
/// recomputes its own merge decision from its recorded scores. Returns true iff
/// faithful for all sample prompts.
pub fn faithfulness_holds(model: &NatModel, prompts: &[&str]) -> bool {
    prompts
        .iter()
        .all(|p| verify_decision_faithful(&model.forward(p, None).trace))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn math_and_narrative_prompts_diverge_in_routing() {
        let m = NatModel::l0();
        let d = routing_divergence(&m, "12 * 7 + 3 = ?", "she walked the quiet shore at dawn");
        assert!(d > 0.0, "routing did not differentiate by class (d={d})");
    }

    #[test]
    fn traces_are_decision_faithful() {
        let m = NatModel::l0();
        assert!(faithfulness_holds(
            &m,
            &["2 + 2", "tell me a story", "fn main() {}"]
        ));
    }
}
