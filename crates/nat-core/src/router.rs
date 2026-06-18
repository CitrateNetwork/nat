//! The router (Architecture §5.2): per-input zone activation + edge modulation.
//!
//! The router modulates a **fixed** topology — it produces a strength for each
//! *declared* edge and an activation for each zone, but it cannot create an edge
//! the sidecar did not declare. That impossibility is structural here: we only
//! ever iterate `sidecar.topology.edges`, so an undeclared edge has no code path
//! to receive a weight. This is the property that keeps the system auditable
//! while adaptive (the C-1 claim).
//!
//! At L0 the mapping from class signals to activation is fixed (not trained);
//! L1 replaces it with a learned gate (ADR-0001).

use crate::featurize::ClassSignals;
use nat_sidecar::Sidecar;
use nat_types::ZoneId;

#[derive(Debug, Clone)]
pub struct RouterOutput {
    /// Activation per zone in canonical `ZoneId::ALL` order, each in [0,1].
    pub zone_activation: Vec<(ZoneId, f32)>,
    /// Strength per *declared* edge, each in [0,1]. Length == declared edges.
    pub edge_modulation: Vec<(ZoneId, ZoneId, f32)>,
}

impl RouterOutput {
    pub fn activation_of(&self, z: ZoneId) -> f32 {
        self.zone_activation
            .iter()
            .find(|(id, _)| *id == z)
            .map(|(_, a)| *a)
            .unwrap_or(0.0)
    }
}

/// Compute the router output for one input. Deterministic.
pub fn route(signals: ClassSignals, sidecar: &Sidecar) -> RouterOutput {
    let clamp01 = |x: f32| x.clamp(0.0, 1.0);

    // Fixed L0 activation map (Architecture §5.2 worked examples): a math prompt
    // drives {CB, CX, PF}; a narrative prompt {HP, PF}; a sensory task {SM}.
    let activation_for = |z: ZoneId| -> f32 {
        match z {
            ZoneId::SM => clamp01(signals.sensory),
            ZoneId::CB => clamp01(0.6 * signals.math + 0.3 * signals.sensory),
            ZoneId::HP => clamp01(signals.narrative),
            // PF is the deep reasoner: always somewhat active, more so for math/narrative/code.
            ZoneId::PF => {
                clamp01(0.3 + 0.4 * signals.math + 0.4 * signals.narrative + 0.3 * signals.code)
            }
            ZoneId::CX => clamp01(signals.code + 0.4 * signals.math),
            // MX is the non-learned harness: it always participates.
            ZoneId::MX => 1.0,
        }
    };

    let zone_activation: Vec<(ZoneId, f32)> = ZoneId::ALL
        .iter()
        .map(|&z| (z, activation_for(z)))
        .collect();

    let act = |z: ZoneId| zone_activation.iter().find(|(id, _)| *id == z).unwrap().1;

    // Edge modulation: ONLY for declared edges. An undeclared edge cannot appear
    // here because there is no iteration over anything but the declared set.
    let edge_modulation: Vec<(ZoneId, ZoneId, f32)> = sidecar
        .topology
        .edges
        .iter()
        .map(|e| (e.from, e.to, clamp01(act(e.from) * act(e.to))))
        .collect();

    RouterOutput {
        zone_activation,
        edge_modulation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::featurize::class_signals;

    #[test]
    fn activation_vector_has_length_six() {
        let sc = Sidecar::default_l0();
        let out = route(class_signals("hello"), &sc);
        assert_eq!(out.zone_activation.len(), 6);
    }

    #[test]
    fn edge_modulation_only_for_declared_edges() {
        let sc = Sidecar::default_l0();
        let out = route(class_signals("2 + 2"), &sc);
        assert_eq!(out.edge_modulation.len(), sc.topology.edges.len());
        for (from, to, _) in &out.edge_modulation {
            assert!(sc.topology.has_edge(*from, *to));
        }
    }

    #[test]
    fn different_classes_drive_different_mixes() {
        // A preview of the H-02 / Gate-3 differentiation property.
        let sc = Sidecar::default_l0();
        let math = route(class_signals("12 * 7 + 3 = ?"), &sc);
        let story = route(class_signals("she walked the quiet shore"), &sc);
        // Math lights CX more than the narrative prompt does; narrative lights HP more.
        assert!(math.activation_of(ZoneId::CX) > story.activation_of(ZoneId::CX));
        assert!(story.activation_of(ZoneId::HP) > math.activation_of(ZoneId::HP));
    }
}
