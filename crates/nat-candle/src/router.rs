//! WP-3 — the learned router gate (NAT-S2, ADR-0001, H-02).
//!
//! The L0 router (`nat_core::router::route`) is a hand-wired map from class
//! signals to per-zone activations. ADR-0001 says L1 replaces it with a *learned*
//! gate. This is that gate: a small trainable network mapping the input features
//! to per-zone activations, and — crucially — it can only ever weight **declared**
//! topology edges. That impossibility is structural (the C-1 auditability claim):
//! the router copies the sidecar's declared edge list at construction and iterates
//! nothing else, so an undeclared edge has no code path to receive a strength.
//!
//! H-02 (does context-aware routing differentiate prompt classes?) is measured by
//! `nat_eval::separation_ratio` over the activation vectors — the trained gate is
//! scored by the same yardstick as the L0 model (that comparison lives in
//! `nat-eval`'s tests, which depend on this crate).

use crate::seed::seeded_linear;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::{loss, ops, Linear, Module, Optimizer, VarBuilder, VarMap};
use nat_sidecar::Sidecar;
use nat_types::ZoneId;

/// Input feature width: the class signals (math, narrative, code, sensory).
pub const FEATURE_DIM: usize = 4;

/// A learned routing gate: features → per-zone activations, modulating a fixed
/// topology. Trainable; the drop-in L1 replacement for the hand-wired L0 router.
pub struct LearnedRouter {
    varmap: VarMap,
    gate1: Linear, // FEATURE_DIM -> hidden
    gate2: Linear, // hidden -> n_zones
    zones: Vec<ZoneId>,
    /// DECLARED edges only, copied from the sidecar at construction. The router
    /// has no other source of edges — the C-1 invariant, by construction.
    edges: Vec<(ZoneId, ZoneId)>,
    seed: u64,
    device: Device,
}

impl LearnedRouter {
    /// The H-02 router: all five learned zones, class-signal features.
    pub fn new(sidecar: &Sidecar, hidden: usize, seed: u64) -> Result<Self> {
        Self::with_zones(sidecar, &ZoneId::LEARNED, FEATURE_DIM, hidden, seed)
    }

    /// A router over a specific zone set and feature width — so the gate can align
    /// to a zone subset (e.g. the 3-zone {HP,PF,CX}, ADR-0008) and read learned
    /// features (e.g. a pooled embedding) instead of the L0 class signals. Edges
    /// are the declared edges **among these zones** — still copied from the
    /// sidecar, so the declared-edges invariant holds for any subset.
    pub fn with_zones(
        sidecar: &Sidecar,
        zones: &[ZoneId],
        feature_dim: usize,
        hidden: usize,
        seed: u64,
    ) -> Result<Self> {
        let dev = crate::device::device();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let zones = zones.to_vec();
        let gate1 = seeded_linear(&varmap, &vb, "gate1", feature_dim, hidden, seed, &dev)?;
        let gate2 = seeded_linear(&varmap, &vb, "gate2", hidden, zones.len(), seed, &dev)?;
        let edges = sidecar
            .topology
            .edges
            .iter()
            .filter(|e| zones.contains(&e.from) && zones.contains(&e.to))
            .map(|e| (e.from, e.to))
            .collect();
        Ok(LearnedRouter {
            varmap,
            gate1,
            gate2,
            zones,
            edges,
            seed,
            device: dev,
        })
    }

    /// The router's parameter map (for the optimizer / checkpointing).
    pub fn varmap(&self) -> &VarMap {
        &self.varmap
    }

    /// Mutable parameter map (for loading a checkpoint).
    pub fn varmap_mut(&mut self) -> &mut VarMap {
        &mut self.varmap
    }

    /// Per-zone activations in `[0, 1]`: `sigmoid(gate2(relu(gate1(features))))`.
    /// `features`: `(batch, FEATURE_DIM)` → `(batch, n_zones)`. Differentiable.
    pub fn activations(&self, features: &Tensor) -> Result<Tensor> {
        let h = self.gate1.forward(features)?.relu()?;
        ops::sigmoid(&self.gate2.forward(&h)?)
    }

    /// Edge strengths over the **declared** edges only: `strength(from→to) =
    /// act[from] · act[to]`, one per declared edge in declared order. There is no
    /// code path that can produce a strength for an undeclared edge.
    pub fn edge_strengths(&self, activations: &Tensor) -> Result<Vec<(ZoneId, ZoneId, Tensor)>> {
        let col = |z: ZoneId| -> Option<usize> { self.zones.iter().position(|&zz| zz == z) };
        let mut out = Vec::with_capacity(self.edges.len());
        for &(from, to) in &self.edges {
            // MX (the non-learned harness) is not in the learned-zone activation
            // vector; an edge touching it carries no learned strength.
            let (fi, ti) = match (col(from), col(to)) {
                (Some(fi), Some(ti)) => (fi, ti),
                _ => continue,
            };
            let a_from = activations.narrow(1, fi, 1)?;
            let a_to = activations.narrow(1, ti, 1)?;
            out.push((from, to, a_from.mul(&a_to)?));
        }
        Ok(out)
    }

    /// The declared edges this router modulates (for the invariant test).
    pub fn declared_edges(&self) -> &[(ZoneId, ZoneId)] {
        &self.edges
    }

    /// The learned zones, in activation-vector order.
    pub fn zones(&self) -> &[ZoneId] {
        &self.zones
    }

    /// The activation vector for a single prompt: featurize → activations → vec.
    /// Length is `n_zones` (the learned zones, canonical order).
    pub fn activation_vec(&self, prompt: &str) -> Result<Vec<f32>> {
        let f = features_of(prompt, &self.device)?;
        self.activations(&f)?.flatten_all()?.to_vec1::<f32>()
    }

    /// Train the gate to make activations class-discriminative: a temporary
    /// classifier head reads the activation vector and predicts the class, so the
    /// gate learns a routing that separates the classes (H-02). The head is a
    /// training scaffold — it is discarded; only the gate is kept. Returns the
    /// final cross-entropy loss.
    pub fn train_to_classify(
        &mut self,
        features: &Tensor,
        labels: &Tensor,
        n_classes: usize,
        steps: usize,
        lr: f64,
    ) -> Result<f32> {
        let head_varmap = VarMap::new();
        let head_vb = VarBuilder::from_varmap(&head_varmap, DType::F32, &self.device);
        let head = seeded_linear(
            &head_varmap,
            &head_vb,
            "cls",
            self.zones.len(),
            n_classes,
            self.seed ^ 0xC1A5,
            &self.device,
        )?;

        let mut vars = self.varmap.all_vars();
        vars.extend(head_varmap.all_vars());
        let mut opt = AdamW::new(
            vars,
            ParamsAdamW {
                lr,
                ..Default::default()
            },
        )?;

        let mut final_loss = 0.0;
        for _ in 0..steps {
            let acts = self.activations(features)?;
            let logits = head.forward(&acts)?;
            let l = loss::cross_entropy(&logits, labels)?;
            opt.backward_step(&l)?;
            final_loss = l.to_scalar::<f32>()?;
        }
        Ok(final_loss)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// Build the class-signal feature tensor for a prompt (the gate's input).
pub fn features_of(prompt: &str, dev: &Device) -> Result<Tensor> {
    let s = nat_core::featurize::class_signals(prompt);
    Tensor::from_vec(
        vec![s.math, s.narrative, s.code, s.sensory],
        (1, FEATURE_DIM),
        dev,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_is_trainable() {
        // A loss on the activations produces gradient on every gate parameter.
        let sc = Sidecar::default_l0();
        let router = LearnedRouter::new(&sc, 16, 7).unwrap();
        let f = features_of("compute 2 + 2 step by step", router.device()).unwrap();
        let acts = router.activations(&f).unwrap();
        let l = acts.sqr().unwrap().sum_all().unwrap();
        let grads = l.backward().unwrap();
        for v in router.varmap.all_vars() {
            let g = grads.get(v.as_tensor()).expect("gradient present");
            let s = g
                .abs()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(s.is_finite() && s > 0.0, "vanishing/absent gate gradient");
        }
    }

    #[test]
    fn activations_are_in_unit_range_and_right_width() {
        let sc = Sidecar::default_l0();
        let router = LearnedRouter::new(&sc, 16, 1).unwrap();
        let v = router.activation_vec("she walked the quiet shore").unwrap();
        assert_eq!(v.len(), ZoneId::LEARNED.len());
        assert!(v.iter().all(|&a| (0.0..=1.0).contains(&a)));
    }

    #[test]
    fn router_only_ever_weights_declared_edges() {
        // The C-1 invariant: every edge the router can produce a strength for is a
        // declared topology edge, and an undeclared edge never appears.
        let sc = Sidecar::default_l0();
        let router = LearnedRouter::new(&sc, 8, 3).unwrap();
        let f = features_of("fn main() {}", router.device()).unwrap();
        let acts = router.activations(&f).unwrap();
        let strengths = router.edge_strengths(&acts).unwrap();

        for (from, to, _) in &strengths {
            assert!(
                sc.topology.has_edge(*from, *to),
                "router produced an UNDECLARED edge {from:?}->{to:?}"
            );
        }
        // Every declared (learned-zone) edge is covered, and nothing beyond them.
        let declared_learned = sc
            .topology
            .edges
            .iter()
            .filter(|e| ZoneId::LEARNED.contains(&e.from) && ZoneId::LEARNED.contains(&e.to))
            .count();
        assert_eq!(strengths.len(), declared_learned);
        // A concretely undeclared edge (CX->PF is not in the default topology) is absent.
        assert!(!sc.topology.has_edge(ZoneId::CX, ZoneId::PF));
        assert!(!strengths
            .iter()
            .any(|(f, t, _)| *f == ZoneId::CX && *t == ZoneId::PF));
    }
}
