//! The NAT sidecar (Architecture §10.2, ADR-0004).
//!
//! GGUF/ONNX stays the tensor container; this sidecar carries the zone graph,
//! topology, router/merge params, and composition rules. A sidecar-unaware
//! runtime runs the tensors opaquely (the Ollama onramp); a sidecar-aware
//! runtime runs the full zone-partitioned pass.
//!
//! Note (critique remediation #7): "runs opaquely in Ollama" applies to a
//! *flattened/distilled* dense export, not to a literal parallel-heterogeneous
//! zone graph — GGUF has no layout for parallel SSM+attention zones. The sidecar
//! is the source of truth for the zone graph; the GGUF carries whatever tensor
//! form the export target can run. The `export_kind` field records which.

use nat_types::{CoreType, ZoneId};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// The export form the paired tensor container holds (remediation #7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportKind {
    /// Tensors are the literal per-zone weights; only a NAT-aware runtime runs them.
    ZonePartitioned,
    /// Tensors are a flattened dense equivalent that any GGUF loader can run as
    /// an opaque transformer; the sidecar still carries the zone graph for
    /// NAT-aware runtimes and for provenance.
    FlattenedDense,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZoneDecl {
    pub id: ZoneId,
    pub core: CoreType,
    /// Offset of this zone's slice into the shared hidden width `D`.
    pub slice_offset: u32,
    /// Width of this zone's slice.
    pub slice_width: u32,
    pub modalities: Vec<String>,
    /// Reference to the training recipe for this zone (free-form id).
    pub recipe_ref: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Edge {
    pub from: ZoneId,
    pub to: ZoneId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Topology {
    pub edges: Vec<Edge>,
}

impl Topology {
    /// Is `(from, to)` a declared edge? The router may only modulate declared
    /// edges; this is the check that keeps the system auditable (Architecture §5.2).
    pub fn has_edge(&self, from: ZoneId, to: ZoneId) -> bool {
        self.edges.iter().any(|e| e.from == from && e.to == to)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MergeParams {
    /// Fraction dropped by the prune step, in (0,1). Set by L1 ablation, not asserted.
    pub prune_threshold: f32,
    /// The async-gather deadline in (logical) milliseconds.
    pub deadline_ms: u64,
}

/// The sidecar document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sidecar {
    pub version: u32,
    pub export_kind: ExportKind,
    pub zones: Vec<ZoneDecl>,
    pub topology: Topology,
    pub merge: MergeParams,
    /// Composition rules: which zone ids may be swapped (Architecture §10.3). A
    /// zone is swappable if its slice width and cross-zone contract match.
    pub composition_rules: Vec<String>,
}

#[derive(Debug, Error, PartialEq)]
pub enum SidecarError {
    #[error("parse error: {0}")]
    Parse(String),
    #[error("topology edge references undeclared zone {0:?}")]
    UndeclaredZoneInEdge(ZoneId),
    #[error("the non-learned MX zone must not declare a learned core")]
    MxHasLearnedCore,
    #[error("prune_threshold must be in (0,1), got {0}")]
    BadPruneThreshold(f32),
    #[error("zone {0:?} declared more than once")]
    DuplicateZone(ZoneId),
}

impl Sidecar {
    pub fn from_json(s: &str) -> Result<Self, SidecarError> {
        let sc: Sidecar =
            serde_json::from_str(s).map_err(|e| SidecarError::Parse(e.to_string()))?;
        sc.validate()?;
        Ok(sc)
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("sidecar always serializes")
    }

    /// Structural validation. These are the invariants a TLA+ model and an
    /// auditor both read directly from the topology (Architecture §5.1).
    pub fn validate(&self) -> Result<(), SidecarError> {
        let declared: Vec<ZoneId> = self.zones.iter().map(|z| z.id).collect();
        for (i, z) in self.zones.iter().enumerate() {
            if declared[..i].contains(&z.id) {
                return Err(SidecarError::DuplicateZone(z.id));
            }
            if z.id == ZoneId::MX && z.core != CoreType::None {
                return Err(SidecarError::MxHasLearnedCore);
            }
        }
        for e in &self.topology.edges {
            if !declared.contains(&e.from) {
                return Err(SidecarError::UndeclaredZoneInEdge(e.from));
            }
            if !declared.contains(&e.to) {
                return Err(SidecarError::UndeclaredZoneInEdge(e.to));
            }
        }
        if !(self.merge.prune_threshold > 0.0 && self.merge.prune_threshold < 1.0) {
            return Err(SidecarError::BadPruneThreshold(self.merge.prune_threshold));
        }
        Ok(())
    }

    /// The default L0 sidecar: six zones with default cores and the default
    /// topology of Architecture §5.1. This is the Gate-2 reference configuration.
    pub fn default_l0() -> Sidecar {
        let slice_width = 16u32;
        let zones = ZoneId::ALL
            .iter()
            .enumerate()
            .map(|(i, &id)| ZoneDecl {
                id,
                core: id.default_core(),
                slice_offset: i as u32 * slice_width,
                slice_width,
                modalities: match id {
                    ZoneId::SM => vec!["text".into(), "audio".into(), "vision".into()],
                    _ => vec!["text".into()],
                },
                recipe_ref: format!("recipe::{}", id.as_str()),
            })
            .collect();

        // Default topology (Architecture §5.1):
        //   SM→CB  SM→HP  SM→PF   CB→PF  HP→PF  PF→CX
        let e = |from, to| Edge { from, to };
        let edges = vec![
            e(ZoneId::SM, ZoneId::CB),
            e(ZoneId::SM, ZoneId::HP),
            e(ZoneId::SM, ZoneId::PF),
            e(ZoneId::CB, ZoneId::PF),
            e(ZoneId::HP, ZoneId::PF),
            e(ZoneId::PF, ZoneId::CX),
        ];

        Sidecar {
            version: 1,
            export_kind: ExportKind::ZonePartitioned,
            zones,
            topology: Topology { edges },
            merge: MergeParams {
                prune_threshold: 0.8,
                deadline_ms: 100,
            },
            composition_rules: vec![
                "swappable: any zone whose slice_width and cross-zone head contract match".into(),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_l0_validates_and_round_trips() {
        let sc = Sidecar::default_l0();
        sc.validate().unwrap();
        let json = sc.to_json();
        let back = Sidecar::from_json(&json).unwrap();
        assert_eq!(sc, back);
    }

    #[test]
    fn default_topology_has_the_six_declared_edges_only() {
        let sc = Sidecar::default_l0();
        assert_eq!(sc.topology.edges.len(), 6);
        assert!(sc.topology.has_edge(ZoneId::PF, ZoneId::CX));
        // An undeclared edge does not exist — this is the auditability property.
        assert!(!sc.topology.has_edge(ZoneId::CX, ZoneId::PF));
        assert!(!sc.topology.has_edge(ZoneId::SM, ZoneId::CX));
    }

    #[test]
    fn mx_with_a_learned_core_is_rejected() {
        let mut sc = Sidecar::default_l0();
        sc.zones
            .iter_mut()
            .find(|z| z.id == ZoneId::MX)
            .unwrap()
            .core = CoreType::Attention;
        assert_eq!(sc.validate(), Err(SidecarError::MxHasLearnedCore));
    }

    #[test]
    fn edge_to_undeclared_zone_is_rejected() {
        let mut sc = Sidecar::default_l0();
        sc.zones.retain(|z| z.id != ZoneId::CX); // drop CX but keep PF→CX edge
        assert_eq!(
            sc.validate(),
            Err(SidecarError::UndeclaredZoneInEdge(ZoneId::CX))
        );
    }
}
