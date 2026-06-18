//! Shared primitives for NAT.
//!
//! This crate has no internal dependencies on purpose: every other crate can
//! depend on it without creating a cycle. It owns the vocabulary that the
//! Architecture spec §3 makes normative — `ZoneId`, `CoreType`, `ZoneStatus` —
//! and the [`Q16`] fixed-point type that the deterministic merge path runs on.

mod fixed;
pub use fixed::Q16;

use serde::{Deserialize, Serialize};

/// The six declared zones (Architecture §4). Order is fixed and meaningful: it
/// is the canonical order zones appear in the provenance trace and the router's
/// activation vector, so it must never be reordered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ZoneId {
    /// Sensorimotor — ingest + temporally bind multimodal input.
    SM,
    /// Cerebellar — timing, motor sequencing, learned reflex.
    CB,
    /// Hippocampal — memory consolidation, novelty/salience.
    HP,
    /// Prefrontal — reasoning, planning, language. The deepest zone.
    PF,
    /// Codec — reasoning → verifiable executable logic. The determinism anchor.
    CX,
    /// MCP Harness — validate/sequence/route tool use. Non-learned.
    MX,
}

impl ZoneId {
    /// Canonical order. The length-6 activation vector and the trace use this.
    pub const ALL: [ZoneId; 6] = [
        ZoneId::SM,
        ZoneId::CB,
        ZoneId::HP,
        ZoneId::PF,
        ZoneId::CX,
        ZoneId::MX,
    ];

    /// The five learned zones — everything except the non-learned `MX` harness.
    pub const LEARNED: [ZoneId; 5] = [ZoneId::SM, ZoneId::CB, ZoneId::HP, ZoneId::PF, ZoneId::CX];

    pub fn as_str(self) -> &'static str {
        match self {
            ZoneId::SM => "SM",
            ZoneId::CB => "CB",
            ZoneId::HP => "HP",
            ZoneId::PF => "PF",
            ZoneId::CX => "CX",
            ZoneId::MX => "MX",
        }
    }

    /// `MX` is the only non-learned zone (Architecture §4.6). Everything that
    /// trains, gathers, and merges applies to the learned zones only.
    pub fn is_learned(self) -> bool {
        self != ZoneId::MX
    }

    /// The core type each zone uses by default (ADR-0002).
    pub fn default_core(self) -> CoreType {
        match self {
            ZoneId::SM | ZoneId::CB => CoreType::Ssm,
            ZoneId::HP | ZoneId::PF | ZoneId::CX => CoreType::Attention,
            ZoneId::MX => CoreType::None,
        }
    }
}

/// The per-zone sequence operator (Architecture §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoreType {
    /// State-space core: linear-time recurrence, for temporal zones.
    Ssm,
    /// Attention core: content-addressable look-back, for reasoning zones.
    Attention,
    /// No learned core: the non-learned `MX` harness.
    None,
}

/// A zone's status in one pass, as recorded in the provenance trace (§7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZoneStatus {
    /// Output arrived before the gather deadline and survived the merge prune.
    Ok,
    /// Output did not arrive before the gather deadline (AsyncGather §A.2).
    TimedOut,
    /// Output arrived but was pruned by the merge as noise (Merge §6 step 2).
    Pruned,
}

impl ZoneStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ZoneStatus::Ok => "ok",
            ZoneStatus::TimedOut => "timed_out",
            ZoneStatus::Pruned => "pruned",
        }
    }
}

/// The Codec zone's verification result (Architecture §4.5). A `Fail` is a
/// first-class output, not a discarded one — it blocks dependent tool execution
/// in the MCP harness (`NoExecOnFailedCodec`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Verification {
    Pass,
    Fail,
    Unverified,
}

impl Verification {
    pub fn as_str(self) -> &'static str {
        match self {
            Verification::Pass => "pass",
            Verification::Fail => "fail",
            Verification::Unverified => "unverified",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_zones_have_stable_canonical_order() {
        assert_eq!(ZoneId::ALL.len(), 6);
        // The router's activation vector is indexed by this order; if it ever
        // reorders, every trace ever committed on-chain becomes unreadable.
        assert_eq!(ZoneId::ALL[3], ZoneId::PF);
    }

    #[test]
    fn only_mx_is_non_learned() {
        assert_eq!(ZoneId::LEARNED.len(), 5);
        assert!(!ZoneId::MX.is_learned());
        assert!(ZoneId::ALL.iter().filter(|z| z.is_learned()).count() == 5);
    }

    #[test]
    fn default_cores_match_adr_0002() {
        assert_eq!(ZoneId::SM.default_core(), CoreType::Ssm);
        assert_eq!(ZoneId::CB.default_core(), CoreType::Ssm);
        assert_eq!(ZoneId::PF.default_core(), CoreType::Attention);
        assert_eq!(ZoneId::MX.default_core(), CoreType::None);
    }

    #[test]
    fn zoneid_serializes_to_its_name() {
        assert_eq!(serde_json::to_string(&ZoneId::PF).unwrap(), "\"PF\"");
    }
}
