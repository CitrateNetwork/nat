//! The H-01 ablation harness — the bet-decider (Master Plan risk #1, ADR-0005).
//!
//! H-01: *zone partitioning does not reduce capability per parameter versus a
//! dense baseline of equal size.* This harness runs the comparison under the
//! pinned ADR-0005 protocol and is GPU-free now, ready to scale on the DGX (where
//! the partitioned arm becomes the full `NatModel` with real Candle cores and the
//! data becomes the real corpus).
//!
//! Two guarantees are enforced in code, not left to discipline:
//!
//! 1. **Equal parameters (ADR-0005).** The partitioned arm is sized to match the
//!    dense arm's parameter count; if it cannot be matched within tolerance the
//!    run is *refused* ([`AblationError::ParamsMismatch`]). An ablation at
//!    unequal params proves nothing, so the harness will not produce one.
//! 2. **No toy cores.** [`guard_not_toy`] rejects a run whose model is on the L0
//!    toy backend (`nat_core::NatModel::uses_toy_cores`), so the bet-deciding run
//!    on the DGX cannot silently measure toys.

mod models;

pub use models::{
    dense_params, partitioned_params, synthetic_data, AblationArm, DenseArm, PartitionedArm,
    TrainData,
};

use candle_core::Device;
use nat_train::repro::RunConfig;
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AblationError {
    #[error("ablation invalid: dense={dense} vs partitioned={partitioned} params exceeds tolerance {tolerance} (ADR-0005 requires equal params)")]
    ParamsMismatch {
        dense: usize,
        partitioned: usize,
        tolerance: f64,
    },
    #[error("toy cores are forbidden in an ablation run; use a real (Candle) backend so H-01 measures the real model")]
    ToyCoresForbidden,
    #[error("candle error: {0}")]
    Candle(String),
}

/// The ablation protocol. Both arms train under *identical* conditions; the only
/// difference is the structure (partitioned vs dense).
#[derive(Debug, Clone)]
pub struct AblationConfig {
    pub in_dim: usize,
    pub out_dim: usize,
    /// Dense baseline hidden width — this sets the parameter budget both arms hit.
    pub dense_hidden: usize,
    /// Number of zones in the partitioned arm (NAT has 5 learned zones).
    pub n_zones: usize,
    pub steps: usize,
    pub lr: f64,
    pub seed: u64,
    /// Max allowed relative parameter difference between the two arms (ADR-0005).
    pub param_tolerance: f64,
}

impl Default for AblationConfig {
    fn default() -> Self {
        AblationConfig {
            in_dim: 16,
            out_dim: 8,
            dense_hidden: 64,
            n_zones: 5,
            steps: 200,
            lr: 0.05,
            seed: 2026,
            param_tolerance: 0.05,
        }
    }
}

/// The result of one ablation run.
#[derive(Debug, Clone)]
pub struct AblationReport {
    pub backend: String,
    pub dense_params: usize,
    pub partitioned_params: usize,
    /// Relative param difference actually achieved (≤ `param_tolerance`).
    pub param_rel_diff: f64,
    pub dense_initial_loss: f32,
    pub dense_final_loss: f32,
    pub partitioned_initial_loss: f32,
    pub partitioned_final_loss: f32,
    /// Capability proxy = 1 / (final_loss + ε). Higher is better.
    pub dense_capability: f64,
    pub partitioned_capability: f64,
    pub dense_capability_per_param: f64,
    pub partitioned_capability_per_param: f64,
    /// The H-01 verdict at this scale: partitioned capability-per-param is at
    /// least the dense baseline's (within a small slack). NOT conclusive at toy
    /// scale — the real verdict is the DGX run with real models and data.
    pub h01_holds: bool,
    /// Reproducibility anchor for this run (Research Method §8).
    pub repro_config_hash: String,
}

impl AblationReport {
    pub fn summary(&self) -> String {
        format!(
            "H-01 ablation [{}] (params dense={} partitioned={} reldiff={:.3})\n  \
             dense:       loss {:.4} -> cap/param {:.3e}\n  \
             partitioned: loss {:.4} -> cap/param {:.3e}\n  \
             verdict: H-01 {} (partitioned cap/param {} dense)\n  repro: {}",
            self.backend,
            self.dense_params,
            self.partitioned_params,
            self.param_rel_diff,
            self.dense_final_loss,
            self.dense_capability_per_param,
            self.partitioned_final_loss,
            self.partitioned_capability_per_param,
            if self.h01_holds { "HOLDS" } else { "REFUTED" },
            if self.h01_holds { "≥" } else { "<" },
            self.repro_config_hash,
        )
    }
}

/// Refuse to run an ablation on toy cores. The DGX driver calls this with
/// `nat_model.uses_toy_cores()` before measuring anything.
pub fn guard_not_toy(uses_toy_cores: bool) -> Result<(), AblationError> {
    if uses_toy_cores {
        Err(AblationError::ToyCoresForbidden)
    } else {
        Ok(())
    }
}

/// Find the partitioned `zone_hidden` whose total params are closest to `target`.
pub fn match_zone_hidden(in_dim: usize, n_zones: usize, out_dim: usize, target: usize) -> usize {
    let mut best = 1usize;
    let mut best_diff = usize::MAX;
    for zh in 1..=8192 {
        let p = partitioned_params(in_dim, n_zones, zh, out_dim);
        let diff = p.abs_diff(target);
        if diff < best_diff {
            best_diff = diff;
            best = zh;
        }
        if p > target {
            break; // params are monotincreasing in zh; we passed the closest
        }
    }
    best
}

fn candle_err(e: candle_core::Error) -> AblationError {
    AblationError::Candle(e.to_string())
}

/// Run the H-01 ablation under the ADR-0005 protocol. Refuses (errors) rather
/// than report an invalid comparison if the arms cannot be param-matched.
pub fn run_ablation(cfg: &AblationConfig) -> Result<AblationReport, AblationError> {
    let dev = Device::Cpu;

    // 1. Size the partitioned arm to the dense arm's parameter budget (ADR-0005).
    let dense_p = dense_params(cfg.in_dim, cfg.dense_hidden, cfg.out_dim);
    let zh = match_zone_hidden(cfg.in_dim, cfg.n_zones, cfg.out_dim, dense_p);
    let part_p = partitioned_params(cfg.in_dim, cfg.n_zones, zh, cfg.out_dim);
    let rel = (part_p.abs_diff(dense_p) as f64) / (dense_p as f64);
    if rel > cfg.param_tolerance {
        return Err(AblationError::ParamsMismatch {
            dense: dense_p,
            partitioned: part_p,
            tolerance: cfg.param_tolerance,
        });
    }

    // 2. Train both arms under identical conditions (same data, steps, lr).
    let data = synthetic_data(64, cfg.in_dim, cfg.out_dim, &dev).map_err(candle_err)?;
    let mut dense =
        DenseArm::new(cfg.in_dim, cfg.dense_hidden, cfg.out_dim, &dev).map_err(candle_err)?;
    let mut part =
        PartitionedArm::new(cfg.in_dim, cfg.n_zones, zh, cfg.out_dim, &dev).map_err(candle_err)?;
    let (di, df) = dense.train(&data, cfg.steps, cfg.lr).map_err(candle_err)?;
    let (pi, pf) = part.train(&data, cfg.steps, cfg.lr).map_err(candle_err)?;

    // 3. Capability proxy and the per-parameter comparison.
    let cap = |l: f32| 1.0f64 / (l as f64 + 1e-6);
    let (dense_cap, part_cap) = (cap(df), cap(pf));
    let dense_cpp = dense_cap / dense_p as f64;
    let part_cpp = part_cap / part_p as f64;
    // "Does not reduce" → at least the baseline within 5% slack.
    let h01_holds = part_cpp >= dense_cpp * 0.95;

    // 4. Reproducibility anchor (Research Method §8).
    let mut hp = BTreeMap::new();
    hp.insert("dense_hidden".into(), cfg.dense_hidden.to_string());
    hp.insert("n_zones".into(), cfg.n_zones.to_string());
    hp.insert("zone_hidden".into(), zh.to_string());
    hp.insert("steps".into(), cfg.steps.to_string());
    hp.insert("lr".into(), format!("{:.5}", cfg.lr));
    let repro = RunConfig {
        rung: "ablation".into(),
        seed: cfg.seed,
        data_config_hash: "synthetic-v1".into(),
        data_manifest_hash: "synthetic-v1".into(),
        hyperparams: hp,
    };

    Ok(AblationReport {
        backend: "candle-cpu".into(),
        dense_params: dense_p,
        partitioned_params: part_p,
        param_rel_diff: rel,
        dense_initial_loss: di,
        dense_final_loss: df,
        partitioned_initial_loss: pi,
        partitioned_final_loss: pf,
        dense_capability: dense_cap,
        partitioned_capability: part_cap,
        dense_capability_per_param: dense_cpp,
        partitioned_capability_per_param: part_cpp,
        h01_holds,
        repro_config_hash: repro.config_hash(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arms_are_param_matched_within_tolerance() {
        let cfg = AblationConfig::default();
        let report = run_ablation(&cfg).unwrap();
        assert!(
            report.param_rel_diff <= cfg.param_tolerance,
            "{}",
            report.summary()
        );
        // Both arms actually trained (loss moved).
        assert!(report.dense_final_loss < report.dense_initial_loss);
        assert!(report.partitioned_final_loss < report.partitioned_initial_loss);
    }

    #[test]
    fn report_is_well_formed_and_records_backend_and_repro() {
        let report = run_ablation(&AblationConfig::default()).unwrap();
        assert_eq!(report.backend, "candle-cpu");
        assert!(!report.repro_config_hash.is_empty());
        assert!(report.dense_capability_per_param > 0.0);
        assert!(report.partitioned_capability_per_param > 0.0);
        // The verdict is a clean bool either way (we don't assert its value at toy scale).
        let _ = report.h01_holds;
    }

    #[test]
    fn unmatched_params_are_refused() {
        // A tolerance of 0 makes an exact match all but impossible → refuse.
        let cfg = AblationConfig {
            param_tolerance: 0.0,
            dense_hidden: 63, // odd budget unlikely to be hit exactly by 5 zones
            ..AblationConfig::default()
        };
        match run_ablation(&cfg) {
            Err(AblationError::ParamsMismatch { .. }) => {}
            other => panic!("expected ParamsMismatch, got {other:?}"),
        }
    }

    #[test]
    fn toy_cores_are_refused() {
        assert!(guard_not_toy(true).is_err()); // a toy-backed model cannot run the ablation
        assert!(guard_not_toy(false).is_ok()); // a real (Candle) backend may
    }

    #[test]
    fn zone_hidden_matching_is_deterministic_and_close() {
        let target = dense_params(16, 64, 8);
        let zh = match_zone_hidden(16, 5, 8, target);
        let p = partitioned_params(16, 5, zh, 8);
        assert_eq!(zh, match_zone_hidden(16, 5, 8, target));
        assert!((p.abs_diff(target) as f64) / target as f64 <= 0.05);
    }
}
