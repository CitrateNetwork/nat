//! The reproducibility floor (Research Method §8).
//!
//! No stage closes without: a config hash that pins model + data slice +
//! hyperparameters, a fixed seed, recorded hardware/software versions, and the
//! exact command to rerun. This is not bureaucracy — it is what lets a federated
//! contributor (the "grandma-proof" node operator) trust that the model they are
//! training toward behaves the same way the reference does.
//!
//! Two hashes, two jobs:
//!
//! - [`RunConfig::config_hash`] pins the *logical* run — model, data, seed,
//!   hyperparameters. It is hardware-independent: the same logical run on two
//!   machines has the same config hash. This is what a node verifies it is
//!   training toward.
//! - [`ReproRecord::record_hash`] identifies a concrete *run instance* — config
//!   plus the hardware it actually ran on. Two machines produce different record
//!   hashes for the same config; that is the point (it records *where* it ran).

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// The hardware + software a run executed on. Recorded, not hashed into the
/// config (so the config stays reproducible across machines).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hardware {
    pub os: String,
    pub arch: String,
    pub logical_cpus: u32,
    /// The pinned toolchain (from the workspace `rust-version` / `rust-toolchain.toml`).
    pub rust_toolchain: String,
}

impl Hardware {
    /// Detect the current machine. Non-deterministic across machines by design —
    /// this is the "recorded hardware" half of the reproducibility floor.
    pub fn detect() -> Hardware {
        Hardware {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            logical_cpus: std::thread::available_parallelism()
                .map(|n| n.get() as u32)
                .unwrap_or(1),
            // The toolchain is pinned at build time; record the pinned channel.
            rust_toolchain: env!("CARGO_PKG_RUST_VERSION").to_string(),
        }
    }
}

/// The pinned configuration of a run. Its hash is the reproducibility anchor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunConfig {
    /// Scale-ladder rung: "L0" | "L1" | "L2" | "L3".
    pub rung: String,
    /// The fixed seed for this run (Research Method §8: fixed or logged seed).
    pub seed: u64,
    /// Hash of the data pipeline config (`nat_data::PipelineConfig::config_hash`).
    pub data_config_hash: String,
    /// Hash of the corpus manifest the run trains on (`CorpusManifest::manifest_hash`).
    pub data_manifest_hash: String,
    /// Hyperparameters, as an ordered map so the hash is order-independent.
    pub hyperparams: BTreeMap<String, String>,
}

impl RunConfig {
    /// Deterministic, hardware-independent hash of the logical run. Two machines
    /// that run the same model on the same data with the same seed and
    /// hyperparameters get the same config hash.
    pub fn config_hash(&self) -> String {
        // Canonical encoding: a BTreeMap iterates in sorted key order, so the
        // hyperparam serialization is independent of insertion order.
        let mut s = format!(
            "rung={};seed={};data_cfg={};data_manifest={};hp=[",
            self.rung, self.seed, self.data_config_hash, self.data_manifest_hash,
        );
        for (k, v) in &self.hyperparams {
            s.push_str(&format!("{k}={v};"));
        }
        s.push(']');
        hex(&Sha256::digest(s.as_bytes()))
    }

    /// The exact command to rerun this configuration. Part of the floor — a
    /// contributor copies this verbatim to reproduce the reference.
    pub fn rerun_command(&self) -> String {
        format!(
            "cargo run -p nat-train -- --rung {} --seed {} --data-config {} --data-manifest {}",
            self.rung, self.seed, self.data_config_hash, self.data_manifest_hash,
        )
    }
}

/// The full reproducibility record for one concrete run instance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReproRecord {
    /// The hardware-independent logical-run hash.
    pub config_hash: String,
    pub seed: u64,
    pub hardware: Hardware,
    pub data_manifest_hash: String,
    /// The exact command to rerun (copied from `RunConfig::rerun_command`).
    pub rerun_command: String,
}

impl ReproRecord {
    /// Build the record for a run on the given hardware.
    pub fn new(config: &RunConfig, hardware: Hardware) -> ReproRecord {
        ReproRecord {
            config_hash: config.config_hash(),
            seed: config.seed,
            hardware,
            data_manifest_hash: config.data_manifest_hash.clone(),
            rerun_command: config.rerun_command(),
        }
    }

    /// Hash of the concrete run instance (config + hardware). Identifies *where*
    /// and *what*; differs across machines even for the same `config_hash`.
    pub fn record_hash(&self) -> String {
        let bytes = serde_json::to_vec(self).expect("repro record always serializes");
        hex(&Sha256::digest(&bytes))
    }
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(char::from_digit((b >> 4) as u32, 16).unwrap());
        s.push(char::from_digit((b & 0xf) as u32, 16).unwrap());
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> RunConfig {
        let mut hp = BTreeMap::new();
        hp.insert("lr".into(), "0.0003".into());
        hp.insert("batch".into(), "32".into());
        RunConfig {
            rung: "L1".into(),
            seed: 1234,
            data_config_hash: "cfg-abc".into(),
            data_manifest_hash: "man-def".into(),
            hyperparams: hp,
        }
    }

    #[test]
    fn config_hash_is_deterministic() {
        assert_eq!(cfg().config_hash(), cfg().config_hash());
    }

    #[test]
    fn config_hash_is_hyperparam_order_independent() {
        // Insert hyperparams in the opposite order → same hash (BTreeMap sorts).
        let mut hp = BTreeMap::new();
        hp.insert("batch".into(), "32".into());
        hp.insert("lr".into(), "0.0003".into());
        let other = RunConfig {
            hyperparams: hp,
            ..cfg()
        };
        assert_eq!(cfg().config_hash(), other.config_hash());
    }

    #[test]
    fn changing_the_seed_changes_the_config_hash() {
        let mut other = cfg();
        other.seed = 9999;
        assert_ne!(cfg().config_hash(), other.config_hash());
    }

    #[test]
    fn rerun_command_pins_seed_and_data() {
        let cmd = cfg().rerun_command();
        assert!(cmd.contains("--seed 1234"));
        assert!(cmd.contains("--data-manifest man-def"));
    }

    #[test]
    fn hardware_detect_is_populated() {
        let hw = Hardware::detect();
        assert!(!hw.os.is_empty());
        assert!(!hw.arch.is_empty());
        assert!(hw.logical_cpus >= 1);
        assert_eq!(hw.rust_toolchain, "1.96.0"); // the pinned channel
    }

    #[test]
    fn record_hash_is_stable_and_config_hash_is_hardware_independent() {
        let c = cfg();
        let hw_a = Hardware {
            os: "linux".into(),
            arch: "x86_64".into(),
            logical_cpus: 64,
            rust_toolchain: "1.96.0".into(),
        };
        let hw_b = Hardware {
            os: "macos".into(),
            arch: "aarch64".into(),
            logical_cpus: 10,
            rust_toolchain: "1.96.0".into(),
        };
        let ra = ReproRecord::new(&c, hw_a);
        let rb = ReproRecord::new(&c, hw_b);

        // Same logical run → same config hash regardless of hardware...
        assert_eq!(ra.config_hash, rb.config_hash);
        // ...but the concrete record hashes differ (they ran in different places).
        assert_ne!(ra.record_hash(), rb.record_hash());
        // record_hash is itself stable.
        assert_eq!(ra.record_hash(), ra.record_hash());
    }
}
