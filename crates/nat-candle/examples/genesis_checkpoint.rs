//! Write the deterministic starting weights for an H-01 ladder run.
//!
//! ## Why "from scratch" is an artifact rather than a code path
//!
//! H-01 trains fresh models at seven rungs in two arms. The obvious way to run
//! that on the co-op backend is to teach the worker to initialise a model when no
//! checkpoint is named. This does the opposite: it materialises the fresh weights
//! as a **content-addressed checkpoint**, so a ladder run is an ordinary resume
//! job and the worker needs no new path at all.
//!
//! That is not just less code. It is the difference between
//!
//!   > "the worker says it initialised with seed 7"
//!
//! and
//!
//!   > "the run started from `0x…`, and here are the bytes; hash them yourself"
//!
//! An ablation whose conclusions depend on every arm starting from the same place
//! should be able to *prove* they did. An in-code initialiser cannot be audited
//! after the fact — a challenger re-running the job would have to trust that the
//! worker's RNG did what it claimed. A checkpoint can simply be re-hashed.
//!
//! It also means every arm, rung and seed of the campaign has a name that appears
//! in the job payload, the mirror path and the settlement record.
//!
//! ## Usage
//!
//! ```sh
//! # arm rung_params seed [vocab] [seq] [dtype]
//! cargo run --release -p nat-candle --example genesis_checkpoint -- \
//!   nat 64000000 1 16384 64 bf16 /tmp/genesis
//! ```
//!
//! Prints the keccak256 of the written `model.safetensors` — the
//! `model_start_hash` for the job payload and the mirror path.

use candle_core::DType;
use nat_candle::autoreg::{AutoregConfig, AutoregDenseLm, AutoregLm};
use nat_types::ZoneId;
use sha3::{Digest, Keccak256};

/// Binary-search `d` so an arm hits the rung's parameter target.
///
/// The two arms must be **param-matched**, or the ablation measures capacity
/// rather than architecture (ADR-0005). Each arm is sized independently against
/// the same target rather than assuming a fixed ratio between them.
fn size_d(target: usize, vocab: usize, seq: usize, count: impl Fn(usize) -> usize) -> usize {
    let (mut lo, mut hi) = (8usize, 4096usize);
    let _ = (vocab, seq);
    while lo < hi {
        let mid = (lo + hi) / 2;
        if count(mid) < target {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

fn zone_cfg(d: usize, vocab: usize, seq: usize, seed: u64) -> AutoregConfig {
    AutoregConfig {
        zones: vec![
            ZoneId::SM,
            ZoneId::CB,
            ZoneId::HP,
            ZoneId::PF,
            ZoneId::CX,
        ],
        vocab,
        seq_len: seq,
        d,
        tau: 1.0,
        // ADR-0012. The floor is part of the protocol now, so the ladder's
        // starting weights are built with the same merge the run will use — a
        // genesis built at floor 0 would be a different experiment.
        merge_floor: nat_candle::autoreg::DEFAULT_MERGE_FLOOR,
        seed,
    }
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 3 {
        anyhow::bail!("usage: genesis_checkpoint <nat|dense> <target_params> <seed> [vocab] [seq] [dtype] [outdir]");
    }
    let arm = a[0].as_str();
    let target: usize = a[1].parse()?;
    let seed: u64 = a[2].parse()?;
    let vocab: usize = a.get(3).map(|s| s.parse()).transpose()?.unwrap_or(16384);
    let seq: usize = a.get(4).map(|s| s.parse()).transpose()?.unwrap_or(64);
    let dtype_name = a.get(5).cloned().unwrap_or_else(|| "bf16".into());
    let out = a.get(6).cloned().unwrap_or_else(|| "./genesis".into());

    let dtype = match dtype_name.as_str() {
        "bf16" => DType::BF16,
        "f32" => DType::F32,
        other => anyhow::bail!("unsupported dtype {other:?} — use bf16 or f32"),
    };

    let (d, params, arch) = match arm {
        "nat" => {
            let d = size_d(target, vocab, seq, |d| {
                AutoregLm::new_with_dtype(&zone_cfg(d, vocab, seq, seed), dtype)
                    .map(|m| m.param_count())
                    .unwrap_or(usize::MAX)
            });
            let m = AutoregLm::new_with_dtype(&zone_cfg(d, vocab, seq, seed), dtype)?;
            let p = m.param_count();
            std::fs::create_dir_all(&out)?;
            m.save(std::path::Path::new(&out))?;
            (d, p, "zone-partitioned")
        }
        "dense" => {
            // d_ff at ~6.9x d, the ratio the 64M run used (d=1183, d_ff=8192).
            let d = size_d(target, vocab, seq, |d| {
                AutoregDenseLm::new_with_dtype(vocab, seq, d, d * 69 / 10, seed, dtype)
                    .map(|m| m.param_count())
                    .unwrap_or(usize::MAX)
            });
            let m = AutoregDenseLm::new_with_dtype(vocab, seq, d, d * 69 / 10, seed, dtype)?;
            let p = m.param_count();
            std::fs::create_dir_all(&out)?;
            m.save(std::path::Path::new(&out))?;
            (d, p, "dense")
        }
        other => anyhow::bail!("unknown arm {other:?} — use nat or dense"),
    };

    // The sidecar the worker reads: shape and commitment grid, declared rather
    // than inferred. `ArtifactStore::resolve` refuses anything it cannot model,
    // so a wrong value here fails loudly instead of training the wrong thing.
    let sidecar = serde_json::json!({
        "architecture": arch,
        "commitment_grid": "q16",
        "vocab": vocab,
        "d": d,
        "seq_len": seq,
        "dtype": dtype_name,
        "zones": if arch == "zone-partitioned" {
            serde_json::json!([{"id":"SM"},{"id":"CB"},{"id":"HP"},{"id":"PF"},{"id":"CX"}])
        } else {
            serde_json::json!([])
        },
    });
    std::fs::write(
        std::path::Path::new(&out).join("sidecar.nat.json"),
        serde_json::to_string_pretty(&sidecar)?,
    )?;

    let bytes = std::fs::read(std::path::Path::new(&out).join("model.safetensors"))?;
    let mut h = Keccak256::new();
    h.update(&bytes);
    let hash = format!("0x{:x}", h.finalize());

    eprintln!("arm={arm} d={d} params={params} target={target} vocab={vocab} seq={seq} dtype={dtype_name}");
    eprintln!("  wrote {}/model.safetensors ({} bytes)", out, bytes.len());
    // stdout is the hash alone, so a script can capture it.
    println!("{hash}");
    Ok(())
}
