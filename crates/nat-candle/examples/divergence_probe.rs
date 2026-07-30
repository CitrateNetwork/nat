//! `divergence_probe` — how far apart does the same job land on different hardware?
//!
//! The co-op's economic security rests on a number nobody has measured. When two
//! members train the same work on different machines, how much do the results
//! differ? Every downstream constant depends on the answer:
//!
//!   * `nat_federated::within_tolerance`'s slack (H-05b: federated ≈ centralized)
//!   * the challenge threshold in `ComputePoolTraining` — set it below the honest
//!     divergence and honest members get slashed for owning the wrong GPU
//!   * whether the Q16 grid is coarse enough to absorb cross-backend drift
//!
//! CUDA, Metal and CPU do **not** produce bit-identical floating-point results:
//! reduction orders and fused kernels differ, and training amplifies small
//! differences. So the question is not *whether* they diverge — they will — but
//! *by how much*, which is what this measures.
//!
//! ## Running it
//!
//! Needs no corpus and no checkpoint: the token stream is generated from a fixed
//! seed, so every machine trains on byte-identical input with nothing to download.
//!
//! ```sh
//! cargo run --release -p nat-candle --example divergence_probe > probe.json
//! # accelerators (pick the one your machine has):
//! cargo run --release -p nat-candle --features cuda  --example divergence_probe > probe.json
//! cargo run --release -p nat-candle --features metal --example divergence_probe > probe.json
//! ```
//!
//! JSON goes to stdout and the human summary to stderr, so the redirect above
//! captures exactly the artifact and you still watch it run.
//!
//! ## The self-repeat control
//!
//! Before comparing two machines, one machine has to agree with **itself**. Some
//! CUDA kernels use atomics and are not run-to-run deterministic, and if a single
//! device cannot reproduce its own result then cross-device comparison measures
//! nothing. So the probe trains the identical job twice in-process and reports
//! `self_repeat_identical`. **Read that field first**: if it is false, every
//! cross-machine number below it is noise plus divergence, not divergence.

use candle_core::{DType, Tensor};
use nat_candle::autoreg::{AutoregConfig, AutoregLm};
use nat_candle::device::{backend_label, device};
use nat_types::{Q16, ZoneId};
use sha2::{Digest, Sha256};
use std::time::Instant;

/// The job every machine must run identically. Changing any field makes probes
/// incomparable, so it is committed into the output and checked on compare —
/// a probe from a modified build is detectable rather than quietly wrong.
mod job {
    pub const SCHEMA: &str = "nat.divergence-probe/1";
    pub const VOCAB: usize = 1024;
    pub const SEQ: usize = 64;
    pub const D: usize = 96;
    pub const WINDOWS: usize = 384;
    pub const BATCH: usize = 16;
    pub const LR: f64 = 3e-3;
    pub const SEED: u64 = 20_260_730;
    pub const TAU: f64 = 1.0;
    /// Sampled coordinates reported verbatim, so two probes can be differenced
    /// element-wise rather than only compared as equal/unequal.
    pub const PROBE_POINTS: usize = 32;
}

fn cfg() -> AutoregConfig {
    AutoregConfig {
        zones: vec![
            ZoneId::SM,
            ZoneId::CB,
            ZoneId::HP,
            ZoneId::PF,
            ZoneId::CX,
        ],
        vocab: job::VOCAB,
        seq_len: job::SEQ,
        d: job::D,
        tau: job::TAU,
        merge_floor: nat_candle::autoreg::DEFAULT_MERGE_FLOOR,
        seed: job::SEED,
    }
}

/// The same deterministic stream `bench_throughput` uses — a multiplicative hash
/// of the flat index, so it needs no RNG crate and is identical on every platform
/// (integer arithmetic only; no float, so no backend can perturb the *input*).
fn tokens(dev: &candle_core::Device, n: usize) -> Tensor {
    Tensor::from_vec(
        (0..(n * job::SEQ) as u64)
            .map(|i| (i.wrapping_mul(2_654_435_761) % job::VOCAB as u64) as u32)
            .collect::<Vec<_>>(),
        (n, job::SEQ),
        dev,
    )
    .unwrap()
}

struct Run {
    loss_before: f32,
    loss_after: f32,
    seconds: f64,
    params: Vec<(String, Vec<f32>)>,
    zone_shares: Vec<f32>,
}

fn train_once(dtype: DType) -> Run {
    let dev = device();
    let c = cfg();
    let mut m = AutoregLm::new_with_dtype(&c, dtype).unwrap();
    let ids = tokens(&dev, job::WINDOWS);

    let loss_before = m.loss_on_batched(&ids, job::BATCH).unwrap();
    let t = Instant::now();
    m.train_minibatched(&ids, 1, job::BATCH, job::LR, job::SEED)
        .unwrap();
    let seconds = t.elapsed().as_secs_f64();
    let loss_after = m.loss_on_batched(&ids, job::BATCH).unwrap();

    // Zone shares on a fixed slice: a dead zone (ADR-0012) on one member's
    // hardware but not another's would be a divergence that matters far more
    // than a float ulp, so it is measured here rather than assumed uniform.
    let zone_shares = m
        .zone_merge_weights(&ids.narrow(0, 0, job::BATCH).unwrap())
        .unwrap();

    Run {
        loss_before,
        loss_after,
        seconds,
        params: m.named_parameters().unwrap(),
        zone_shares,
    }
}

/// Commitment on the Q16 grid — the same fixed-point kernel the chain and the
/// settlement path use (`nat_types::Q16` re-exports `citrate_fed_types::Q16`), so
/// "did the grid absorb the drift?" is answered on the real grid and not an
/// approximation of it. Two machines agreeing here means their float differences
/// all fell inside one Q16 step.
fn q16_commitment(params: &[(String, Vec<f32>)]) -> String {
    let mut h = Sha256::new();
    for (name, vals) in params {
        h.update(name.as_bytes());
        for v in vals {
            h.update(Q16::from_f32(*v).raw().to_le_bytes());
        }
    }
    format!("{:x}", h.finalize())
}

fn l2(vals: &[f32]) -> f64 {
    vals.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt()
}

/// Flatten name-sorted parameters and take `PROBE_POINTS` evenly spaced values.
/// Reported raw so a comparison can compute an actual element-wise delta instead
/// of only observing that two hashes differ.
fn probe_points(params: &[(String, Vec<f32>)]) -> Vec<f32> {
    let flat: Vec<f32> = params.iter().flat_map(|(_, v)| v.iter().copied()).collect();
    if flat.is_empty() {
        return vec![];
    }
    let stride = (flat.len() / job::PROBE_POINTS).max(1);
    (0..job::PROBE_POINTS)
        .map(|i| flat[(i * stride).min(flat.len() - 1)])
        .collect()
}

fn main() {
    let dtype = match std::env::var("NAT_DTYPE").as_deref() {
        Ok("bf16") => DType::BF16,
        Ok("f16") => DType::F16,
        // f32 by default and on purpose: Candle's CPU backend has no BF16 matmul,
        // so f32 is the only dtype every contributor can actually run, and a
        // probe is worthless if half the fleet cannot produce one.
        _ => DType::F32,
    };
    let dtype_name = match dtype {
        DType::BF16 => "bf16",
        DType::F16 => "f16",
        _ => "f32",
    };

    eprintln!("nat divergence_probe — backend {}, dtype {dtype_name}", backend_label());
    eprintln!("  training the fixed job (run 1 of 2)...");
    let a = train_once(dtype);
    eprintln!("  training the fixed job (run 2 of 2, self-repeat control)...");
    let b = train_once(dtype);

    let commit_a = q16_commitment(&a.params);
    let commit_b = q16_commitment(&b.params);
    let self_repeat_identical = commit_a == commit_b;

    let n_params: usize = a.params.iter().map(|(_, v)| v.len()).sum();
    let toks = (job::WINDOWS * job::SEQ) as f64;

    let per_tensor: Vec<serde_json::Value> = a
        .params
        .iter()
        .map(|(name, vals)| {
            serde_json::json!({ "name": name, "n": vals.len(), "l2": l2(vals) })
        })
        .collect();

    let global_l2 = l2(&a.params.iter().flat_map(|(_, v)| v.iter().copied()).collect::<Vec<_>>());

    let zones = ["SM", "CB", "HP", "PF", "CX"];
    let zone_shares: serde_json::Map<String, serde_json::Value> = zones
        .iter()
        .zip(a.zone_shares.iter())
        .map(|(z, s)| ((*z).to_string(), serde_json::json!(*s)))
        .collect();

    let out = serde_json::json!({
        "schema": job::SCHEMA,
        "backend": backend_label(),
        "dtype": dtype_name,
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "job": {
            "vocab": job::VOCAB, "seq": job::SEQ, "d": job::D,
            "windows": job::WINDOWS, "batch": job::BATCH,
            "lr": job::LR, "seed": job::SEED, "tau": job::TAU,
            "merge_floor": nat_candle::autoreg::DEFAULT_MERGE_FLOOR,
        },
        "params": n_params,
        "perf": {
            "seconds": a.seconds,
            "tokens_per_second": toks / a.seconds,
        },
        // Read this first — see the module docs. False means the numbers below
        // mix run-to-run noise with cross-machine divergence.
        "self_repeat_identical": self_repeat_identical,
        "loss": { "before": a.loss_before, "after": a.loss_after },
        "weights": {
            "q16_commitment": commit_a,
            "global_l2": global_l2,
            "per_tensor": per_tensor,
            "probe_points": probe_points(&a.params),
        },
        "zone_shares": zone_shares,
    });

    eprintln!("  params            {n_params}");
    eprintln!("  loss              {:.4} → {:.4}", a.loss_before, a.loss_after);
    eprintln!("  throughput        {:.0} tok/s", toks / a.seconds);
    eprintln!("  self-repeat       {}", if self_repeat_identical {
        "IDENTICAL (this device reproduces itself)"
    } else {
        "*** DIFFERS *** — this device is not run-to-run deterministic"
    });
    eprintln!("  q16 commitment    {commit_a}");
    if !self_repeat_identical {
        eprintln!("  run-2 commitment  {commit_b}");
    }
    eprintln!("\n  Send the JSON on stdout. It contains no personal data — only");
    eprintln!("  hardware class, timings, and weights from synthetic tokens.");

    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
