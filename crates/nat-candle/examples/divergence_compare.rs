//! `divergence_compare` — turn a pile of `divergence_probe` outputs into the one
//! number the co-op needs: how far apart honest members land.
//!
//! ```sh
//! cargo run --release -p nat-candle --example divergence_compare -- probe-*.json
//! ```
//!
//! ## What it is deciding
//!
//! The settlement path has to distinguish *a different GPU* from *a liar*. Those
//! look the same to an equality check and completely different to a tolerance
//! check, so the tolerance has to sit above honest hardware divergence and far
//! below anything a cheat could produce. This prints both edges of that gap.
//!
//! Cheating is not subtle: skipping steps, training on the wrong data or making
//! numbers up moves weights by O(0.01–1). Honest backend divergence, measured, is
//! O(1e-7). Those are ~5 orders of magnitude apart, which is why a tolerance works
//! at all — but the gap has to be *measured* per hardware class, not assumed.

use std::collections::BTreeMap;

/// Q16.16: one grid step is 2^-16. The reference scale for every delta below —
/// a divergence under one step is invisible to the fixed-point grid *except* at
/// quantization boundaries, which is the whole subtlety (see `main`).
const Q16_STEP: f64 = 1.0 / 65536.0;

struct Probe {
    label: String,
    backend: String,
    dtype: String,
    os: String,
    arch: String,
    commitment: String,
    self_repeat: bool,
    loss_after: f64,
    global_l2: f64,
    tok_s: f64,
    points: Vec<f64>,
    job: serde_json::Value,
    zones: BTreeMap<String, f64>,
}

fn load(path: &str) -> anyhow::Result<Probe> {
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    anyhow::ensure!(
        v["schema"] == "nat.divergence-probe/1",
        "{path}: not a divergence-probe/1 document"
    );
    let zones = v["zone_shares"]
        .as_object()
        .map(|o| {
            o.iter()
                .map(|(k, x)| (k.clone(), x.as_f64().unwrap_or(f64::NAN)))
                .collect()
        })
        .unwrap_or_default();
    Ok(Probe {
        label: std::path::Path::new(path)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| path.to_string()),
        backend: v["backend"].as_str().unwrap_or("?").to_string(),
        dtype: v["dtype"].as_str().unwrap_or("?").to_string(),
        os: v["os"].as_str().unwrap_or("?").to_string(),
        arch: v["arch"].as_str().unwrap_or("?").to_string(),
        commitment: v["weights"]["q16_commitment"].as_str().unwrap_or("?").to_string(),
        self_repeat: v["self_repeat_identical"].as_bool().unwrap_or(false),
        loss_after: v["loss"]["after"].as_f64().unwrap_or(f64::NAN),
        global_l2: v["weights"]["global_l2"].as_f64().unwrap_or(f64::NAN),
        tok_s: v["perf"]["tokens_per_second"].as_f64().unwrap_or(f64::NAN),
        points: v["weights"]["probe_points"]
            .as_array()
            .map(|a| a.iter().filter_map(|x| x.as_f64()).collect())
            .unwrap_or_default(),
        job: v["job"].clone(),
        zones,
    })
}

fn main() -> anyhow::Result<()> {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    anyhow::ensure!(
        paths.len() >= 2,
        "need at least two probe files: divergence_compare probe-a.json probe-b.json ..."
    );
    let probes: Vec<Probe> = paths.iter().map(|p| load(p)).collect::<Result<_, _>>()?;

    println!("=== probes ===");
    for p in &probes {
        println!(
            "  {:<22} {:<14} {:<5} {}/{:<8} {:>9.0} tok/s  self-repeat {}",
            p.label,
            p.backend,
            p.dtype,
            p.os,
            p.arch,
            p.tok_s,
            if p.self_repeat { "ok" } else { "**FAILED**" }
        );
    }

    // A probe from a modified build is not comparable. Catch it loudly rather
    // than reporting a divergence that is really a different experiment.
    let job0 = &probes[0].job;
    for p in &probes[1..] {
        if &p.job != job0 {
            println!("\n*** {} ran a DIFFERENT job spec — results are not comparable ***", p.label);
            println!("    {}", p.job);
            println!("    {job0}");
            return Ok(());
        }
    }

    // Self-repeat is the precondition for everything else: a device that cannot
    // reproduce itself contributes noise to every pairwise number below.
    let flaky: Vec<&str> = probes.iter().filter(|p| !p.self_repeat).map(|p| p.label.as_str()).collect();
    if !flaky.is_empty() {
        println!(
            "\n*** {} failed the self-repeat control. Pairwise deltas involving these\n\
             *** mix run-to-run nondeterminism with cross-machine divergence.",
            flaky.join(", ")
        );
    }

    let identical = probes.iter().all(|p| p.commitment == probes[0].commitment);
    println!(
        "\n=== q16 commitments === {}",
        if identical { "ALL IDENTICAL" } else { "differ (expected across backends)" }
    );
    for p in &probes {
        println!("  {:<22} {}", p.label, &p.commitment[..16.min(p.commitment.len())]);
    }

    println!("\n=== pairwise weight divergence ===");
    println!(
        "  {:<20} {:<20} {:>12} {:>12} {:>10} {:>12}",
        "a", "b", "max|Δ|", "max rel Δ", "Δ/Q16step", "Δloss"
    );
    let mut worst = 0.0f64;
    for i in 0..probes.len() {
        for j in (i + 1)..probes.len() {
            let (a, b) = (&probes[i], &probes[j]);
            let n = a.points.len().min(b.points.len());
            let (mut mx, mut mrel) = (0.0f64, 0.0f64);
            for k in 0..n {
                let d = (a.points[k] - b.points[k]).abs();
                mx = mx.max(d);
                mrel = mrel.max(d / a.points[k].abs().max(1e-12));
            }
            worst = worst.max(mx);
            println!(
                "  {:<20} {:<20} {:>12.3e} {:>12.3e} {:>10.4} {:>12.3e}",
                a.label,
                b.label,
                mx,
                mrel,
                mx / Q16_STEP,
                (a.loss_after - b.loss_after).abs()
            );
        }
    }

    println!("\n=== zone shares (a dead zone on one machine and not another matters\n\
             === far more than a float ulp) ===");
    let names: Vec<&String> = probes[0].zones.keys().collect();
    print!("  {:<22}", "probe");
    for z in &names {
        print!("{:>10}", z);
    }
    println!();
    for p in &probes {
        print!("  {:<22}", p.label);
        for z in &names {
            print!("{:>10.6}", p.zones.get(*z).copied().unwrap_or(f64::NAN));
        }
        println!();
    }

    println!("\n=== verdict ===");
    println!("  worst honest divergence : {worst:.3e}");
    println!("  one Q16 grid step       : {Q16_STEP:.3e}  ({:.1}x the divergence)", Q16_STEP / worst.max(1e-30));
    println!("  global L2 spread        : {:.3e}", {
        let (lo, hi) = probes.iter().fold((f64::MAX, f64::MIN), |(lo, hi), p| {
            (lo.min(p.global_l2), hi.max(p.global_l2))
        });
        hi - lo
    });
    if identical {
        println!(
            "\n  Commitments match across every probe, so exact equality is usable as the\n  \
             settlement primitive for THIS hardware set."
        );
    } else {
        println!(
            "\n  Commitments differ while the divergence sits far below one grid step. That\n  \
             is quantization-boundary straddling, not real disagreement: a coordinate\n  \
             sitting near a grid edge rounds either way under a 1-ulp nudge. Exact\n  \
             equality therefore CANNOT be the settlement primitive across these\n  \
             backends — settlement needs a tolerance, and {:.0e} sits ~{:.0e}x above the\n  \
             measured divergence while staying orders of magnitude below any cheat.",
            (worst * 10.0).max(1e-6),
            ((worst * 10.0).max(1e-6)) / worst.max(1e-30)
        );
    }
    Ok(())
}
