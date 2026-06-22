//! Run the H-01 ablation (the bet-decider) and print the seed-averaged report.
//!
//!   cargo run -p nat-ablation --example ablation                              # CPU
//!   scripts/dgx-gpu.sh run -p nat-ablation --features cuda --example ablation # DGX GPU
//!
//! What this measures: zone-partitioned vs equal-param dense structure, averaged
//! over several seeds under the ADR-0005 protocol (same data/seed/optimizer/
//! compute; only the partitioning differs). The arms are the structural *analogs*
//! — the full `NatModel` (routing + merge + SSM cores) arm is gated on backlog
//! item #4 (a trainable end-to-end zone pass). So a HOLDS here is a necessary,
//! not a final, read on H-01. Honest posture: if it REFUTES at scale, say so.

fn main() {
    let cfg = nat_ablation::AblationConfig::scaled();
    let seeds = [1u64, 2, 3, 4, 5];

    println!(
        "running H-01 ablation: in={} out={} dense_hidden={} zones={} steps={} over {} seeds",
        cfg.in_dim,
        cfg.out_dim,
        cfg.dense_hidden,
        cfg.n_zones,
        cfg.steps,
        seeds.len()
    );

    let report = nat_ablation::run_ablation_seeds(&cfg, &seeds).expect("ablation run");
    println!("{}", report.summary());

    println!("\nper-seed cap/param (dense vs partitioned):");
    for r in &report.per_seed {
        println!(
            "  seed: dense={:.3e} partitioned={:.3e} -> {}",
            r.dense_capability_per_param,
            r.partitioned_capability_per_param,
            if r.h01_holds { "holds" } else { "refuted" }
        );
    }
}
