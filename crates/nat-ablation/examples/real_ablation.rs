//! The conclusive H-01 ablation (WP-5): the **real** trainable NatModel vs an
//! equal-param dense transformer, multi-seed, under ADR-0005.
//!
//!   cargo run -p nat-ablation --example real_ablation                              # CPU
//!   scripts/dgx-gpu.sh run -p nat-ablation --features cuda --example real_ablation # DGX GPU
//!
//! The verdict here is the real-model read on the synthetic-but-structured task
//! (binned token-sum). Honest posture: the final word needs real-corpus data; if
//! partitioned < dense at equal params, H-01 is refuted — the harness says so.

fn main() {
    let cfg = nat_ablation::real::RealAblationConfig::scaled();
    let seeds = [1u64, 2, 3, 4, 5];
    println!(
        "running REAL H-01 ablation: zones={:?} d_emb={} seq={} steps={} over {} seeds",
        cfg.nat.zones,
        cfg.nat.d_emb,
        cfg.nat.seq_len,
        cfg.steps,
        seeds.len()
    );

    let report = nat_ablation::real::run_real_ablation_seeds(&cfg, &seeds).expect("real ablation");
    println!("{}", report.summary());

    println!("\nper-seed cap/param (nat vs dense):");
    for r in &report.per_seed {
        println!(
            "  nat={:.3e} dense={:.3e} -> {}",
            r.nat_capability_per_param,
            r.dense_capability_per_param,
            if r.h01_holds { "holds" } else { "refuted" }
        );
    }
}
