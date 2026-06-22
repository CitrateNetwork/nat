//! WP-D6 — the CONCLUSIVE H-01 on real data: the real NatModel vs an equal-param
//! dense transformer, both mini-batch-trained on a real corpus (next-byte LM),
//! multi-seed, under ADR-0005. This is the bet-deciding verdict on real text.
//!
//!   scripts/dgx-gpu.sh run -p nat-ablation --features cuda --example real_h01_corpus -- <corpus-dir>
//!
//! Pass the corpus dir built by `nat-corpus run` (e.g. via scripts/fetch-values-spine.sh).
//! Honest posture: if partitioned < dense at equal params on real data, H-01 is
//! refuted — say so and change course.

fn main() {
    let dir = match std::env::args().nth(1) {
        Some(d) => d,
        None => {
            eprintln!("usage: real_h01_corpus <corpus-dir>");
            std::process::exit(2);
        }
    };
    let seeds = [1u64, 2, 3, 4, 5];
    println!(
        "CONCLUSIVE H-01 on real data: corpus={dir}, {} seeds",
        seeds.len()
    );

    let report = nat_ablation::real::run_real_corpus_ablation_seeds(
        std::path::Path::new(&dir),
        6,      // epochs
        256,    // batch size
        0.003,  // lr
        80_000, // max windows
        0.05,   // ADR-0005 param tolerance
        &seeds,
    )
    .expect("real corpus ablation");

    println!("{}", report.summary());
    println!("\nper-seed (held-out cap/param, nat vs dense):");
    for r in &report.per_seed {
        println!(
            "  nat={:.3e} (loss {:.3}) dense={:.3e} (loss {:.3}) -> {}",
            r.nat_capability_per_param,
            r.nat_final_loss,
            r.dense_capability_per_param,
            r.dense_final_loss,
            if r.h01_holds { "holds" } else { "refuted" }
        );
    }
}
