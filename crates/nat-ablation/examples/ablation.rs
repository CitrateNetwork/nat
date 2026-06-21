//! Run the H-01 ablation and print the report.
//!
//!   cargo run -p nat-ablation --example ablation
//!
//! At this (CPU, synthetic) scale the verdict is illustrative, not conclusive —
//! the real H-01 answer is the DGX run with the full NatModel and real corpus.

fn main() {
    let report =
        nat_ablation::run_ablation(&nat_ablation::AblationConfig::default()).expect("ablation run");
    println!("{}", report.summary());
}
