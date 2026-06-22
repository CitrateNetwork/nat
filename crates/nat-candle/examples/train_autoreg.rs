//! Train the per-position autoregressive NAT LM (WP-D7) on a real corpus.
//!
//!   scripts/dgx-gpu.sh run -p nat-candle --features cuda --example train_autoreg -- <corpus-dir>
//!
//! Each sequence yields seq_len-1 next-token predictions, so the corpus is used far
//! more efficiently than the single-output next-byte model. Reports held-out
//! bits/byte; uniform = 8.0.

use nat_candle::autoreg::{AutoregConfig, AutoregLm};
use nat_candle::corpus::sequences_from_dir;

fn main() {
    let dir = match std::env::args().nth(1) {
        Some(d) => d,
        None => {
            eprintln!("usage: train_autoreg <corpus-dir>");
            std::process::exit(2);
        }
    };
    let cfg = AutoregConfig::byte_3zone();
    let mut model = AutoregLm::new(&cfg).unwrap();
    println!(
        "autoregressive NAT LM: backend={} params={} seq_len={} zones={}",
        model.backend(),
        model.param_count(),
        cfg.seq_len,
        cfg.zones.len()
    );

    let ids = sequences_from_dir(
        std::path::Path::new(&dir),
        cfg.seq_len,
        30_000,
        model.device(),
    )
    .unwrap();
    let n = ids.dims2().unwrap().0;
    let n_tr = n * 4 / 5;
    let xtr = ids.narrow(0, 0, n_tr).unwrap();
    let xva = ids.narrow(0, n_tr, n - n_tr).unwrap();
    println!(
        "sequences: {n_tr} train / {} val (×{} next-token predictions each)",
        n - n_tr,
        cfg.seq_len - 1
    );

    let bits = |nats: f32| nats / std::f32::consts::LN_2;
    let before = model.loss_on(&xva).unwrap();
    println!(
        "  epoch 0: held-out {before:.4} nats ({:.3} bits/byte; uniform = 8.0)",
        bits(before)
    );
    for epoch in 0..8 {
        model
            .train_minibatched(&xtr, 1, 64, 0.003, 2026 ^ epoch as u64)
            .unwrap();
        let l = model.loss_on(&xva).unwrap();
        println!(
            "  epoch {}: held-out {:.4} nats ({:.3} bits/byte)",
            epoch + 1,
            l,
            bits(l)
        );
    }
}
