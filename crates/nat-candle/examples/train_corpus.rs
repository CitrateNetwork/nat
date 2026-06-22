//! Train the 3-zone NAT model on the real seed corpus (next-byte LM, DATA-S1).
//!
//!   scripts/dgx-gpu.sh run -p nat-candle --features cuda --example train_corpus  # DGX GPU
//!   cargo run -p nat-candle --example train_corpus                                # CPU (slow)
//!
//! Reports held-out cross-entropy in nats and bits/byte; the uniform-byte baseline
//! is ln(256) ≈ 5.545 nats = 8.0 bits/byte.

use nat_candle::corpus::seed_windows;
use nat_candle::train_loop::{NatTrainConfig, NatTrainModel};

fn main() {
    let cfg = NatTrainConfig::byte_lm_3zone();
    let mut model = NatTrainModel::new(&cfg).unwrap();
    println!(
        "backend = {}  params = {}",
        model.backend(),
        model.param_count()
    );

    let (ids, targets) = seed_windows(cfg.seq_len, 3000, model.device()).unwrap();
    let n = ids.dims2().unwrap().0;
    let n_tr = n * 4 / 5;
    let xtr = ids.narrow(0, 0, n_tr).unwrap();
    let ytr = targets.narrow(0, 0, n_tr).unwrap();
    let xva = ids.narrow(0, n_tr, n - n_tr).unwrap();
    let yva = targets.narrow(0, n_tr, n - n_tr).unwrap();
    println!("windows: {n_tr} train / {} val", n - n_tr);

    let bits = |nats: f32| nats / std::f32::consts::LN_2;
    let before = model.loss_on(&xva, &yva).unwrap();
    for epoch in 0..8 {
        model.train(&xtr, &ytr, 100, 0.003).unwrap();
        let l = model.loss_on(&xva, &yva).unwrap();
        println!(
            "  epoch {}: held-out {:.4} nats ({:.3} bits/byte)",
            epoch + 1,
            l,
            bits(l)
        );
    }
    let after = model.loss_on(&xva, &yva).unwrap();
    println!(
        "held-out: {before:.4} -> {after:.4} nats  ({:.3} -> {:.3} bits/byte; uniform = 8.0)",
        bits(before),
        bits(after)
    );
}
