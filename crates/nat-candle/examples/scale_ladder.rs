//! Scale ladder toward L2 (DATA-S1): train the byte-LM at increasing model sizes
//! on the same real corpus and report held-out bits/byte vs parameter count. If
//! loss falls as the model grows, the zone architecture *scales* on real data —
//! the evidence the scale ladder exists to produce before committing L2 compute.
//!
//!   scripts/dgx-gpu.sh run -p nat-candle --features cuda --example scale_ladder -- <corpus-dir>
//!
//! NOTE: the model still predicts one next byte per fixed context (single-output);
//! per-position autoregression (WP-D7) is the efficiency step for true L2 scale.

use nat_candle::corpus::windows_from_dir;
use nat_candle::train_loop::{NatTrainConfig, NatTrainModel};

fn main() {
    let dir = match std::env::args().nth(1) {
        Some(d) => d,
        None => {
            eprintln!("usage: scale_ladder <corpus-dir>");
            std::process::exit(2);
        }
    };
    let bits = |nats: f32| nats / std::f32::consts::LN_2;

    let rungs = [
        ("S 3-zone", NatTrainConfig::byte_lm_3zone()),
        ("M 3-zone", NatTrainConfig::byte_lm_medium()),
        ("L 5-zone", NatTrainConfig::byte_lm_large()),
    ];

    println!("scale ladder on {dir}\n  rung        params   zones  held-out (bits/byte)");
    for (name, cfg) in rungs {
        let mut model = NatTrainModel::new(&cfg).unwrap();
        let (ids, targets) = windows_from_dir(
            std::path::Path::new(&dir),
            cfg.seq_len,
            120_000,
            model.device(),
        )
        .unwrap();
        let n = ids.dims2().unwrap().0;
        let n_tr = n * 4 / 5;
        let xtr = ids.narrow(0, 0, n_tr).unwrap();
        let ytr = targets.narrow(0, 0, n_tr).unwrap();
        let xva = ids.narrow(0, n_tr, n - n_tr).unwrap();
        let yva = targets.narrow(0, n_tr, n - n_tr).unwrap();
        for epoch in 0..5 {
            model
                .train_minibatched(&xtr, &ytr, 1, 256, 0.003, 2026 ^ epoch as u64)
                .unwrap();
        }
        let val = model.loss_on(&xva, &yva).unwrap();
        println!(
            "  {:<10}  {:>7}   {:>3}    {:.3}",
            name,
            model.param_count(),
            cfg.zones.len(),
            bits(val)
        );
    }
}
