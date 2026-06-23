//! Train the autoregressive NAT LM on **BPE** tokens (WP-D5 payoff).
//!
//!   nat-corpus train-bpe --input <jsonl> --vocab 1024 --out bpe.json
//!   scripts/dgx-gpu.sh run -p nat-candle --features cuda --example train_autoreg_bpe -- <corpus-dir> <bpe.json>
//!
//! Because BPE compresses (~2 bytes/token), a `seq_len`-token window covers ~2×
//! more text than a byte window. We report held-out bits/token AND bits/byte
//! (= bits/token ÷ bytes-per-token), so it's directly comparable to the byte LM.

use nat_candle::autoreg::{AutoregConfig, AutoregLm};
use nat_candle::corpus::sequence_windows_bpe;
use nat_data::bpe::Bpe;
use nat_data::persist::read_shards;

fn main() {
    let mut args = std::env::args().skip(1);
    let (dir, bpe_path) = match (args.next(), args.next()) {
        (Some(d), Some(b)) => (d, b),
        _ => {
            eprintln!("usage: train_autoreg_bpe <corpus-dir> <bpe.json>");
            std::process::exit(2);
        }
    };
    let bpe = Bpe::load(std::path::Path::new(&bpe_path)).unwrap();
    let shards = read_shards(std::path::Path::new(&dir)).unwrap();

    // Corpus compression ratio (bytes per BPE token) for the bits/byte conversion.
    let (mut bytes, mut toks) = (0u64, 0u64);
    for s in &shards {
        for d in &s.docs {
            bytes += d.text.len() as u64;
            toks += bpe.encode(&d.text).len() as u64;
        }
    }
    let bytes_per_token = bytes as f64 / toks.max(1) as f64;

    let cfg = AutoregConfig {
        vocab: bpe.vocab_size(),
        ..AutoregConfig::byte_3zone()
    };
    let mut model = AutoregLm::new(&cfg).unwrap();
    println!(
        "BPE autoregressive LM: params={} vocab={} seq_len={} ({:.2} bytes/token)",
        model.param_count(),
        cfg.vocab,
        cfg.seq_len,
        bytes_per_token
    );

    let ids = sequence_windows_bpe(&shards, &bpe, cfg.seq_len, 30_000, model.device()).unwrap();
    let n = ids.dims2().unwrap().0;
    let n_tr = n * 4 / 5;
    let xtr = ids.narrow(0, 0, n_tr).unwrap();
    let xva = ids.narrow(0, n_tr, n - n_tr).unwrap();
    println!("sequences: {n_tr} train / {} val", n - n_tr);

    let bpb = |nats: f32| (nats / std::f32::consts::LN_2) as f64 / bytes_per_token;
    for epoch in 0..8 {
        model
            .train_minibatched(&xtr, 1, 64, 0.003, 2026 ^ epoch as u64)
            .unwrap();
        // Batched eval: a single full-val forward materializes a (n_val, seq, vocab)
        // logit tensor that OOMs the GPU at large vocab. Same number, bounded memory.
        let l = model.loss_on_batched(&xva, 64).unwrap();
        println!(
            "  epoch {}: held-out {:.4} nats/token ({:.3} bits/byte)",
            epoch + 1,
            l,
            bpb(l)
        );
    }
}
