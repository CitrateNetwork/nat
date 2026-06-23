//! Param-matched BPE vocab sweep (WP-D5 — isolating the tokenizer).
//!
//! The plain vocab sweep is confounded: a bigger BPE vocab enlarges the embedding
//! AND output tables, so a vocab-8192 LM has ~6.4x the parameters of a vocab-1024
//! LM at the same width. Its lower bits/byte is then mostly "more parameters," not
//! a better tokenizer.
//!
//! This sweep removes that confound: it fixes a total parameter BUDGET and, for each
//! vocab, binary-searches the model width `d` so every model lands at ~the same param
//! count. The only thing that varies is how that fixed budget splits between
//! token-embeddings (vocab-tied) and compute-width (the cores). So a lower held-out
//! bits/byte at equal params is attributable to the tokenizer choice, not model size.
//!
//!   scripts/dgx-gpu.sh run -p nat-candle --features cuda --release \
//!     --example param_matched_bpe_sweep -- <corpus-dir> <target_params> <bpe1.json> [bpe2.json ...]

use nat_candle::autoreg::{AutoregConfig, AutoregLm};
use nat_candle::corpus::sequence_windows_bpe;
use nat_data::bpe::Bpe;
use nat_data::persist::read_shards;

/// Exact param count for a (vocab, d) model — built and measured, not estimated
/// (the cores scale ~d^2, so an analytic formula would be fragile).
fn params_for(vocab: usize, d: usize) -> usize {
    let cfg = AutoregConfig {
        d,
        vocab,
        ..AutoregConfig::byte_3zone()
    };
    AutoregLm::new(&cfg)
        .map(|m| m.param_count())
        .unwrap_or(usize::MAX)
}

/// Smallest-error width `d` (in [8, 512]) whose param count is closest to `target`.
fn pick_d(vocab: usize, target: usize) -> usize {
    let (mut lo, mut hi) = (8usize, 512usize);
    while lo < hi {
        let mid = (lo + hi) / 2;
        if params_for(vocab, mid) < target {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    // `lo` is the smallest d with params >= target; pick whichever of lo/lo-1 is closer.
    if lo > 8 {
        let below = params_for(vocab, lo - 1);
        let at = params_for(vocab, lo);
        if target.abs_diff(below) <= target.abs_diff(at) {
            lo - 1
        } else {
            lo
        }
    } else {
        lo
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next();
    let target: Option<usize> = args.next().and_then(|s| s.parse().ok());
    let bpe_paths: Vec<String> = args.collect();
    let (dir, target) = match (dir, target) {
        (Some(d), Some(t)) if !bpe_paths.is_empty() => (d, t),
        _ => {
            eprintln!(
                "usage: param_matched_bpe_sweep <corpus-dir> <target_params> <bpe1.json> [bpe2.json ...]"
            );
            std::process::exit(2);
        }
    };

    let shards = read_shards(std::path::Path::new(&dir)).unwrap();
    println!(
        "param-matched BPE sweep — target ~{target} params, backend {}",
        nat_candle::device::backend_label()
    );
    println!(
        "{:>6}  {:>4}  {:>9}  {:>11}  {:>12}",
        "vocab", "d", "params", "bytes/token", "bits/byte"
    );

    for bpe_path in &bpe_paths {
        let bpe = Bpe::load(std::path::Path::new(bpe_path)).unwrap();
        let vocab = bpe.vocab_size();
        let d = pick_d(vocab, target);

        // Compression ratio (bytes per BPE token) for the bits/byte conversion.
        let (mut bytes, mut toks) = (0u64, 0u64);
        for s in &shards {
            for doc in &s.docs {
                bytes += doc.text.len() as u64;
                toks += bpe.encode(&doc.text).len() as u64;
            }
        }
        let bytes_per_token = bytes as f64 / toks as f64;

        let cfg = AutoregConfig {
            d,
            vocab,
            ..AutoregConfig::byte_3zone()
        };
        let mut model = AutoregLm::new(&cfg).unwrap();
        let params = model.param_count();

        // Same window cap across vocabs → equal #training sequences (fair budget).
        let ids = sequence_windows_bpe(&shards, &bpe, cfg.seq_len, 30_000, model.device()).unwrap();
        let n = ids.dims2().unwrap().0;
        let n_tr = n * 4 / 5;
        let xtr = ids.narrow(0, 0, n_tr).unwrap();
        let xva = ids.narrow(0, n_tr, n - n_tr).unwrap();

        let bpb = |nats: f32| (nats / std::f32::consts::LN_2) as f64 / bytes_per_token;
        let mut last = f64::NAN;
        for epoch in 0..8 {
            model
                .train_minibatched(&xtr, 1, 64, 0.003, 2026 ^ epoch as u64)
                .unwrap();
            last = bpb(model.loss_on_batched(&xva, 64).unwrap());
        }
        println!("{vocab:>6}  {d:>4}  {params:>9}  {bytes_per_token:>11.3}  {last:>12.4}");
    }
}
