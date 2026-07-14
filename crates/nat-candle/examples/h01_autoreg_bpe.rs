//! H-01 on the WP-D7 architecture, scaled toward L2.
//!
//! Every prior H-01 read used the *single-output* byte-LM (`NatTrainModel`, vocab 256,
//! ~20K params). This runs H-01 on the architecture we actually intend to scale: the
//! **per-position autoregressive LM** (`AutoregLm`) on **BPE-4096** tokens, at ~1M
//! params — ~50x L1-S, ~9x L1-L. The NAT arm (5 zones, SM/CB SSM + HP/PF/CX attention,
//! per-position soft merge) is compared against a **param-matched per-position dense
//! Transformer** (`AutoregDenseLm`: one causal-attention block + FFN, no partitioning).
//! Both share a bit-identical embedding + readout (same vocab, same width `d`), so any
//! held-out difference is attributable to zone partitioning, not parameter count
//! (ADR-0005). Both train with the identical mini-batch protocol on the same corpus.
//!
//! Honest scope: this is a scale-UP toward L2, not L2 itself — true L2 (~10B params,
//! committed compute, gate g5-l2) is owner-gated. At BPE-4096 the embedding+readout
//! dominate the budget, so partitioning governs a minority of params; a hold here is a
//! per-parameter signal in the cores, not a whole-model claim.
//!
//!   scripts/dgx-gpu.sh run -p nat-candle --features cuda --release \
//!     --example h01_autoreg_bpe -- <corpus-dir> <bpe.json> [target_params]

use candle_core::DType;
use nat_candle::autoreg::{AutoregConfig, AutoregDenseLm, AutoregLm};
use nat_candle::corpus::sequence_windows_bpe;
use nat_data::bpe::Bpe;
use nat_data::persist::read_shards;
use nat_types::ZoneId;

const SEQ_LEN: usize = 64;
const EPOCHS: usize = 8;
const BATCH: usize = 64;
const LR: f64 = 0.003;

fn nat_cfg(d: usize, vocab: usize, seed: u64) -> AutoregConfig {
    AutoregConfig {
        zones: ZoneId::LEARNED.to_vec(),
        vocab,
        seq_len: SEQ_LEN,
        d,
        tau: 1.0,
        seed,
    }
}

fn nat_params(d: usize, vocab: usize) -> usize {
    AutoregLm::new(&nat_cfg(d, vocab, 1))
        .map(|m| m.param_count())
        .unwrap_or(usize::MAX)
}

fn dense_params(d: usize, d_ff: usize, vocab: usize) -> usize {
    AutoregDenseLm::new(vocab, SEQ_LEN, d, d_ff, 1)
        .map(|m| m.param_count())
        .unwrap_or(usize::MAX)
}

/// Smallest-error `x` in `[lo, hi]` minimizing `|f(x) - target|`, f monotonic increasing.
fn bsearch(lo: usize, hi: usize, target: usize, f: impl Fn(usize) -> usize) -> usize {
    let (mut a, mut b) = (lo, hi);
    while a < b {
        let mid = (a + b) / 2;
        if f(mid) < target {
            a = mid + 1;
        } else {
            b = mid;
        }
    }
    if a > lo && target.abs_diff(f(a - 1)) <= target.abs_diff(f(a)) {
        a - 1
    } else {
        a
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let (dir, bpe_path) = match (args.next(), args.next()) {
        (Some(d), Some(b)) => (d, b),
        _ => {
            eprintln!(
                "usage: h01_autoreg_bpe <corpus-dir> <bpe.json> [target_params] [max_windows] [n_seeds]"
            );
            std::process::exit(2);
        }
    };
    let target: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);
    // max_windows caps training sequences. Scale it with target params on a larger
    // corpus, or the bigger model starves and overfits (the whole point of corpus-v4).
    let max_windows: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(30_000);
    let n_seeds: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(5);
    // Crash recovery (WP-S1 interim): the ladder path has no checkpoint/resume, so a
    // killed run loses every unfinished seed. NAT_SEED_START lets a re-launch skip
    // seeds whose rows already landed in the log instead of re-burning days of GPU.
    let seed_start: u64 = std::env::var("NAT_SEED_START")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let seeds: Vec<u64> = (seed_start..=n_seeds).collect();
    // Both arms share the same dtype (ADR-0005). NAT_DTYPE=bf16 enables the
    // mixed-precision throughput path (SCALE-S1 WP-S2); default f32.
    let dtype = match std::env::var("NAT_DTYPE").as_deref() {
        Ok("bf16") => DType::BF16,
        Ok("f16") => DType::F16,
        _ => DType::F32,
    };

    let bpe = Bpe::load(std::path::Path::new(&bpe_path)).unwrap();
    let vocab = bpe.vocab_size();
    let shards = read_shards(std::path::Path::new(&dir)).unwrap();

    // Size the NAT arm to ~target params, then param-match the dense FFN at the same d.
    // The d ceiling must clear what big vocabs need (at BPE-16k the embedding dominates,
    // so a 32M target wants d~800, well above the old 512 cap).
    let d = bsearch(8, 4096, target, |d| nat_params(d, vocab));
    let nat_p = nat_params(d, vocab);
    let d_ff = bsearch(1, 8192, nat_p, |f| dense_params(d, f, vocab));
    let dense_p = dense_params(d, d_ff, vocab);

    // Compression (bytes/token) for the bits/byte conversion.
    let (mut bytes, mut toks) = (0u64, 0u64);
    for s in &shards {
        for doc in &s.docs {
            bytes += doc.text.len() as u64;
            toks += bpe.encode(&doc.text).len() as u64;
        }
    }
    let bpt = bytes as f64 / toks as f64;
    let bpb = |nats: f32| (nats / std::f32::consts::LN_2) as f64 / bpt;

    println!(
        "H-01 @ WP-D7 (per-position autoreg, BPE vocab {vocab}, {bpt:.3} bytes/token) — backend {}",
        nat_candle::device::backend_label()
    );
    println!(
        "  NAT 5-zone d={d} params={nat_p}  vs  dense d={d} d_ff={d_ff} params={dense_p}  (target ~{target}, dtype {dtype:?})"
    );

    // Build BPE windows once; same 80/20 split feeds both arms every seed.
    let ids = sequence_windows_bpe(
        &shards,
        &bpe,
        SEQ_LEN,
        max_windows,
        AutoregLm::new(&nat_cfg(d, vocab, 1)).unwrap().device(),
    )
    .unwrap();
    let n = ids.dims2().unwrap().0;
    let n_tr = n * 4 / 5;
    let xtr = ids.narrow(0, 0, n_tr).unwrap();
    let xva = ids.narrow(0, n_tr, n - n_tr).unwrap();
    println!(
        "  sequences: {n_tr} train / {} val (cap {max_windows}); {EPOCHS} epochs, batch {BATCH}, lr {LR}, {} seeds\n",
        n - n_tr,
        seeds.len()
    );
    println!(
        "  {:>4}  {:>10}  {:>10}  {:>8}",
        "seed", "nat b/byte", "dense b/byte", "NAT<dense"
    );

    let cap_per_param = |loss: f32, p: usize| (1.0f64 / (loss as f64 + 1e-6)) / p as f64;
    let (mut nat_cpps, mut dense_cpps) = (Vec::new(), Vec::new());
    let mut holds = 0usize;

    // NAT_CKPT_DIR (WP-S1): when set, both arms train through the checkpointed loop
    // (identical math, per-epoch model.safetensors+meta.json under <dir>/<arm>-seed<N>)
    // so a host reboot mid-rung loses at most one epoch, not a 40h+ arm. Unset → the
    // original in-memory path, byte-identical behavior to every prior ladder run.
    let ckpt_root = std::env::var("NAT_CKPT_DIR").ok().map(std::path::PathBuf::from);

    for &seed in &seeds {
        let mut nat = AutoregLm::new_with_dtype(&nat_cfg(d, vocab, seed), dtype).unwrap();
        match &ckpt_root {
            Some(root) => nat
                .train_minibatched_checkpointed(
                    &xtr,
                    EPOCHS,
                    BATCH,
                    LR,
                    seed,
                    &root.join(format!("nat-seed{seed}")),
                )
                .unwrap(),
            None => nat.train_minibatched(&xtr, EPOCHS, BATCH, LR, seed).unwrap(),
        }
        let nat_loss = nat.loss_on_batched(&xva, BATCH).unwrap();

        let mut dense =
            AutoregDenseLm::new_with_dtype(vocab, SEQ_LEN, d, d_ff, seed, dtype).unwrap();
        match &ckpt_root {
            Some(root) => dense
                .train_minibatched_checkpointed(
                    &xtr,
                    EPOCHS,
                    BATCH,
                    LR,
                    seed,
                    &root.join(format!("dense-seed{seed}")),
                )
                .unwrap(),
            None => dense
                .train_minibatched(&xtr, EPOCHS, BATCH, LR, seed)
                .unwrap(),
        }
        let dense_loss = dense.loss_on_batched(&xva, BATCH).unwrap();

        nat_cpps.push(cap_per_param(nat_loss, nat_p));
        dense_cpps.push(cap_per_param(dense_loss, dense_p));
        let win = nat_loss < dense_loss;
        if win {
            holds += 1;
        }
        println!(
            "  {seed:>4}  {:>10.4}  {:>10.4}  {:>8}",
            bpb(nat_loss),
            bpb(dense_loss),
            if win { "yes" } else { "no" }
        );
    }

    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let (nat_mean, dense_mean) = (mean(&nat_cpps), mean(&dense_cpps));
    // ADR-0005 verdict: NAT capability/param within 0.95 of (here, exceeding) dense.
    let h01 = nat_mean >= dense_mean * 0.95;
    println!(
        "\n  mean cap/param: NAT {nat_mean:.4e}  dense {dense_mean:.4e}  ->  H-01 {}  ({holds}/{} seeds NAT<dense)",
        if h01 { "HOLDS" } else { "REFUTED" },
        seeds.len()
    );
}
