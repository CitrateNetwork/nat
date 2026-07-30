//! Throughput micro-benchmark — SCALE-S1 WP-S3.
//!
//! Measures training tokens/sec for the per-position `AutoregLm` at a given parameter
//! scale, so we can (a) see where the ladder's wall-clock actually goes as params grow,
//! and (b) quantify the bf16 speedup once WP-S2 lands (add a `--dtype` arm then).
//!
//! Run on the GPU for meaningful numbers (pair with `nvidia-smi` for peak memory):
//!   scripts/dgx-gpu.sh run -p nat-candle --features cuda --release \
//!     --example bench_throughput -- [target_params] [batch] [seq] [steps] [vocab]
//!
//! On CPU (no --features cuda) it only smoke-tests the harness — the numbers are not
//! representative of GPU throughput.

use candle_core::{DType, Tensor};
use nat_candle::autoreg::{AutoregConfig, AutoregDenseLm, AutoregLm};
use nat_candle::device::backend_label;
use nat_types::ZoneId;
use std::time::Instant;

fn cfg_for(d: usize, vocab: usize, seq: usize) -> AutoregConfig {
    AutoregConfig {
        zones: ZoneId::LEARNED.to_vec(),
        vocab,
        seq_len: seq,
        d,
        tau: 1.0,
        merge_floor: nat_candle::autoreg::DEFAULT_MERGE_FLOOR,
        seed: 1,
    }
}

fn params_at(d: usize, vocab: usize, seq: usize) -> usize {
    AutoregLm::new(&cfg_for(d, vocab, seq))
        .map(|m| m.param_count())
        .unwrap_or(usize::MAX)
}

/// Smallest-error `d` in `[lo, hi]` for `target` params (param count is monotonic in d).
fn size_d(lo: usize, hi: usize, target: usize, vocab: usize, seq: usize) -> usize {
    let (mut a, mut b) = (lo, hi);
    while a < b {
        let mid = (a + b) / 2;
        if params_at(mid, vocab, seq) < target {
            a = mid + 1;
        } else {
            b = mid;
        }
    }
    a
}

fn main() {
    let mut args = std::env::args().skip(1);
    let target: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8_000_000);
    let batch: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    let seq: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    let steps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(20);
    let vocab: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(4096);

    let dtype = match std::env::var("NAT_DTYPE").as_deref() {
        Ok("bf16") => DType::BF16,
        Ok("f16") => DType::F16,
        _ => DType::F32,
    };

    let d = size_d(8, 2048, target, vocab, seq);
    let cfg = cfg_for(d, vocab, seq);

    // NAT_ARM=dense benches the H-01 DENSE baseline instead of the zone arm.
    // Both arms are trained in every ablation, so a wall-clock estimate that
    // benches only the zone arm and doubles it is assuming parity it never
    // measured.
    let dense = std::env::var("NAT_ARM").as_deref() == Ok("dense");
    // Only the arm under test is built — allocating both would put the other
    // arm's parameters and optimizer state on the device for no reason.
    let mut model = (!dense).then(|| AutoregLm::new_with_dtype(&cfg, dtype).unwrap());
    // d_ff sized as the 64M run did: dense d=1183, d_ff=8192, ~6.9x d.
    let mut dense_model = dense
        .then(|| AutoregDenseLm::new_with_dtype(vocab, seq, d, d * 69 / 10, 1, dtype).unwrap());
    let params = match (&model, &dense_model) {
        (Some(m), _) => m.param_count(),
        (_, Some(m)) => m.param_count(),
        _ => unreachable!("exactly one arm is built"),
    };

    // Deterministic synthetic windows (no rng dep): `steps` batches of `batch` rows.
    let n = batch * steps;
    let ids = Tensor::from_vec(
        (0..(n * seq) as u64)
            .map(|i| (i.wrapping_mul(2_654_435_761) % vocab as u64) as u32)
            .collect::<Vec<_>>(),
        (n, seq),
        match (&model, &dense_model) {
            (Some(m), _) => m.device(),
            (_, Some(m)) => m.device(),
            _ => unreachable!("exactly one arm is built"),
        },
    )
    .unwrap();

    // Warm up (primes lazy init / CUDA kernels) on one batch, then time one full pass.
    let warm = ids.narrow(0, 0, batch).unwrap();
    train(&mut model, &mut dense_model, &warm, batch, 0.0, 1);

    let t = Instant::now();
    train(&mut model, &mut dense_model, &ids, batch, 0.003, 7);
    let dt = t.elapsed().as_secs_f64();

    let toks = (n * seq) as f64;
    println!(
        "bench_throughput — backend {} dtype {dtype:?} arm {}",
        backend_label(),
        if dense { "dense" } else { "nat" }
    );
    println!("  d={d} params={params} vocab={vocab} | {steps} steps × batch {batch} × seq {seq}");
    println!(
        "  {toks:.0} tokens in {dt:.3}s = {:.0} tok/s ({:.1} ms/step)",
        toks / dt,
        dt * 1000.0 / steps as f64
    );
    println!("  (GPU run for representative numbers; pair with nvidia-smi for peak memory)");
}

/// Step whichever arm was built. Both expose the same `train_minibatched`
/// shape, so the only difference between the arms is the model itself.
fn train(
    nat: &mut Option<AutoregLm>,
    dense: &mut Option<AutoregDenseLm>,
    ids: &Tensor,
    batch: usize,
    lr: f64,
    seed: u64,
) {
    match (nat, dense) {
        (Some(m), _) => m.train_minibatched(ids, 1, batch, lr, seed).unwrap(),
        (_, Some(m)) => m.train_minibatched(ids, 1, batch, lr, seed).unwrap(),
        _ => unreachable!("exactly one arm is built"),
    }
}
