//! GPU smoke probe: confirm the `cuda` build actually resolves a CUDA device and
//! runs a real forward pass on it (no silent CPU fallback). Run with:
//!   cargo run -p nat-candle --features cuda --example gpu_probe
//! On a CPU build it prints "candle-cpu" and exits 0 (nothing to prove).

fn main() {
    let label = nat_candle::device::backend_label();
    let dev = nat_candle::device::device();
    println!("backend_label = {label}");
    println!("device.is_cuda = {}", dev.is_cuda());

    // Run a real forward pass through a Candle core (matmul + bucketize) and a
    // tiny training step, both on whatever device resolved.
    use nat_core::cores::ZoneCore;
    let core = nat_candle::CandleSsmCore::default();
    let slice: Vec<f32> = (0..32).map(|i| i as f32 / 32.0 - 0.5).collect();
    let out = core.forward(&slice);
    println!(
        "ssm forward ok: summary[0]={:.4} conf={:.4}",
        out.summary[0], out.confidence
    );

    let r = nat_candle::train_tiny_zone_head(8, 4, 200).expect("train step");
    println!(
        "train: initial_loss={:.4} final_loss={:.4} converged={}",
        r.initial_loss,
        r.final_loss,
        r.converged()
    );

    #[cfg(feature = "cuda")]
    {
        assert!(
            dev.is_cuda(),
            "cuda feature is on but no GPU device came up — this would silently train on CPU"
        );
        assert_eq!(label, "candle-cuda");
        println!("OK: GPU path confirmed live (candle-cuda).");
    }
}
