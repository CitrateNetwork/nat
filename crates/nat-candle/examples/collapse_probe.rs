//! Does a zone collapse to zero merge share during training, and at what scale?
//!
//! Scoping input for the H-01 re-run: if collapse happens at every rung, every
//! rung's numbers were measured with a dead zone. If it only appears above some
//! width, the lower rungs are still valid.
//!
//! Trains a small model with merge_floor = 0 (the ORIGINAL pure softmax) and
//! reports each zone's share as it goes.

use candle_core::DType;
use nat_candle::autoreg::{AutoregConfig, AutoregLm};
use nat_candle::corpus::next_byte_windows;
use nat_types::ZoneId;

fn main() -> anyhow::Result<()> {
    let nat = std::env::args().nth(1).unwrap_or_else(|| ".".into());
    let d: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(64);
    let steps: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(30);
    let floor: f64 = std::env::args().nth(4).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let bf16 = std::env::args().nth(5).map(|s| s == "bf16").unwrap_or(false);

    let corpus = std::path::Path::new(&nat).join(
        "corpus/values-spine/corpus-v6/c64a034b203c4b1cb8c74944b934c4c36783a77fd4e56d63b785b475d22433cb",
    );
    let mut shards = Vec::new();
    for i in 0..16u32 {
        let p = corpus.join(format!("shard_{i:04}.json"));
        shards.push(serde_json::from_str::<nat_data::manifest::Shard>(
            &std::fs::read_to_string(p)?,
        )?);
    }

    let zones = ZoneId::LEARNED.to_vec();
    let cfg = AutoregConfig {
        zones: zones.clone(),
        vocab: 256,
        seq_len: 64,
        d,
        tau: 1.0,
        merge_floor: floor,
        seed: 2026,
    };
    let dt = if bf16 { DType::BF16 } else { DType::F32 };
    let mut m = AutoregLm::new_with_dtype(&cfg, dt)?;
    println!("d={d} params={} floor={floor} steps={steps} dtype={dt:?} on {}", m.param_count(), m.backend());

    let (ids, _t) = next_byte_windows(&shards, cfg.seq_len, 512, m.device())?;

    let hdr: Vec<String> = zones.iter().map(|z| format!("{:>9}", z.as_str())).collect();
    println!("\n  {:<6}{}", "step", hdr.join(""));
    for step in 0..=steps {
        if step > 0 {
            m.train_minibatched(&ids, 1, 32, 3e-3, step as u64)?;
        }
        let w = m.zone_merge_weights(&ids)?;
        let cells: Vec<String> = w.iter().map(|s| format!("{s:>9.5}")).collect();
        let dead = w.iter().filter(|s| **s < 1e-6).count();
        if step % 5 == 0 || dead > 0 {
            println!(
                "  {:<6}{}{}",
                step,
                cells.join(""),
                if dead > 0 { format!("   <- {dead} DEAD") } else { String::new() }
            );
        }
        if dead > 0 && step > 0 {
            println!("\n  COLLAPSE at step {step}, d={d}");
            return Ok(());
        }
    }
    println!("\n  no collapse in {steps} steps at d={d}");
    Ok(())
}
