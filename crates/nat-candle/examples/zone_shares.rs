//! Measure each zone's actual merge share on a real checkpoint + real corpus.
//!
//! The merge is a per-position softmax over zone scores, so a zone's gradient
//! scales with its share. This prints the shares so a dead zone is visible.

use candle_core::DType;
use nat_candle::autoreg::{AutoregConfig, AutoregLm};
use nat_types::ZoneId;

fn main() -> anyhow::Result<()> {
    let nat = std::env::args().nth(1).unwrap_or_else(|| ".".into());
    let ckpt = std::path::Path::new(&nat).join("checkpoints-64m/nat-seed2");
    let corpus = std::path::Path::new(&nat).join(
        "corpus/values-spine/corpus-v6/c64a034b203c4b1cb8c74944b934c4c36783a77fd4e56d63b785b475d22433cb",
    );

    let floor: f64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(nat_candle::autoreg::DEFAULT_MERGE_FLOOR);
    let zones = vec![ZoneId::SM, ZoneId::CB, ZoneId::HP, ZoneId::PF, ZoneId::CX];
    let cfg = AutoregConfig { zones: zones.clone(), vocab: 16384, seq_len: 128, d: 1183, tau: 1.0, merge_floor: floor, seed: 2026 };
    println!("merge_floor = {floor}");
    let mut m = AutoregLm::new_with_dtype(&cfg, DType::BF16)?;
    m.load(&ckpt)?;
    println!("loaded 64M checkpoint on {}", m.backend());

    // Real documents from the real corpus.
    let mut shards = Vec::new();
    for i in 0..8u32 {
        let p = corpus.join(format!("shard_{i:04}.json"));
        shards.push(serde_json::from_str::<nat_data::manifest::Shard>(&std::fs::read_to_string(p)?)?);
    }
    let (ids, _t) = nat_candle::corpus::next_byte_windows(&shards, cfg.seq_len, 32, m.device())?;

    let w = m.zone_merge_weights(&ids)?;
    println!("\n  zone   merge share");
    for (z, s) in zones.iter().zip(w.iter()) {
        let bar = "#".repeat(((*s) * 50.0).round() as usize);
        println!("  {:<5} {:>10.6}  {}", z.as_str(), s, bar);
    }
    let total: f32 = w.iter().sum();
    println!("\n  sum = {total:.6} (softmax, so ~1.0)");
    Ok(())
}
