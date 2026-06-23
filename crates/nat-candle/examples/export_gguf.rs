//! Train the autoregressive NAT LM briefly on a corpus, then export it to GGUF
//! (g3-gguf). Prints the file size and verifies it reads back.
//!
//!   scripts/dgx-gpu.sh run -p nat-candle --features cuda --example export_gguf -- <corpus-dir> <out.gguf>

use nat_candle::autoreg::{AutoregConfig, AutoregLm};
use nat_candle::corpus::sequences_from_dir;

fn main() {
    let mut args = std::env::args().skip(1);
    let (dir, out) = match (args.next(), args.next()) {
        (Some(d), Some(o)) => (d, o),
        _ => {
            eprintln!("usage: export_gguf <corpus-dir> <out.gguf>");
            std::process::exit(2);
        }
    };
    let cfg = AutoregConfig::byte_3zone();
    let mut model = AutoregLm::new(&cfg).unwrap();
    let ids = sequences_from_dir(
        std::path::Path::new(&dir),
        cfg.seq_len,
        20_000,
        model.device(),
    )
    .unwrap();
    model.train_minibatched(&ids, 2, 64, 0.003, 2026).unwrap();
    println!("trained {} params", model.param_count());

    let path = std::path::Path::new(&out);
    model.export_gguf(path).unwrap();
    let bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let names = nat_candle::gguf::tensor_names(path).unwrap();
    println!(
        "exported GGUF: {out} ({} bytes, {} tensors) — reads back OK",
        bytes,
        names.len()
    );
}
