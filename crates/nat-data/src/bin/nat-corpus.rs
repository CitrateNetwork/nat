//! `nat-corpus` — the data pipeline runner CLI (HERMES-S1 WP-H3).
//!
//! Hermes (and a human) drive the corpus with this: `RawDoc` JSONL in → the
//! `nat-data` pipeline (license / length / dedup / PII / quality gates) → an
//! on-disk corpus (shards + manifest), with an auditable summary of what was kept
//! and why anything was quarantined.
//!
//!   # write the CC0 seed corpus as JSONL (the format reference)
//!   cargo run -p nat-data --bin nat-corpus -- emit-seed --out seed.jsonl
//!
//!   # run the pipeline on a JSONL file → a corpus directory
//!   cargo run -p nat-data --bin nat-corpus -- run --input seed.jsonl --out ./corpus
//!
//! `run` flags: --input <jsonl> --out <root> [--shard-size N] [--min-quality F]
//!              [--min-len N] [--max-len N] [--near-dup F] [--seed N]

use nat_data::{jsonl, persist, run_pipeline, PipelineConfig, QuarantineReason};
use nat_types::Q16;
use std::collections::BTreeMap;
use std::path::Path;
use std::process::exit;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let rest = if args.len() > 2 { &args[2..] } else { &[] };
    let code = match args.get(1).map(String::as_str) {
        Some("run") => cmd_run(rest),
        Some("emit-seed") => cmd_emit_seed(rest),
        Some("--help") | Some("-h") | None => {
            usage();
            0
        }
        Some(other) => {
            eprintln!("nat-corpus: unknown command '{other}'\n");
            usage();
            2
        }
    };
    exit(code);
}

fn usage() {
    eprintln!(
        "nat-corpus — NAT data pipeline runner (HERMES-S1)\n\n\
         USAGE:\n  \
           nat-corpus run --input <jsonl> --out <corpus-root> [config flags]\n  \
           nat-corpus emit-seed --out <file.jsonl>\n\n\
         run config flags (defaults from PipelineConfig::default):\n  \
           --shard-size N   --min-quality F   --min-len N   --max-len N\n  \
           --near-dup F     --seed N\n"
    );
}

/// Collect `--key value` pairs.
fn flags(args: &[String]) -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    let mut i = 0;
    while i < args.len() {
        if let Some(key) = args[i].strip_prefix("--") {
            let val = args.get(i + 1).cloned().unwrap_or_default();
            m.insert(key.to_string(), val);
            i += 2;
        } else {
            i += 1;
        }
    }
    m
}

fn require<'a>(f: &'a BTreeMap<String, String>, key: &str) -> &'a str {
    match f.get(key) {
        Some(v) if !v.is_empty() => v,
        _ => {
            eprintln!("nat-corpus: missing required --{key}");
            exit(2);
        }
    }
}

fn parse_seed(s: &str) -> Option<u64> {
    if let Some(hex) = s.strip_prefix("0x") {
        u64::from_str_radix(hex, 16).ok()
    } else {
        s.parse().ok()
    }
}

fn cmd_emit_seed(args: &[String]) -> i32 {
    let f = flags(args);
    let out = require(&f, "out");
    match jsonl::write_rawdocs_file(Path::new(out), &nat_data::seed::seed_corpus()) {
        Ok(()) => {
            println!(
                "wrote {} seed docs to {out}",
                nat_data::seed::seed_corpus().len()
            );
            0
        }
        Err(e) => {
            eprintln!("nat-corpus: failed to write {out}: {e}");
            1
        }
    }
}

fn cmd_run(args: &[String]) -> i32 {
    let f = flags(args);
    let input = require(&f, "input").to_string();
    let out = require(&f, "out").to_string();

    let mut cfg = PipelineConfig::default();
    if let Some(v) = f.get("shard-size").and_then(|v| v.parse().ok()) {
        cfg.shard_size = v;
    }
    if let Some(v) = f.get("min-len").and_then(|v| v.parse().ok()) {
        cfg.min_len = v;
    }
    if let Some(v) = f.get("max-len").and_then(|v| v.parse().ok()) {
        cfg.max_len = v;
    }
    if let Some(v) = f.get("seed").and_then(|v| parse_seed(v)) {
        cfg.seed = v;
    }
    if let Some(v) = f.get("min-quality").and_then(|v| v.parse::<f32>().ok()) {
        cfg.min_quality = Q16::from_f32(v);
    }
    if let Some(v) = f.get("near-dup").and_then(|v| v.parse::<f32>().ok()) {
        cfg.near_dup_threshold = Q16::from_f32(v);
    }

    let docs = match jsonl::read_rawdocs_file(Path::new(&input)) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("nat-corpus: reading {input}: {e}");
            return 1;
        }
    };
    println!("read {} docs from {input}", docs.len());

    let output = run_pipeline(docs, &cfg);

    // Group quarantine reasons for an auditable summary.
    let mut reasons: BTreeMap<&str, usize> = BTreeMap::new();
    for q in &output.quarantine {
        *reasons.entry(reason_label(&q.reason)).or_insert(0) += 1;
    }

    let dir = match persist::write_corpus(Path::new(&out), &output) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("nat-corpus: writing corpus to {out}: {e}");
            return 1;
        }
    };

    let m = &output.manifest;
    println!(
        "kept: {} docs in {} shards, {} tokens, aggregate_quality={:.3}",
        m.total_docs,
        m.shard_count,
        m.total_tokens,
        m.aggregate_quality.to_f32()
    );
    if output.quarantine.is_empty() {
        println!("quarantined: 0");
    } else {
        let detail: Vec<String> = reasons.iter().map(|(k, n)| format!("{k} x{n}")).collect();
        println!(
            "quarantined: {} ({})",
            output.quarantine.len(),
            detail.join(", ")
        );
    }
    println!("corpus: {}", dir.display());
    println!("manifest_hash: {}", m.manifest_hash());
    0
}

fn reason_label(r: &QuarantineReason) -> &'static str {
    match r {
        QuarantineReason::UnreviewedLicense(_) => "unreviewed_license",
        QuarantineReason::TooShort => "too_short",
        QuarantineReason::TooLong => "too_long",
        QuarantineReason::LowQuality(_) => "low_quality",
        QuarantineReason::ExactDuplicate => "exact_duplicate",
        QuarantineReason::NearDuplicate => "near_duplicate",
        QuarantineReason::PiiDetected(_) => "pii_detected",
    }
}
