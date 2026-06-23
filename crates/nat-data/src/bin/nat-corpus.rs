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

use nat_data::{
    bpe, code, gutenberg, jsonl, latex, persist, run_pipeline, text, PipelineConfig,
    QuarantineReason,
};
use nat_types::Q16;
use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;
use std::process::exit;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let rest = if args.len() > 2 { &args[2..] } else { &[] };
    let code = match args.get(1).map(String::as_str) {
        Some("run") => cmd_run(rest),
        Some("emit-seed") => cmd_emit_seed(rest),
        Some("from-gutenberg") => cmd_from_gutenberg(rest),
        Some("from-code") => cmd_from_code(rest),
        Some("train-bpe") => cmd_train_bpe(rest),
        Some("from-text") => cmd_from_text(rest),
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
           nat-corpus emit-seed --out <file.jsonl>\n  \
           nat-corpus from-gutenberg --id <N> [--input <file|->] [--out <jsonl|->] [--append] [--target-chars N]\n  \
           nat-corpus from-code --dir <repo> --license <SPDX> [--source <name>] [--out <jsonl|->] [--append] [--target-chars N] [--max-line-len N]\n  \
           nat-corpus train-bpe --input <jsonl> --vocab <N> --out <bpe.json>\n  \
           nat-corpus from-text --input <file|-> --license <SPDX> --source <name> [--id-prefix P] [--strip latex] [--out <jsonl|->] [--append] [--target-chars N]\n\n\
         run config flags (defaults from PipelineConfig::default):\n  \
           --shard-size N   --min-quality F   --min-len N   --max-len N\n  \
           --near-dup F     --seed N\n"
    );
}

/// Collect flags. `--key value` → (key, value); a bare `--key` followed by another
/// flag (or nothing) → (key, "true"), so boolean flags like `--append` work.
fn flags(args: &[String]) -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    let mut i = 0;
    while i < args.len() {
        if let Some(key) = args[i].strip_prefix("--") {
            let next_is_flag = args.get(i + 1).map(|n| n.starts_with("--")).unwrap_or(true);
            if next_is_flag {
                m.insert(key.to_string(), "true".to_string());
                i += 1;
            } else {
                m.insert(key.to_string(), args[i + 1].clone());
                i += 2;
            }
        } else {
            i += 1;
        }
    }
    m
}

/// Today's date as `YYYY-MM-DD` (UTC), dependency-free (Hinnant's civil-from-days).
fn today() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let z = (secs / 86400) as i64 + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = y + if m <= 2 { 1 } else { 0 };
    format!("{y:04}-{m:02}-{d:02}")
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

fn cmd_from_gutenberg(args: &[String]) -> i32 {
    let f = flags(args);
    let id: u32 = match require(&f, "id").parse() {
        Ok(v) => v,
        Err(_) => {
            eprintln!("nat-corpus: --id must be a number");
            return 2;
        }
    };
    let target_chars: usize = f
        .get("target-chars")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1500);
    let fetch_date = f.get("fetch-date").cloned().unwrap_or_else(today);

    // Read the downloaded book text from --input (a file) or stdin.
    let text = match f.get("input").map(String::as_str) {
        Some(p) if p != "-" => match std::fs::read_to_string(p) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("nat-corpus: reading {p}: {e}");
                return 1;
            }
        },
        _ => {
            let mut s = String::new();
            if let Err(e) = std::io::stdin().read_to_string(&mut s) {
                eprintln!("nat-corpus: reading stdin: {e}");
                return 1;
            }
            s
        }
    };

    let docs = gutenberg::to_rawdocs(id, &fetch_date, &text, target_chars);
    if docs.is_empty() {
        eprintln!("nat-corpus: produced 0 docs from book {id} (empty after stripping?)");
        return 1;
    }

    let out = f.get("out").map(String::as_str).unwrap_or("-");
    let append = f.contains_key("append");
    let res = if out == "-" {
        jsonl::write_rawdocs(std::io::stdout().lock(), &docs)
    } else {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(append)
            .write(true)
            .truncate(!append)
            .open(out);
        match file {
            Ok(file) => jsonl::write_rawdocs(std::io::BufWriter::new(file), &docs),
            Err(e) => {
                eprintln!("nat-corpus: opening {out}: {e}");
                return 1;
            }
        }
    };
    match res {
        Ok(()) => {
            eprintln!(
                "from-gutenberg {id}: {} passages -> {} ({}append)",
                docs.len(),
                out,
                if append { "" } else { "over" }
            );
            0
        }
        Err(e) => {
            eprintln!("nat-corpus: writing {out}: {e}");
            1
        }
    }
}

fn cmd_from_code(args: &[String]) -> i32 {
    let f = flags(args);
    let dir = require(&f, "dir").to_string();
    let license = require(&f, "license").to_string();
    if !nat_data::ALLOWED_LICENSES.contains(&license.as_str()) {
        eprintln!(
            "nat-corpus: license '{license}' is not permissive/allow-listed.\n  allowed: {}",
            nat_data::ALLOWED_LICENSES.join(", ")
        );
        return 2;
    }
    let target_chars: usize = f
        .get("target-chars")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1500);
    let max_line_len: usize = f
        .get("max-line-len")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1000);
    let fetch_date = f.get("fetch-date").cloned().unwrap_or_else(today);
    let root = Path::new(&dir);
    let source = f.get("source").cloned().unwrap_or_else(|| {
        root.file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "repo".to_string())
    });

    let files = match code::walk_code_files(root) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("nat-corpus: walking {dir}: {e}");
            return 1;
        }
    };

    let mut docs = Vec::new();
    let (mut processed, mut skipped) = (0usize, 0usize);
    for path in &files {
        // Skip non-UTF8 / unreadable, and minified (very long lines).
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };
        if content.lines().any(|l| l.len() > max_line_len) {
            skipped += 1;
            continue;
        }
        let rel = path
            .strip_prefix(root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        docs.extend(code::file_to_rawdocs(
            &source,
            &license,
            &fetch_date,
            &rel,
            &content,
            target_chars,
        ));
        processed += 1;
    }

    if docs.is_empty() {
        eprintln!(
            "nat-corpus: no code passages from {dir} ({} files seen)",
            files.len()
        );
        return 1;
    }

    let out = f.get("out").map(String::as_str).unwrap_or("-");
    let append = f.contains_key("append");
    let res = if out == "-" {
        jsonl::write_rawdocs(std::io::stdout().lock(), &docs)
    } else {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(append)
            .write(true)
            .truncate(!append)
            .open(out);
        match file {
            Ok(file) => jsonl::write_rawdocs(std::io::BufWriter::new(file), &docs),
            Err(e) => {
                eprintln!("nat-corpus: opening {out}: {e}");
                return 1;
            }
        }
    };
    match res {
        Ok(()) => {
            eprintln!(
                "from-code {source} [{license}]: {processed} files ({skipped} skipped) -> {} passages -> {out}",
                docs.len()
            );
            0
        }
        Err(e) => {
            eprintln!("nat-corpus: writing {out}: {e}");
            1
        }
    }
}

fn cmd_train_bpe(args: &[String]) -> i32 {
    let f = flags(args);
    let input = require(&f, "input").to_string();
    let out = require(&f, "out").to_string();
    let vocab: usize = match f.get("vocab").and_then(|v| v.parse().ok()) {
        Some(v) if v >= 256 => v,
        _ => {
            eprintln!("nat-corpus: --vocab must be an integer >= 256");
            return 2;
        }
    };

    let docs = match jsonl::read_rawdocs_file(Path::new(&input)) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("nat-corpus: reading {input}: {e}");
            return 1;
        }
    };
    let texts: Vec<&str> = docs.iter().map(|d| d.text.as_str()).collect();
    eprintln!(
        "training BPE (target vocab {vocab}) on {} docs ...",
        texts.len()
    );
    let bpe = bpe::Bpe::train(texts.iter().copied(), vocab);

    // Compression report over the corpus: bytes/token.
    let (mut bytes, mut tokens) = (0u64, 0u64);
    for d in &docs {
        bytes += d.text.len() as u64;
        tokens += bpe.encode(&d.text).len() as u64;
    }
    let ratio = if tokens == 0 {
        0.0
    } else {
        bytes as f64 / tokens as f64
    };

    if let Err(e) = bpe.save(Path::new(&out)) {
        eprintln!("nat-corpus: writing {out}: {e}");
        return 1;
    }
    println!(
        "bpe: vocab {} -> {out} ; corpus {bytes} bytes / {tokens} tokens = {ratio:.2} bytes/token",
        bpe.vocab_size()
    );
    0
}

fn cmd_from_text(args: &[String]) -> i32 {
    let f = flags(args);
    let license = require(&f, "license").to_string();
    if !nat_data::ALLOWED_LICENSES.contains(&license.as_str()) {
        eprintln!(
            "nat-corpus: license '{license}' not allow-listed.\n  allowed: {}",
            nat_data::ALLOWED_LICENSES.join(", ")
        );
        return 2;
    }
    let source = require(&f, "source").to_string();
    let id_prefix = f
        .get("id-prefix")
        .cloned()
        .unwrap_or_else(|| source.clone());
    let target_chars: usize = f
        .get("target-chars")
        .and_then(|v| v.parse().ok())
        .unwrap_or(2000);
    let fetch_date = f.get("fetch-date").cloned().unwrap_or_else(today);

    // Read the document from --input (a file) or stdin.
    let mut raw = match f.get("input").map(String::as_str) {
        Some(p) if p != "-" => match std::fs::read_to_string(p) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("nat-corpus: reading {p}: {e}");
                return 1;
            }
        },
        _ => {
            let mut s = String::new();
            if let Err(e) = std::io::stdin().read_to_string(&mut s) {
                eprintln!("nat-corpus: reading stdin: {e}");
                return 1;
            }
            s
        }
    };
    if f.get("strip").map(String::as_str) == Some("latex") {
        raw = latex::strip(&raw);
    }

    let docs = text::to_rawdocs(
        &id_prefix,
        &source,
        &license,
        &fetch_date,
        &raw,
        target_chars,
    );
    if docs.is_empty() {
        eprintln!("nat-corpus: produced 0 passages");
        return 1;
    }

    let out = f.get("out").map(String::as_str).unwrap_or("-");
    let append = f.contains_key("append");
    let res = if out == "-" {
        jsonl::write_rawdocs(std::io::stdout().lock(), &docs)
    } else {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(append)
            .write(true)
            .truncate(!append)
            .open(out);
        match file {
            Ok(file) => jsonl::write_rawdocs(std::io::BufWriter::new(file), &docs),
            Err(e) => {
                eprintln!("nat-corpus: opening {out}: {e}");
                return 1;
            }
        }
    };
    match res {
        Ok(()) => {
            eprintln!(
                "from-text {source} [{license}]: {} passages -> {out}",
                docs.len()
            );
            0
        }
        Err(e) => {
            eprintln!("nat-corpus: writing {out}: {e}");
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
