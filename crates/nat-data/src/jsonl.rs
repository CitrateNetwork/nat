//! The `RawDoc` JSONL ingest contract (HERMES-S1 WP-H2).
//!
//! One JSON object per line, each a [`RawDoc`]:
//! ```json
//! {"id":"gutenberg-1342-p1","source":"gutenberg","license":"public-domain","fetch_date":"2026-06-22","text":"It is a truth universally acknowledged ...","modality_refs":[]}
//! ```
//! `modality_refs` may be omitted (defaults to empty). Blank lines are skipped.
//! Lines that fail to parse are reported with their line number rather than
//! silently dropped — a collector must see what it got wrong.

use crate::RawDoc;
use std::io::{BufRead, BufReader, Error, ErrorKind, Result, Write};
use std::path::Path;

/// Parse `RawDoc` JSONL from a reader. Returns the docs, or an error naming the
/// first bad line (1-based).
pub fn read_rawdocs<R: BufRead>(reader: R) -> Result<Vec<RawDoc>> {
    let mut docs = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let doc: RawDoc = serde_json::from_str(trimmed)
            .map_err(|e| Error::new(ErrorKind::InvalidData, format!("line {}: {e}", i + 1)))?;
        docs.push(doc);
    }
    Ok(docs)
}

/// Read `RawDoc` JSONL from a file path.
pub fn read_rawdocs_file(path: &Path) -> Result<Vec<RawDoc>> {
    let f = std::fs::File::open(path)?;
    read_rawdocs(BufReader::new(f))
}

/// Write docs as `RawDoc` JSONL to a writer (one compact object per line).
pub fn write_rawdocs<W: Write>(mut w: W, docs: &[RawDoc]) -> Result<()> {
    for d in docs {
        let line = serde_json::to_string(d).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
        w.write_all(line.as_bytes())?;
        w.write_all(b"\n")?;
    }
    Ok(())
}

/// Write docs as `RawDoc` JSONL to a file path.
pub fn write_rawdocs_file(path: &Path, docs: &[RawDoc]) -> Result<()> {
    let f = std::fs::File::create(path)?;
    write_rawdocs(std::io::BufWriter::new(f), docs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jsonl_round_trips_the_seed_corpus() {
        let docs = crate::seed::seed_corpus();
        let mut buf: Vec<u8> = Vec::new();
        write_rawdocs(&mut buf, &docs).unwrap();
        let back = read_rawdocs(buf.as_slice()).unwrap();
        assert_eq!(back.len(), docs.len());
        assert_eq!(back[0].id, docs[0].id);
        assert_eq!(back[0].text, docs[0].text);
    }

    #[test]
    fn modality_refs_default_when_omitted_and_blanks_are_skipped() {
        let jsonl = "\n{\"id\":\"a\",\"source\":\"s\",\"license\":\"MIT\",\"fetch_date\":\"2026-06-22\",\"text\":\"hello world this is fine\"}\n\n";
        let docs = read_rawdocs(jsonl.as_bytes()).unwrap();
        assert_eq!(docs.len(), 1);
        assert!(docs[0].modality_refs.is_empty());
    }

    #[test]
    fn a_bad_line_is_reported_with_its_number() {
        let jsonl = "{\"id\":\"ok\",\"source\":\"s\",\"license\":\"MIT\",\"fetch_date\":\"d\",\"text\":\"long enough text here\"}\nnot json\n";
        let err = read_rawdocs(jsonl.as_bytes()).unwrap_err();
        assert!(err.to_string().contains("line 2"), "{err}");
    }
}
