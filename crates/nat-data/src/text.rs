//! WP-D9 — a generic licensed-text connector. Splits a permissively-licensed text
//! (or LaTeX-stripped, or markdown) document into `RawDoc` passages for the
//! pipeline. The caller supplies the license (vetted) and source; the pipeline's
//! allow-list is the fail-closed gate. Used for CC text like SICP (CC-BY-SA) and
//! the LaTeX-only Gutenberg primaries (via [`crate::latex::strip`]).

use crate::RawDoc;

/// Split text into passages of roughly `target_chars`, on blank-line (paragraph)
/// boundaries. CRLF-safe and deterministic.
pub fn passages(text: &str, target_chars: usize) -> Vec<String> {
    let body = text.replace("\r\n", "\n").replace('\r', "\n");
    let target = target_chars.max(1);
    let mut out = Vec::new();
    let mut buf = String::new();
    for para in body.split("\n\n").map(str::trim).filter(|p| !p.is_empty()) {
        if !buf.is_empty() {
            buf.push(' ');
        }
        buf.push_str(para);
        if buf.len() >= target {
            out.push(std::mem::take(&mut buf));
        }
    }
    let tail = buf.trim();
    if !tail.is_empty() {
        out.push(tail.to_string());
    }
    out
}

/// Convert a text document to `RawDoc` passages with the given provenance.
pub fn to_rawdocs(
    id_prefix: &str,
    source: &str,
    license: &str,
    fetch_date: &str,
    text: &str,
    target_chars: usize,
) -> Vec<RawDoc> {
    passages(text, target_chars)
        .into_iter()
        .enumerate()
        .map(|(i, p)| RawDoc {
            id: format!("{id_prefix}-p{i:04}"),
            source: source.to_string(),
            license: license.to_string(),
            fetch_date: fetch_date.to_string(),
            text: p,
            modality_refs: vec![],
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_into_passages_with_provenance() {
        let text: String = (0..30)
            .map(|i| format!("Paragraph {i} with enough words to count.\n\n"))
            .collect();
        let docs = to_rawdocs("sicp-1", "sicp", "CC-BY-SA-4.0", "2026-06-22", &text, 100);
        assert!(docs.len() > 1);
        for d in &docs {
            assert_eq!(d.license, "CC-BY-SA-4.0");
            assert!(d.id.starts_with("sicp-1-p"));
        }
    }
}
