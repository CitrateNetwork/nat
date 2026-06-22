//! Project Gutenberg connector (HERMES-S1 WP-H5) — the deterministic conversion
//! half: a downloaded Gutenberg plain-text book → `RawDoc`s, ready for the
//! pipeline. The *fetch* is the agent's job (it holds the network grant); this is
//! pure, testable, dependency-free conversion.
//!
//! Gutenberg texts are public-domain (the allow-list passes `public-domain`), but
//! each file is wrapped in a license header/footer that must be stripped, and the
//! body is one long stream that we split into passages so dedup/quality act at a
//! sane granularity.

use crate::RawDoc;

/// Strip the Project Gutenberg header and footer, returning just the book body.
/// Falls back to the whole (trimmed) text if the standard markers are absent.
pub fn strip_boilerplate(raw: &str) -> String {
    // Header: everything up to and including the "*** START ... ***" line.
    let body_start = raw
        .find("START OF THE PROJECT GUTENBERG")
        .or_else(|| raw.find("START OF THIS PROJECT GUTENBERG"))
        .and_then(|pos| raw[pos..].find('\n').map(|nl| pos + nl + 1))
        .unwrap_or(0);
    // Footer: everything from the "*** END ... ***" line onward.
    let body_end = raw
        .find("END OF THE PROJECT GUTENBERG")
        .or_else(|| raw.find("END OF THIS PROJECT GUTENBERG"))
        .map(|pos| raw[..pos].rfind('\n').unwrap_or(pos))
        .unwrap_or(raw.len());

    if body_start < body_end {
        raw[body_start..body_end].trim().to_string()
    } else {
        raw.trim().to_string()
    }
}

/// Convert a downloaded Gutenberg book to `RawDoc` passages. The body is split on
/// blank-line (paragraph) boundaries and accumulated into passages of roughly
/// `target_chars`. Deterministic: same input → same docs. Provenance records the
/// book id; the license is `public-domain` (the pipeline's allow-list passes it).
pub fn to_rawdocs(book_id: u32, fetch_date: &str, raw: &str, target_chars: usize) -> Vec<RawDoc> {
    // Gutenberg files are CRLF; normalize so paragraph breaks ("\n\n") are found.
    let body = strip_boilerplate(raw)
        .replace("\r\n", "\n")
        .replace('\r', "\n");
    let target = target_chars.max(1);
    let mut docs = Vec::new();
    let mut buf = String::new();
    let mut idx = 0u32;

    let flush = |buf: &mut String, idx: &mut u32, docs: &mut Vec<RawDoc>| {
        let text = buf.trim();
        if !text.is_empty() {
            docs.push(RawDoc {
                id: format!("gutenberg-{book_id}-p{:04}", *idx),
                source: format!("gutenberg/{book_id}"),
                license: "public-domain".to_string(),
                fetch_date: fetch_date.to_string(),
                text: text.to_string(),
                modality_refs: vec![],
            });
            *idx += 1;
        }
        buf.clear();
    };

    for para in body.split("\n\n").map(str::trim).filter(|p| !p.is_empty()) {
        if !buf.is_empty() {
            buf.push(' ');
        }
        buf.push_str(para);
        if buf.len() >= target {
            flush(&mut buf, &mut idx, &mut docs);
        }
    }
    flush(&mut buf, &mut idx, &mut docs);
    docs
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "The Project Gutenberg eBook of Something\n\
        Some license preamble here.\n\
        *** START OF THE PROJECT GUTENBERG EBOOK SOMETHING ***\n\
        \n\
        First paragraph of the real body, long enough to mean something.\n\
        \n\
        Second paragraph continues the real text with more words here.\n\
        \n\
        *** END OF THE PROJECT GUTENBERG EBOOK SOMETHING ***\n\
        Some trailing license footer that must not be trained on.";

    #[test]
    fn boilerplate_is_stripped() {
        let body = strip_boilerplate(SAMPLE);
        assert!(body.starts_with("First paragraph"));
        assert!(!body.contains("license preamble"));
        assert!(!body.contains("trailing license footer"));
        assert!(!body.contains("PROJECT GUTENBERG"));
    }

    #[test]
    fn to_rawdocs_is_deterministic_and_well_formed() {
        let a = to_rawdocs(123, "2026-06-22", SAMPLE, 40);
        let b = to_rawdocs(123, "2026-06-22", SAMPLE, 40);
        assert_eq!(a.len(), b.len());
        assert!(!a.is_empty());
        for d in &a {
            assert!(d.id.starts_with("gutenberg-123-p"));
            assert_eq!(d.license, "public-domain");
            assert_eq!(d.source, "gutenberg/123");
            assert!(!d.text.contains("PROJECT GUTENBERG"));
        }
    }

    #[test]
    fn crlf_paragraphs_split_into_multiple_passages() {
        // Gutenberg uses CRLF; passages must still split (regression: a whole book
        // became one over-long passage and got quarantined).
        let body: String = (0..20)
            .map(|i| format!("Paragraph number {i} with enough words to be a real line.\r\n\r\n"))
            .collect();
        let raw = format!(
            "*** START OF THE PROJECT GUTENBERG EBOOK X ***\r\n\r\n{body}\r\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\r\n"
        );
        let docs = to_rawdocs(7, "2026-06-22", &raw, 120);
        assert!(
            docs.len() > 1,
            "CRLF body did not split: {} passages",
            docs.len()
        );
        assert!(docs.iter().all(|d| d.text.len() < 10_000));
    }

    #[test]
    fn missing_markers_fall_back_to_whole_text() {
        let plain = "just some text with no gutenberg markers at all in it here";
        let docs = to_rawdocs(1, "2026-06-22", plain, 1000);
        assert_eq!(docs.len(), 1);
        assert!(docs[0].text.contains("just some text"));
    }
}
