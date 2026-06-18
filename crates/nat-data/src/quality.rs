//! QUALITY_SCORE stage (Data Ops §4 step 4).
//!
//! The score this produces is the economic signal: it aggregates into the
//! manifest's `aggregate_quality`, which is the `data_quality` term in
//! `reward_weight = compute × quality`. So the scorer is deliberately
//! conservative and explainable — at L0 it is rule-based heuristics; L1 adds
//! model-based filters (perplexity gate, learned quality classifier).
//!
//! PII screening is a **gate**, not a score adjustment (Data Ops §4.1): a doc
//! that trips it is quarantined, never trained on.

use nat_types::Q16;

/// Heuristic quality score in [0,1]. The mean of three explainable sub-scores:
/// printable-ASCII ratio, alphabetic ratio, and token diversity. Each is a proxy
/// for a way text goes bad (mojibake, symbol soup, degenerate repetition).
pub fn score(text: &str) -> Q16 {
    let chars: Vec<char> = text.chars().collect();
    let total = chars.len().max(1) as f32;

    let printable = chars
        .iter()
        .filter(|c| c.is_ascii_graphic() || **c == ' ')
        .count() as f32;
    let s_ascii = printable / total;

    let alpha = chars.iter().filter(|c| c.is_alphabetic()).count() as f32;
    // Prose wants a healthy alphabetic ratio; reward up to ~0.7, then plateau so
    // code (lower alpha ratio) is not unfairly crushed.
    let s_alpha = (alpha / total / 0.7).clamp(0.0, 1.0);

    let tokens: Vec<&str> = text.split_whitespace().collect();
    let s_diversity = if tokens.is_empty() {
        0.0
    } else {
        let unique = tokens
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len() as f32;
        unique / tokens.len() as f32
    };

    let mean = (s_ascii + s_alpha + s_diversity) / 3.0;
    Q16::from_f32(mean.clamp(0.0, 1.0))
}

/// PII screen. Returns the kind of hit if the text looks like it contains
/// personal data (email-like token, or a long digit run resembling an SSN/card).
/// Intentionally high-recall for L0; a reviewer audits the quarantine.
pub fn pii_hit(text: &str) -> Option<String> {
    for tok in text.split_whitespace() {
        // Email-like: has '@' with a dot somewhere after it.
        if let Some(at) = tok.find('@') {
            if tok[at + 1..].contains('.') {
                return Some("email-like".into());
            }
        }
    }
    // A run of 9+ consecutive ASCII digits (SSN/card-like).
    let mut run = 0usize;
    for c in text.chars() {
        if c.is_ascii_digit() {
            run += 1;
            if run >= 9 {
                return Some("digit-sequence".into());
            }
        } else {
            run = 0;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_prose_scores_higher_than_symbol_soup() {
        let good = score("a clear sentence about rivers and the slow work of cartography");
        let bad = score("@@@@ #### $$$$ %%%% @@@@ #### $$$$ %%%% @@@@ ####");
        assert!(good > bad, "good={} bad={}", good.to_f32(), bad.to_f32());
    }

    #[test]
    fn repetition_lowers_diversity_and_score() {
        let diverse = score("the quick brown fox jumps over a lazy sleeping dog nearby");
        let repeat = score("spam spam spam spam spam spam spam spam spam spam spam spam");
        assert!(diverse > repeat);
    }

    #[test]
    fn email_is_flagged() {
        assert_eq!(
            pii_hit("contact me at jane@example.com tomorrow"),
            Some("email-like".into())
        );
    }

    #[test]
    fn long_digit_run_is_flagged() {
        assert_eq!(
            pii_hit("ssn 123456789 on file"),
            Some("digit-sequence".into())
        );
        assert_eq!(pii_hit("the year was 2026 and all was well"), None);
    }
}
