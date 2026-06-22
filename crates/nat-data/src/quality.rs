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

/// WP-D5 (part 2) — a model-based quality scorer: a byte **bigram** language model
/// (add-1 smoothed) over a clean reference corpus. A document's score falls as its
/// bits-per-byte under the model rises — well-formed text is predictable (low
/// bits/byte → high score); mojibake / symbol soup / gibberish is not (high
/// bits/byte → low score). Complements the heuristic [`score`]; both are available,
/// and the pipeline can gate on either via [`crate::run_pipeline_with_scorer`].
pub struct NgramModel {
    /// `counts[a*256 + b]` = how often byte `b` followed byte `a` in the reference.
    counts: Vec<u64>,
    /// `ctx[a]` = how often byte `a` appeared as a context.
    ctx: Vec<u64>,
}

impl NgramModel {
    /// Train the bigram model on a clean reference corpus (e.g. the CC0 seed).
    pub fn train<'a, I: IntoIterator<Item = &'a str>>(texts: I) -> Self {
        let mut counts = vec![0u64; 256 * 256];
        let mut ctx = vec![0u64; 256];
        for text in texts {
            for w in text.as_bytes().windows(2) {
                let (a, b) = (w[0] as usize, w[1] as usize);
                counts[a * 256 + b] += 1;
                ctx[a] += 1;
            }
        }
        NgramModel { counts, ctx }
    }

    /// Average bits per byte of `text` (add-1 smoothed, so an unseen pair costs
    /// ~8 bits, not infinity).
    pub fn bits_per_byte(&self, text: &str) -> f32 {
        let bytes = text.as_bytes();
        if bytes.len() < 2 {
            return 8.0;
        }
        let mut total = 0.0f64;
        let mut n = 0u64;
        for w in bytes.windows(2) {
            let (a, b) = (w[0] as usize, w[1] as usize);
            let p = (self.counts[a * 256 + b] as f64 + 1.0) / (self.ctx[a] as f64 + 256.0);
            total += -p.log2();
            n += 1;
        }
        (total / n as f64) as f32
    }

    /// Quality score in [0,1] = `1 - bits_per_byte/8` (8 = uniform random → 0; clean
    /// English ≈ 2–3 bits/byte → ≈ 0.6–0.75).
    pub fn score(&self, text: &str) -> Q16 {
        Q16::from_f32((1.0 - self.bits_per_byte(text) / 8.0).clamp(0.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_score_ranks_clean_above_gibberish() {
        let model = NgramModel::train(crate::seed::seed_corpus().iter().map(|d| d.text.as_str()));
        let clean = model
            .score("a clear and reasonably diverse english sentence about rivers")
            .to_f32();
        let gibberish = model
            .score("zxqj wkvb ppphhh 9183 zzz qqqq vbvbvb kkkk")
            .to_f32();
        assert!(
            clean > gibberish + 0.1,
            "clean {clean} not above gibberish {gibberish}"
        );
    }

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
