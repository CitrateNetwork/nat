//! WP-D5 — a byte-level BPE tokenizer.
//!
//! The byte tokenizer ([`crate::tokenizer`]) is lossless but uses one token per
//! byte. BPE learns merges of frequent adjacent pairs, so common substrings become
//! single tokens — far fewer tokens per character, hence more *effective context*
//! per position for the LM. It is byte-level (starts from the 256 byte tokens), so
//! it stays lossless and language-agnostic, and it is **deterministic**: the same
//! corpus + target vocab → the same merges (the reproducibility floor).
//!
//! Pre-tokenization splits text into maximal runs of whitespace vs non-whitespace,
//! so merges never cross that boundary — and a run of indentation can become one
//! token (valuable for code). A learned vocab slots in behind the same
//! `encode`/`decode`/`vocab_size` interface as the byte tokenizer
//! ([`Bpe::byte_level`] is the trivial 256-token case).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A trained byte-level BPE tokenizer. Only the ordered merge list is the
/// canonical state; the vocabulary and rank index are derived from it.
#[derive(Debug, Clone)]
pub struct Bpe {
    merges: Vec<(u32, u32)>,
    vocab: Vec<Vec<u8>>,
    ranks: HashMap<(u32, u32), u32>,
}

#[derive(Serialize, Deserialize)]
struct BpeFile {
    merges: Vec<(u32, u32)>,
}

impl Bpe {
    /// The trivial byte-level tokenizer (256 tokens, no merges).
    pub fn byte_level() -> Self {
        Self::from_merges(Vec::new())
    }

    /// Build from an ordered merge list (rebuilds the vocab + rank index).
    pub fn from_merges(merges: Vec<(u32, u32)>) -> Self {
        let mut vocab: Vec<Vec<u8>> = (0u16..256).map(|b| vec![b as u8]).collect();
        let mut ranks = HashMap::with_capacity(merges.len());
        for (rank, &(a, b)) in merges.iter().enumerate() {
            let mut bytes = vocab[a as usize].clone();
            bytes.extend_from_slice(&vocab[b as usize]);
            vocab.push(bytes);
            ranks.insert((a, b), rank as u32);
        }
        Bpe {
            merges,
            vocab,
            ranks,
        }
    }

    /// Train BPE over the texts until the vocabulary reaches `target_vocab`
    /// (≥ 256). Deterministic: ties on pair frequency break to the smaller pair.
    pub fn train<'a, I: IntoIterator<Item = &'a str>>(texts: I, target_vocab: usize) -> Self {
        // Word frequencies (a word = a same-class run), as sequences of byte ids.
        let mut word_freqs: HashMap<Vec<u32>, u64> = HashMap::new();
        for text in texts {
            for word in pretokenize(text) {
                let ids: Vec<u32> = word.iter().map(|&b| b as u32).collect();
                *word_freqs.entry(ids).or_insert(0) += 1;
            }
        }
        let mut words: Vec<(Vec<u32>, u64)> = word_freqs.into_iter().collect();
        words.sort(); // deterministic iteration order

        let mut merges = Vec::new();
        let target = target_vocab.max(256);
        let mut next_id = 256u32;
        while (next_id as usize) < target {
            // Count adjacent pairs (frequency-weighted).
            let mut pair_counts: HashMap<(u32, u32), u64> = HashMap::new();
            for (w, f) in &words {
                for pair in w.windows(2) {
                    *pair_counts.entry((pair[0], pair[1])).or_insert(0) += f;
                }
            }
            // Most frequent pair; tie-break to the smaller pair (deterministic).
            let best = pair_counts.iter().fold(None, |acc, (p, &c)| match acc {
                None => Some((*p, c)),
                Some((bp, bc)) if c > bc || (c == bc && *p < bp) => Some((*p, c)),
                other => other,
            });
            let Some(((a, b), count)) = best else { break };
            if count == 0 {
                break;
            }
            for (w, _) in words.iter_mut() {
                *w = merge_word(w, a, b, next_id);
            }
            merges.push((a, b));
            next_id += 1;
        }
        Self::from_merges(merges)
    }

    /// Encode text to token ids (greedy: repeatedly apply the lowest-rank merge).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        for word in pretokenize(text) {
            let mut ids: Vec<u32> = word.iter().map(|&b| b as u32).collect();
            loop {
                let mut best_rank = u32::MAX;
                let mut best_pair = None;
                for i in 0..ids.len().saturating_sub(1) {
                    if let Some(&r) = self.ranks.get(&(ids[i], ids[i + 1])) {
                        if r < best_rank {
                            best_rank = r;
                            best_pair = Some((ids[i], ids[i + 1]));
                        }
                    }
                }
                let Some((a, b)) = best_pair else { break };
                ids = merge_word(&ids, a, b, 256 + best_rank);
            }
            out.extend(ids);
        }
        out
    }

    /// Decode token ids back to text (lossless for ids this BPE produced).
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if let Some(v) = self.vocab.get(id as usize) {
                bytes.extend_from_slice(v);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Persist the BPE (just the merge list) as JSON.
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        let bytes = serde_json::to_vec_pretty(&BpeFile {
            merges: self.merges.clone(),
        })
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, bytes)
    }

    /// Load a BPE from a JSON file written by [`Bpe::save`].
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let file: BpeFile = serde_json::from_slice(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(Self::from_merges(file.merges))
    }
}

/// Split text into maximal runs of whitespace vs non-whitespace (each a "word").
fn pretokenize(text: &str) -> Vec<Vec<u8>> {
    let bytes = text.as_bytes();
    let mut words = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let ws = bytes[i].is_ascii_whitespace();
        let start = i;
        while i < bytes.len() && bytes[i].is_ascii_whitespace() == ws {
            i += 1;
        }
        words.push(bytes[start..i].to_vec());
    }
    words
}

/// Replace every adjacent `(a, b)` in `w` with `new_id`.
fn merge_word(w: &[u32], a: u32, b: u32, new_id: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(w.len());
    let mut i = 0;
    while i < w.len() {
        if i + 1 < w.len() && w[i] == a && w[i + 1] == b {
            out.push(new_id);
            i += 2;
        } else {
            out.push(w[i]);
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus() -> Vec<String> {
        crate::seed::seed_corpus()
            .into_iter()
            .map(|d| d.text)
            .collect()
    }

    #[test]
    fn byte_level_is_256_and_lossless() {
        let bpe = Bpe::byte_level();
        assert_eq!(bpe.vocab_size(), 256);
        let s = "logic and language ⊤⊥";
        assert_eq!(bpe.decode(&bpe.encode(s)), s);
    }

    #[test]
    fn trains_grows_vocab_and_round_trips() {
        let texts = corpus();
        let bpe = Bpe::train(texts.iter().map(String::as_str), 1024);
        assert!(bpe.vocab_size() > 256 && bpe.vocab_size() <= 1024);
        let s = "the rules of the room are a form of life";
        assert_eq!(bpe.decode(&bpe.encode(s)), s);
    }

    #[test]
    fn bpe_compresses_versus_bytes() {
        let texts = corpus();
        let bpe = Bpe::train(texts.iter().map(String::as_str), 1024);
        let sample = &texts[0];
        let toks = bpe.encode(sample).len();
        let bytes = sample.len();
        assert!(
            toks < bytes,
            "no compression: {toks} tokens vs {bytes} bytes"
        );
    }

    #[test]
    fn training_is_deterministic() {
        let texts = corpus();
        let a = Bpe::train(texts.iter().map(String::as_str), 600);
        let b = Bpe::train(texts.iter().map(String::as_str), 600);
        assert_eq!(a.merges, b.merges);
    }

    #[test]
    fn save_load_round_trips() {
        let texts = corpus();
        let bpe = Bpe::train(texts.iter().map(String::as_str), 600);
        let path = std::env::temp_dir().join("nat_bpe_test.json");
        bpe.save(&path).unwrap();
        let loaded = Bpe::load(&path).unwrap();
        let s = "Belnap's four values: true, false, both, neither.";
        assert_eq!(loaded.encode(s), bpe.encode(s));
        let _ = std::fs::remove_file(&path);
    }
}
