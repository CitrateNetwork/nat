//! Byte-level tokenizer (Data Ops §4 step 6, the L1 "real tokenizer").
//!
//! The L0 pipeline only *counted* whitespace tokens; training needs actual token
//! IDs. A byte-level tokenizer is the right first real tokenizer: it is
//! deterministic, lossless, language-agnostic, needs no trained merge table, and
//! has a fixed vocab of 256. Same text → same IDs on every machine, which the
//! reproducibility floor requires. A learned BPE vocab is a later upgrade
//! (DATA-S1 WP-D5) that slots in behind the same `encode`/`vocab` interface.

/// Vocabulary size for the byte-level tokenizer.
pub const BYTE_VOCAB: usize = 256;

/// Encode text to byte token IDs (UTF-8 bytes, one ID per byte). Lossless and
/// deterministic.
pub fn encode(text: &str) -> Vec<u32> {
    text.as_bytes().iter().map(|&b| b as u32).collect()
}

/// Decode byte token IDs back to a string (lossy only if the IDs were not a valid
/// UTF-8 byte stream — used for inspection, not the training path).
pub fn decode(ids: &[u32]) -> String {
    let bytes: Vec<u8> = ids.iter().map(|&i| (i & 0xFF) as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_is_deterministic_and_in_vocab() {
        let a = encode("Belnap's four values: true, false, both, neither.");
        let b = encode("Belnap's four values: true, false, both, neither.");
        assert_eq!(a, b);
        assert!(a.iter().all(|&id| (id as usize) < BYTE_VOCAB));
    }

    #[test]
    fn round_trips_utf8() {
        let s = "logic, language, and the four-valued lattice ⊤⊥";
        assert_eq!(decode(&encode(s)), s);
    }
}
