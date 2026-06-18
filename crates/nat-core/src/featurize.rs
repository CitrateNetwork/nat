//! Deterministic featurization of a prompt into a hidden vector + class signals.
//!
//! At L0 the router is not trained (Master Plan: L0 wires the pass; routing
//! *differentiation* is proven at L1/Gate 3). So featurization is a fixed,
//! deterministic function: a hidden embedding the zone cores read their slices
//! from, plus four interpretable class signals the router uses to differentiate.

/// Total hidden width `D`: six zones × the default slice width (16).
pub const D_HIDDEN: usize = 96;

/// Interpretable prompt-class signals in [0,1], used by the router so different
/// prompt classes drive different zone mixes (the H-02 hypothesis, tested for
/// real at L1; here it is hand-wired so the pass is exercised end to end).
#[derive(Debug, Clone, Copy)]
pub struct ClassSignals {
    pub math: f32,
    pub narrative: f32,
    pub code: f32,
    pub sensory: f32,
}

/// Compute class signals from cheap surface features of the prompt text.
pub fn class_signals(prompt: &str) -> ClassSignals {
    let total = prompt.chars().count().max(1) as f32;
    let digits = prompt.chars().filter(|c| c.is_ascii_digit()).count() as f32;
    let math_ops = prompt
        .chars()
        .filter(|c| matches!(c, '+' | '-' | '*' | '/' | '=' | '^' | '%'))
        .count() as f32;
    let code_syms = prompt
        .chars()
        .filter(|c| matches!(c, '{' | '}' | '(' | ')' | ';' | '<' | '>' | '[' | ']'))
        .count() as f32;
    let letters = prompt.chars().filter(|c| c.is_alphabetic()).count() as f32;
    let sensory_words = [
        "see", "hear", "feel", "sound", "image", "touch", "smell", "loud", "bright",
    ]
    .iter()
    .filter(|w| prompt.to_lowercase().contains(*w))
    .count() as f32;

    let clamp01 = |x: f32| x.clamp(0.0, 1.0);
    ClassSignals {
        math: clamp01((digits + 2.0 * math_ops) / total * 4.0),
        narrative: clamp01(letters / total),
        code: clamp01(code_syms / total * 6.0),
        sensory: clamp01(sensory_words / 3.0),
    }
}

/// A deterministic hidden embedding seeded by the prompt bytes. Not learned at
/// L0; it exists so each zone's slice carries a stable, prompt-dependent signal.
pub fn embed(prompt: &str) -> [f32; D_HIDDEN] {
    // FNV-1a 64 seed, then an LCG to fill the vector. All integer math →
    // identical bits on every platform and every federated node.
    let mut seed: u64 = 0xcbf29ce484222325;
    for b in prompt.bytes() {
        seed ^= b as u64;
        seed = seed.wrapping_mul(0x100000001b3);
    }
    let mut state = seed | 1;
    let mut out = [0.0f32; D_HIDDEN];
    for slot in out.iter_mut() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (state >> 33) as u32; // top 31 bits
        *slot = (u as f32 / u32::MAX as f32) * 2.0 - 1.0; // [-1, 1]
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_is_deterministic() {
        assert_eq!(embed("hello world"), embed("hello world"));
        assert_ne!(embed("a"), embed("b"));
    }

    #[test]
    fn math_prompt_reads_as_math() {
        let s = class_signals("compute 2 + 2 * 3 = ?");
        assert!(s.math > 0.2, "math signal was {}", s.math);
    }

    #[test]
    fn code_prompt_reads_as_code() {
        let s = class_signals("fn main() { let x = vec![1,2,3]; }");
        assert!(s.code > 0.1, "code signal was {}", s.code);
    }

    #[test]
    fn narrative_prompt_reads_as_narrative() {
        let s = class_signals("she walked along the quiet shore at dawn");
        assert!(s.narrative > 0.5);
    }
}
