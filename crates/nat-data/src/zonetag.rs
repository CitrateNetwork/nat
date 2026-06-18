//! ZONE_TAG stage (Data Ops §3, §4 step 5).
//!
//! Each document gets one or more zone-affinity tags so the router has a
//! differentiated training signal (the model is unified; the *data* is tagged).
//! Rule-based at L0 (source- and surface-derived); a light classifier refines it
//! at L1. A document can serve multiple zones.
//!
//! Note: `MX` is never a tag — it is the non-learned harness; its "data" is tool
//! schemas validated elsewhere, not training text.

use nat_types::ZoneId;

/// Assign zone-affinity tags. Always returns at least `PF` (general reasoning /
/// language is the floor every document contributes to).
pub fn tags(text: &str) -> Vec<ZoneId> {
    let lower = text.to_lowercase();
    let total = text.chars().count().max(1) as f32;

    let digits = text.chars().filter(|c| c.is_ascii_digit()).count() as f32;
    let math_ops = text
        .chars()
        .filter(|c| matches!(c, '+' | '-' | '*' | '/' | '=' | '^' | '%'))
        .count() as f32;
    let code_syms = text
        .chars()
        .filter(|c| matches!(c, '{' | '}' | '(' | ')' | ';' | '<' | '>' | '[' | ']'))
        .count() as f32;
    let alpha_ratio = text.chars().filter(|c| c.is_alphabetic()).count() as f32 / total;

    let has_code_kw = [
        "fn ",
        "def ",
        "class ",
        "import ",
        "function ",
        "return ",
        "let ",
        "const ",
    ]
    .iter()
    .any(|kw| lower.contains(kw));
    let sensory = [
        "see", "hear", "feel", "sound", "image", "touch", "smell", "loud", "bright", "rain", "warm",
    ]
    .iter()
    .any(|w| lower.contains(w));

    let mut out: Vec<ZoneId> = Vec::new();

    // Codec: code symbols or code keywords.
    if code_syms / total > 0.02 || has_code_kw {
        out.push(ZoneId::CX);
    }
    // Cerebellar: numeric/sequential/timing content (math leans here + PF).
    if digits / total > 0.03 || math_ops / total > 0.01 {
        out.push(ZoneId::CB);
    }
    // Hippocampal: narrative / memoir / dialogue (high alphabetic prose).
    if alpha_ratio > 0.6 {
        out.push(ZoneId::HP);
    }
    // Sensorimotor: sensory language (a thin multimodal slice at v1).
    if sensory {
        out.push(ZoneId::SM);
    }

    // Prefrontal is the floor: reasoning/language applies to every document.
    out.push(ZoneId::PF);

    // De-dup, emit in canonical ZoneId order.
    out.sort();
    out.dedup();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_is_tagged_cx() {
        let t = tags("fn main() { let x = vec![1, 2, 3]; }");
        assert!(t.contains(&ZoneId::CX));
    }

    #[test]
    fn narrative_is_tagged_hp() {
        let t = tags("she walked along the quiet shore at dawn thinking of home");
        assert!(t.contains(&ZoneId::HP));
    }

    #[test]
    fn sensory_is_tagged_sm() {
        let t = tags("the bright loud image filled the room with warm sound");
        assert!(t.contains(&ZoneId::SM));
    }

    #[test]
    fn pf_is_always_present_and_mx_never() {
        let t = tags("anything at all");
        assert!(t.contains(&ZoneId::PF));
        assert!(!t.contains(&ZoneId::MX));
    }
}
