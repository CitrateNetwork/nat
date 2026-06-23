//! WP-D9 — a heuristic LaTeX-to-text stripper, for public-domain primaries that
//! Gutenberg only offers as LaTeX source (e.g. Boole's *Laws of Thought*,
//! Wittgenstein's *Tractatus*).
//!
//! It is deliberately simple — strip `%` comments, drop command names and
//! environment tags while keeping their text content, drop braces, turn `\\` into a
//! line break — so most prose-bearing LaTeX becomes readable text. Math-heavy
//! regions come out noisy; the pipeline's quality gate (`quality::NgramModel`)
//! filters the worst of those. It is not a LaTeX parser and does not try to be.

/// Strip LaTeX markup to plain-ish text.
pub fn strip(tex: &str) -> String {
    // 1. Drop `%` line comments (but keep an escaped `\%`).
    let no_comments: String = tex
        .lines()
        .map(|l| {
            let b = l.as_bytes();
            let mut cut = l.len();
            let mut i = 0;
            while i < b.len() {
                if b[i] == b'%' && (i == 0 || b[i - 1] != b'\\') {
                    cut = i;
                    break;
                }
                i += 1;
            }
            &l[..cut]
        })
        .collect::<Vec<_>>()
        .join("\n");

    // 2. Scan: drop command names + \begin/\end groups, keep braced content, drop braces.
    let chars: Vec<char> = no_comments.chars().collect();
    let mut out = String::with_capacity(chars.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '\\' {
            i += 1;
            // Read the command name (letters).
            let start = i;
            while i < chars.len() && chars[i].is_ascii_alphabetic() {
                i += 1;
            }
            let cmd: String = chars[start..i].iter().collect();

            if cmd.is_empty() {
                // \\ (line break) or an escaped char like \% \& \_ \{ \}.
                if i < chars.len() {
                    let ch = chars[i];
                    out.push(if ch == '\\' { '\n' } else { ch });
                    i += 1;
                }
                continue;
            }
            // Optional [..] arguments — skip them.
            if i < chars.len() && chars[i] == '[' {
                while i < chars.len() && chars[i] != ']' {
                    i += 1;
                }
                if i < chars.len() {
                    i += 1;
                }
            }
            // \begin{env} / \end{env}: drop the {env} group too.
            if (cmd == "begin" || cmd == "end") && i < chars.len() && chars[i] == '{' {
                while i < chars.len() && chars[i] != '}' {
                    i += 1;
                }
                if i < chars.len() {
                    i += 1;
                }
            }
            out.push(' '); // command becomes a separator; its braced content survives
            continue;
        }
        if c == '{' || c == '}' {
            i += 1; // drop braces, keep content
            continue;
        }
        out.push(c);
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_commands_keeps_text() {
        let tex = "\\section{Of Signs}\nLogic is the science of \\emph{reasoning}. % a comment\nAll men are mortal.\\\\\nSocrates is a man.\n";
        let s = strip(tex);
        // The pipeline normalizes whitespace downstream; compare collapsed.
        let n: String = s.split_whitespace().collect::<Vec<_>>().join(" ");
        assert!(n.contains("Of Signs"));
        assert!(n.contains("Logic is the science of reasoning."));
        assert!(n.contains("All men are mortal."));
        assert!(!n.contains("a comment"));
        assert!(!s.contains('\\') && !s.contains('{') && !s.contains('}'));
    }

    #[test]
    fn drops_environment_tags_keeps_body() {
        let tex = "\\begin{theorem}\nThe whole is greater than the part.\n\\end{theorem}";
        let s = strip(tex);
        assert!(s.contains("The whole is greater than the part."));
        assert!(!s.contains("theorem"));
    }
}
