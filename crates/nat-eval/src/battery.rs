//! Labeled prompt batteries for the routing-differentiation metric (H-02).
//!
//! Each class is a set of prompts that *should* drive a characteristic zone mix.
//! The metric (in `routing`) measures whether the model's per-prompt zone
//! activation separates by class. At L0 the router is hand-wired, so this is a
//! preview of H-02; the real test runs the same battery against a *trained*
//! router at L1 (the harness does not change, only the model under it does —
//! Data Ops §7, "built at L1, used everywhere after").

/// A class of prompts that share an expected routing signature.
#[derive(Debug, Clone)]
pub struct PromptClass {
    pub label: String,
    pub prompts: Vec<String>,
}

/// A labeled battery: several classes, each with several prompts.
#[derive(Debug, Clone)]
pub struct PromptBattery {
    pub classes: Vec<PromptClass>,
}

impl PromptBattery {
    /// The default L0 battery: four classes with distinct expected zone mixes.
    pub fn default_l0() -> PromptBattery {
        let class = |label: &str, prompts: &[&str]| PromptClass {
            label: label.into(),
            prompts: prompts.iter().map(|s| s.to_string()).collect(),
        };
        PromptBattery {
            classes: vec![
                class(
                    "math",
                    &[
                        "compute 12 * 7 + 3 and show each step",
                        "what is 144 divided by 12 minus 5",
                        "evaluate 2^10 + 3^4 - 17",
                        "solve for x: 4 * x + 9 = 33",
                    ],
                ),
                class(
                    "narrative",
                    &[
                        "she walked along the quiet shore at dawn thinking of home",
                        "tell me a story about an old lighthouse keeper",
                        "describe a long afternoon in a small mountain village",
                        "write a memoir paragraph about your grandmother's kitchen",
                    ],
                ),
                class(
                    "code",
                    &[
                        "fn main() { let x = vec![1, 2, 3]; println!(\"{}\", x.len()); }",
                        "write a function that returns the nth fibonacci number",
                        "class Stack { push(v) {} pop() {} }",
                        "import os; for f in os.listdir('.'): print(f)",
                    ],
                ),
                class(
                    "sensory",
                    &[
                        "the bright loud image filled the room with warm sound",
                        "describe how the cold rain feels on warm stone",
                        "a sharp smell of smoke and the sound of distant bells",
                        "the touch of rough bark and the bright glare of noon",
                    ],
                ),
            ],
        }
    }
}
