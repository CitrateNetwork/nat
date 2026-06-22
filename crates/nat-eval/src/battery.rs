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
    /// An extended battery (10 prompts per class) for **held-out** H-02: train the
    /// router on a subset and score routing-differentiation on prompts it never saw.
    pub fn default_l0_extended() -> PromptBattery {
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
                        "find the greatest common divisor of 48 and 60",
                        "sum the integers from 1 to 100 using a closed form",
                        "differentiate the polynomial 3x^2 + 2x - 7",
                        "factor the quadratic x^2 - 5x + 6",
                        "compute the area of a circle with radius 4",
                        "convert 0.625 to a fraction in lowest terms",
                    ],
                ),
                class(
                    "narrative",
                    &[
                        "she walked along the quiet shore at dawn thinking of home",
                        "tell me a story about an old lighthouse keeper",
                        "describe a long afternoon in a small mountain village",
                        "write a memoir paragraph about your grandmother's kitchen",
                        "the traveller remembered the city she had left behind",
                        "a letter home from a sailor who has been gone for years",
                        "two friends meet again on a bridge after a long parting",
                        "the old house held the quiet of everyone who had lived there",
                        "she opened the door and the past came in with the cold air",
                        "he told the children the story his father had told him",
                    ],
                ),
                class(
                    "code",
                    &[
                        "fn main() { let x = vec![1, 2, 3]; println!(\"{}\", x.len()); }",
                        "write a function that returns the nth fibonacci number",
                        "class Stack { push(v) {} pop() {} }",
                        "import os; for f in os.listdir('.'): print(f)",
                        "def quicksort(a): return a if len(a) < 2 else partition(a)",
                        "let total = items.iter().map(|i| i.price).sum::<u64>();",
                        "async function fetchUser(id) { return await db.get(id); }",
                        "struct Point { x: f64, y: f64 } impl Point { fn norm() {} }",
                        "SELECT id, name FROM users WHERE active = 1 ORDER BY name;",
                        "for i in range(len(arr)): arr[i] = arr[i] * 2",
                    ],
                ),
                class(
                    "sensory",
                    &[
                        "the bright loud image filled the room with warm sound",
                        "describe how the cold rain feels on warm stone",
                        "a sharp smell of smoke and the sound of distant bells",
                        "the touch of rough bark and the bright glare of noon",
                        "warm light, the smell of bread, the hum of a quiet street",
                        "cold wind on the face and the taste of salt in the air",
                        "the soft weight of a blanket and the dim glow of a lamp",
                        "a loud clap of thunder and the flash of white light",
                        "the rough grain of wood under a slow moving hand",
                        "the sweet smell of rain on dust and the green of wet leaves",
                    ],
                ),
            ],
        }
    }

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
