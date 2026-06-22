//! A small CC0 seed corpus — original text authored for NAT's first real-data
//! run (DATA-S1). It spans the domains the model should learn first and the four
//! eval-battery classes, so the pipeline → train → eval path can be exercised on
//! *real* (if tiny) text before a full corpus is collected. Hermes (HERMES-S1)
//! scales this with permissively-licensed external sources; this seed only proves
//! the path and seeds the data-rich zones {HP, PF, CX} (ADR-0008).
//!
//! Every document is CC0-1.0 (authored here, dedicated to the public domain), so
//! it passes the license gate, carries clean provenance, and is safe to train on.

use crate::RawDoc;

fn doc(id: &str, text: &str) -> RawDoc {
    RawDoc {
        id: id.to_string(),
        source: "nat-seed".to_string(),
        license: "CC0-1.0".to_string(),
        fetch_date: "2026-06-22".to_string(),
        text: text.to_string(),
        modality_refs: vec![],
    }
}

/// The seed corpus: a handful of clean, diverse, public-domain documents.
pub fn seed_corpus() -> Vec<RawDoc> {
    vec![
        // --- Logic & language philosophy, and Belnap's four-valued logic ---
        doc(
            "belnap-four",
            "Belnap's four-valued logic answers a practical question: how should a \
             reasoner compute when its information may be incomplete or even \
             contradictory? Instead of two values it uses four, drawn from the told \
             status of a sentence. A sentence may be told only true, told only false, \
             told both true and false, or told neither. These four values form a \
             lattice. The value both marks a contradiction the system must carry \
             without exploding; the value neither marks the absence of information. \
             A database fed by many sources naturally lands in this lattice, because \
             sources disagree and sources fall silent.",
        ),
        doc(
            "fde-entailment",
            "First-degree entailment treats inference so that a single contradiction \
             does not entail every sentence. Classical logic is explosive: from a \
             contradiction it proves anything, which is useless for a machine reading \
             noisy evidence. Paraconsistent logic refuses that step. It keeps the \
             reasoning local, so a contradiction about one matter does not poison \
             conclusions about another. This is exactly the discipline a provenance \
             system needs when its zones disagree about what was seen.",
        ),
        doc(
            "sense-reference",
            "Frege distinguished the sense of an expression from its reference. Two \
             names can refer to the same object yet differ in sense, in the way they \
             present that object to a mind. Meaning is therefore not exhausted by what \
             a word points at; it includes the mode of presentation. A theory of \
             language that ignores sense cannot explain how someone learns that the \
             morning star and the evening star are one and the same.",
        ),
        doc(
            "truth-bearers",
            "What is it that is true or false? Sentences, propositions, beliefs, and \
             assertions have all been proposed as the bearers of truth. The choice \
             matters for a verifier. If truth attaches to propositions, then the same \
             claim recorded in two formats has one truth value; if it attaches to \
             sentences, the format can change the verdict. A careful system fixes its \
             truth-bearer first, then records evidence against it.",
        ),
        // --- Narrative (HP / hippocampal) ---
        doc(
            "narrative-shore",
            "She walked the quiet shore at dawn, thinking of home and the long road \
             that had carried her away from it. The tide had turned in the night and \
             left a wide grey mirror of wet sand, and her footprints filled with light \
             behind her. She remembered the kitchen, the smell of bread, the low talk \
             of people who loved her, and she let the memory walk beside her for a \
             while before she let it go.",
        ),
        doc(
            "narrative-lighthouse",
            "The old keeper climbed the iron stair each evening the way other people \
             say their prayers. He trimmed the wick, he wound the clockwork, he set the \
             great lens turning, and then he stood at the gallery rail and watched the \
             dark come in from the sea. Ships he would never meet steered by his light. \
             That was enough for him, to be useful to strangers in the night.",
        ),
        // --- Sensory (SM / sensorimotor) ---
        doc(
            "sensory-rain",
            "The cold rain felt sharp on warm stone, and the air filled with the bright \
             smell of wet dust and distant smoke. Bells rang somewhere across the \
             valley, faint and clear at once. Rough bark pressed against an open palm, \
             and the glare of noon broke through the cloud in a single warm bar of \
             light that moved slowly across the floor.",
        ),
        // --- Math & structured reasoning (PF / prefrontal) ---
        doc(
            "math-steps",
            "To evaluate the expression, work from the inside out and keep each step in \
             view. First compute the product of seven and three, which is twenty one. \
             Then add the remaining term, four, to reach twenty five. The order of \
             operations is not a convention to memorize but a way to keep a shared \
             meaning: multiplication binds tighter than addition, so the product is \
             formed before the sum. Show the work and the answer carries its own proof.",
        ),
        doc(
            "math-induction",
            "Mathematical induction proves a statement for every natural number in two \
             moves. First establish the base case, that the statement holds for the \
             smallest number. Then prove the inductive step, that whenever the statement \
             holds for some number it also holds for the next. Together these two facts \
             knock down the whole infinite line of dominoes, because any number is \
             reached by starting at the base and stepping forward a finite number of \
             times.",
        ),
        // --- Code (CX / codec) ---
        doc(
            "code-fibonacci",
            "Here is a function that returns the nth Fibonacci number using a loop \
             rather than recursion, so it runs in linear time and constant space. \
             It keeps two running values, the previous and the current, and updates \
             them together on each step. fn fib(n: u64) -> u64 { let mut a = 0u64; let \
             mut b = 1u64; for _ in 0..n { let next = a + b; a = b; b = next; } a }",
        ),
        doc(
            "code-stack",
            "A stack is a last in, first out collection: the item pushed most recently \
             is the first one popped. The interface is small and total. push places a \
             value on top, pop removes and returns the top value if one exists, and \
             peek looks at the top without removing it. A vector backs the stack \
             cleanly, since appending and removing from its end are both fast.",
        ),
        // --- Rules, language, computation: the values spine (owner intent 2026-06-22) ---
        // CC0 explainers of copyrighted ideas (we own the framing); the public-domain
        // primary sources (Wittgenstein's Tractatus, Boole, Turing's papers as of
        // life+70) are fetch targets in the research loop.
        doc(
            "wittgenstein-private-language",
            "A private language would be one whose words referred to what only the \
             speaker could know, his immediate private sensations, so that no one else \
             could understand it. Wittgenstein argues that such a language is \
             impossible. To mean something by a sign is to use it according to a rule, \
             and a rule needs a criterion of correctness, a way to tell genuine use \
             from merely seeming right. In a purely private case there is no such \
             criterion: whatever is going to seem right to me is right, which only \
             means that here we cannot speak of right at all. Meaning is not a private \
             inner act but a public practice, answerable to a shared standard.",
        ),
        doc(
            "wittgenstein-rule-following",
            "A rule does not contain its own application. Any course of action can be \
             brought into accord with a rule under some interpretation, so \
             interpretation alone cannot fix what the rule requires. What fixes it is a \
             practice. To obey a rule, to make a report, to give an order, to play a \
             game of chess, are customs, uses, institutions. You cannot follow a rule \
             privately or only once, for then there would be no difference between \
             following it and thinking you are. Following a rule is mastering a \
             technique, sustained by a community that agrees in its judgments. The \
             agreement is not agreement in opinions but in form of life.",
        ),
        doc(
            "form-of-life-and-the-room",
            "Meaning lives in a form of life, the shared ways of acting, judging, and \
             correcting that give signs their grip. The same holds for any room a maker \
             enters. A codebase, a protocol, a community each has its conventions, and \
             to act well is to follow the rules of that room rather than a private rule \
             of one's own invention. A rule with no community and no spirit behind it \
             is not yet a rule; it is a mark waiting for a practice. This is why a \
             system that records its reasoning against a public standard can be trusted, \
             while a private and unauditable one cannot.",
        ),
        doc(
            "turing-machine",
            "A Turing machine is a simple imagined device: a tape of cells, a head that \
             reads and writes one symbol at a time, and a table of rules saying what to \
             do in each state. Despite this simplicity it can carry out any procedure \
             that could be called mechanical; this is the Church-Turing thesis. Turing \
             also proved a limit. There is no general procedure that decides, for every \
             program and input, whether that program will eventually halt. Some \
             questions have no algorithm. Computation has both an inside and an edge, \
             and knowing where the edge lies is part of knowing the craft.",
        ),
        doc(
            "turing-test",
            "Turing asked whether machines can think and replaced the vague question \
             with a game. A judge converses by text with a human and a machine and \
             tries to tell which is which; if the machine cannot reliably be told \
             apart, Turing proposed that we grant it is, for the purpose, intelligent. \
             The move is philosophical. It makes intelligence a matter of observable \
             practice rather than of hidden essence, much as meaning is a matter of \
             public use rather than private sensation. What a thing does in the open, \
             answerable to others, is what we can know of its mind.",
        ),
        doc(
            "code-as-rule-following",
            "Good code follows the rules of the room it is written in. A new function \
             should read like the code around it, the same names, the same idioms, the \
             same density of comment, because a convention is a small form of life, \
             sustained by a community of readers who must maintain it. Clarity is a \
             social act: you write for the next person, against a public standard of \
             what counts as readable. The best engineers are expressive within the \
             rules rather than in spite of them; they make the shared language do new \
             work without breaking the practice that gives it meaning.",
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{run_pipeline, PipelineConfig};

    #[test]
    fn seed_corpus_passes_the_pipeline() {
        // Every seed doc is clean, permissive, and long enough — none should be
        // quarantined, and the corpus produces a well-formed manifest.
        let out = run_pipeline(seed_corpus(), &PipelineConfig::default());
        assert!(
            out.quarantine.is_empty(),
            "seed docs were quarantined: {:?}",
            out.quarantine
        );
        assert!(out.manifest.total_tokens > 0);
        let q = out.manifest.aggregate_quality.to_f32();
        assert!((0.0..=1.0).contains(&q) && q > 0.4, "aggregate quality {q}");
    }
}
