//! WP-B2 — the consensus-grade weight commitment.
//!
//! For any on-chain weight forward pass, the committed weight digest must be (a)
//! **deterministic** and reproducible cross-architecture, (b) **permutation-invariant**
//! (relabeling neurons must not change what was committed — otherwise two honest peers
//! holding the same model commit different digests), and (c) **tamper-detecting** (any
//! weight change flips the digest). A flat hash of the weight bytes fails (a) and (b).
//!
//! We use a **Weisfeiler-Leman canonical digest**: iteratively refine a per-node color
//! from the Q16-quantized incident weights and neighbor colors (multisets, so order-free),
//! then hash the sorted multiset of final node colors and canonical edges. Everything on
//! this path is integer (Q16 raw `i64`) — no float — so the digest is bit-reproducible and
//! frozen (golden bytes below). The paired TLA+ spec `nat/formal/WeightCommitment.tla`
//! proves the soundness + tamper-detection this function realizes.

use crate::WeightGraph;
use nat_types::Q16;
use sha2::{Digest, Sha256};

/// WL refinement rounds. Three rounds distinguishes nodes up to 3-hop weighted structure,
/// ample for the model graphs here; more rounds only refine, never break invariance.
const WL_ROUNDS: usize = 3;
const DOMAIN: &[u8] = b"nat-weightspace-commit-v1";

type Color = [u8; 32];
/// One incident edge as seen from a node: (direction, kind slot, Q16 weight, neighbor).
type Incidence = (u8, u8, [u8; 8], usize);

fn h32(parts: &[&[u8]]) -> Color {
    let mut h = Sha256::new();
    for p in parts {
        h.update(p);
    }
    h.finalize().into()
}

fn q16_raw(w: f32) -> [u8; 8] {
    Q16::from_f32(w).raw().to_le_bytes()
}

/// The permutation-invariant, tamper-detecting, Q16-exact weight commitment.
pub fn canonical_digest(g: &WeightGraph) -> String {
    let n = g.nodes.len();

    // Initial colors: node kind one-hot + Q16-quantized features (integer, no float bytes).
    let mut colors: Vec<Color> = g
        .nodes
        .iter()
        .map(|node| {
            let mut feat_bytes = Vec::with_capacity(node.feats.len() * 8);
            for &f in &node.feats {
                feat_bytes.extend_from_slice(&q16_raw(f));
            }
            h32(&[b"node", &feat_bytes])
        })
        .collect();

    // Incidence: for each node, its incident edges as (direction, kind, Q16 weight, other).
    // direction 0 = outgoing, 1 = incoming; a self-loop contributes both.
    let mut incidence: Vec<Vec<Incidence>> = vec![Vec::new(); n];
    for e in &g.edges {
        let wq = q16_raw(e.weight);
        let slot = e.kind.slot() as u8;
        incidence[e.src].push((0, slot, wq, e.dst));
        incidence[e.dst].push((1, slot, wq, e.src));
    }

    // Weisfeiler-Leman refinement: each round folds the multiset of incident
    // (direction, kind, weight, neighbor-color) into a new color. Multisets are sorted,
    // so the result is independent of node/edge ordering.
    for _ in 0..WL_ROUNDS {
        let mut next = vec![[0u8; 32]; n];
        for v in 0..n {
            let mut tuples: Vec<Vec<u8>> = incidence[v]
                .iter()
                .map(|(dir, slot, wq, other)| {
                    let mut t = Vec::with_capacity(1 + 1 + 8 + 32);
                    t.push(*dir);
                    t.push(*slot);
                    t.extend_from_slice(wq);
                    t.extend_from_slice(&colors[*other]);
                    t
                })
                .collect();
            tuples.sort_unstable(); // canonical multiset order
            let mut h = Sha256::new();
            h.update(b"wl");
            h.update(colors[v]);
            for t in &tuples {
                h.update((t.len() as u32).to_le_bytes());
                h.update(t);
            }
            next[v] = h.finalize().into();
        }
        colors = next;
    }

    // Graph digest: sorted multiset of final node colors + sorted canonical edges
    // (keyed by endpoint colors, not indices → permutation-invariant).
    let mut node_ms: Vec<Color> = colors.clone();
    node_ms.sort_unstable();

    let mut edge_ms: Vec<Vec<u8>> = g
        .edges
        .iter()
        .map(|e| {
            let mut t = Vec::with_capacity(32 + 32 + 1 + 8);
            t.extend_from_slice(&colors[e.src]);
            t.extend_from_slice(&colors[e.dst]);
            t.push(e.kind.slot() as u8);
            t.extend_from_slice(&q16_raw(e.weight));
            t
        })
        .collect();
    edge_ms.sort_unstable();

    let mut h = Sha256::new();
    h.update(DOMAIN);
    h.update((n as u64).to_le_bytes());
    h.update((g.edges.len() as u64).to_le_bytes());
    for c in &node_ms {
        h.update(c);
    }
    for e in &edge_ms {
        h.update((e.len() as u32).to_le_bytes());
        h.update(e);
    }
    hex(&h.finalize())
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0xf) as usize] as char);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::permute_nodes;
    use crate::tests::{nat_checkpoint, transformer_checkpoint};
    use crate::{lower_nat, lower_transformer, EdgeKind, GraphEdge};

    #[test]
    fn digest_is_permutation_invariant() {
        for g in [
            lower_nat(&nat_checkpoint(11)),
            lower_transformer(&transformer_checkpoint(11, 2, 4)),
        ] {
            let perm: Vec<usize> = (0..g.nodes.len()).rev().collect();
            let permuted = permute_nodes(&g, &perm);
            assert_eq!(
                canonical_digest(&g),
                canonical_digest(&permuted),
                "commitment must not change under neuron relabeling"
            );
        }
    }

    #[test]
    fn digest_detects_weight_tampering() {
        let g = lower_nat(&nat_checkpoint(11));
        let before = canonical_digest(&g);
        let mut tampered = g.clone();
        // flip a single weight on a single edge — must change the digest.
        let target = tampered
            .edges
            .iter_mut()
            .find(|e| e.kind == EdgeKind::AttnQ)
            .expect("an AttnQ edge");
        target.weight += 0.01;
        assert_ne!(
            before,
            canonical_digest(&tampered),
            "tampering must be detected"
        );
    }

    #[test]
    fn digest_is_deterministic_cross_run() {
        let g = lower_transformer(&transformer_checkpoint(2, 3, 4));
        assert_eq!(canonical_digest(&g), canonical_digest(&g));
    }

    #[test]
    fn nat_and_transformer_commit_to_distinct_stable_digests() {
        let nat = canonical_digest(&lower_nat(&nat_checkpoint(11)));
        let xf = canonical_digest(&lower_transformer(&transformer_checkpoint(11, 2, 4)));
        assert_ne!(nat, xf, "different architectures → different commitments");
    }

    // FROZEN golden bytes — the consensus-grade ratchet. Regenerate intentionally only
    // when the committed canonicalization changes (and review before re-freezing).
    #[test]
    fn nat_commitment_is_frozen() {
        let g = lower_nat(&nat_checkpoint(11));
        assert_eq!(
            canonical_digest(&g),
            "5571addc39d022907445561f5a49700d9e9b293283f48d5ebef582a87baebcdd"
        );
    }

    #[test]
    fn transformer_commitment_is_frozen() {
        let g = lower_transformer(&transformer_checkpoint(11, 2, 4));
        assert_eq!(
            canonical_digest(&g),
            "79e3bdeb6ff532c17155506e52bccf9b890c4b606c9fa6334993df7c9f8379cc"
        );
    }

    #[test]
    fn adding_an_edge_changes_the_digest() {
        let g = lower_transformer(&transformer_checkpoint(11, 2, 4));
        let before = canonical_digest(&g);
        let mut more = g.clone();
        more.edges.push(GraphEdge {
            src: 0,
            dst: 1,
            kind: EdgeKind::FfnUp,
            weight: 0.3,
        });
        assert_ne!(before, canonical_digest(&more));
    }
}
