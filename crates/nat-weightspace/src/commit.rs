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

/// The largest finite `f32` magnitude that survives `Q16::from_f32` without
/// saturating the `i64` grid: `i64::MAX / 65536 ≈ 1.407e14`. Any `|w|` above this
/// is mapped onto the same grid point a legal saturating value occupies, so it
/// cannot be committed injectively.
const Q16_MAX_FINITE_MAGNITUDE: f32 = 1.4073748e14;

/// Why a weight graph cannot be committed soundly (NAT2-B-003).
#[derive(Debug, Clone, PartialEq)]
pub enum CommitError {
    /// A node feature or edge weight was non-finite (`NaN`/`±inf`), which
    /// `Q16::from_f32` collapses to raw `0` — colliding with a genuine zero weight.
    NonFiniteWeight,
    /// A node feature or edge weight was outside the representable Q16 grid, so it
    /// saturates to `±i64::MAX` and collides with other saturating magnitudes.
    OutOfGridWeight(f32),
}

impl std::fmt::Display for CommitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CommitError::NonFiniteWeight => write!(f, "commit: non-finite weight"),
            CommitError::OutOfGridWeight(w) => {
                write!(
                    f,
                    "commit: weight {w} is outside the representable Q16 grid"
                )
            }
        }
    }
}

impl std::error::Error for CommitError {}

fn check_weight(w: f32) -> Result<(), CommitError> {
    if !w.is_finite() {
        return Err(CommitError::NonFiniteWeight);
    }
    if w.abs() > Q16_MAX_FINITE_MAGNITUDE {
        return Err(CommitError::OutOfGridWeight(w));
    }
    Ok(())
}

/// Validate that every feature and edge weight in `g` lies on the representable,
/// injective part of the Q16 grid (NAT2-B-003). `canonical_digest` quantizes every
/// float through `Q16::from_f32`, which maps `NaN`, `±inf` and `±0.0` onto raw `0`
/// and saturates any `|w| ≳ 1.4e14` to `±i64::MAX` — so a non-finite or
/// out-of-grid model commits to the *same* digest as a materially different legal
/// one. Any committed path must call this first and fail closed on `Err`, instead
/// of mapping bad inputs onto a grid point a legal value also occupies. Honest,
/// finite, in-range weights pass and their digest is unchanged, so the frozen
/// consensus goldens are unaffected.
pub fn validate_commit_domain(g: &WeightGraph) -> Result<(), CommitError> {
    for node in &g.nodes {
        for &f in &node.feats {
            check_weight(f)?;
        }
    }
    for e in &g.edges {
        check_weight(e.weight)?;
    }
    Ok(())
}

/// The fail-closed commitment entry point (NAT2-B-003): validate the weight domain,
/// then compute the canonical digest. This is the constructor a sound on-chain
/// weight commitment must use — it rejects the non-finite / saturating inputs that
/// [`canonical_digest`] would otherwise silently collide, while returning the exact
/// same digest as [`canonical_digest`] for every honest finite model.
pub fn commit_checked(g: &WeightGraph) -> Result<String, CommitError> {
    validate_commit_domain(g)?;
    Ok(canonical_digest(g))
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

    // NAT2-B-002 RED-witness (disposition: HELD/OWNER — the fix is a
    // canonicalization change to a *consensus-grade* commitment whose golden
    // bytes above are a deliberately review-gated ratchet; re-freezing them
    // invalidates any previously-published commitment, an owner decision).
    //
    // `lower_nat` emits every entry of a readout/projection ROW as a parallel
    // edge between the SAME `(gate, hidden, kind)` triple, and `canonical_digest`
    // hashes only the sorted multiset — so permuting the entries WITHIN a row
    // (which input channel gets which weight, i.e. a functionally different map)
    // leaves the digest bit-identical. This is strictly weaker than the
    // transformer path (`fan_in` varies `src` per column, binding column order).
    //
    // This test asserts the CURRENT (permutation-blind) behavior as a canary.
    // The fix binds the in-channel position into the pre-image (e.g. an
    // `(out_index, in_index)` slot hashed alongside `kind`+`weight`), which makes
    // the two digests DIFFER while keeping neuron-relabel invariance
    // (`digest_is_permutation_invariant`) intact — invert this test then, and
    // re-freeze the goldens above under owner review.
    #[test]
    fn within_row_permutation_leaves_digest_unchanged_nat2_b_002_witness() {
        use crate::ZoneWeights;

        let ckpt_a = nat_checkpoint(11);
        let mut ckpt_b = ckpt_a.clone();
        // Reverse the first readout row of the first zone: a genuine permutation
        // of which input channel maps to output channel 0 — a DIFFERENT model.
        let wo = match &mut ckpt_b.zone_weights[0].1 {
            ZoneWeights::Attention { wo, .. } => wo,
            ZoneWeights::Ssm { wo, .. } => wo,
        };
        wo.w[0].reverse();

        let ga = lower_nat(&ckpt_a);
        let gb = lower_nat(&ckpt_b);

        // The graphs are genuinely different (edge order encodes the mapping)...
        assert_ne!(ga, gb, "the two checkpoints are functionally different");
        // ...yet the "tamper-detecting" commitment cannot tell them apart.
        assert_eq!(
            canonical_digest(&ga),
            canonical_digest(&gb),
            "WITNESS: within-row permutation is invisible to the commitment"
        );
    }

    /// NAT2-B-003 tripwire + witness. `canonical_digest` quantizes through
    /// `Q16::from_f32`, which maps `NaN`/`±inf` and out-of-grid magnitudes onto grid
    /// points a legal weight also occupies — so materially different models collide.
    /// The witness asserts the CURRENT unchecked collision (a `NaN` edge and a `0`
    /// edge produce the same digest); the fix is the fail-closed `commit_checked`,
    /// which rejects the non-finite / saturating inputs while leaving every honest
    /// digest — and therefore the frozen goldens — unchanged.
    #[test]
    fn commit_checked_rejects_non_finite_and_out_of_grid_nat2_b_003() {
        let g = lower_nat(&nat_checkpoint(11));
        // Honest finite model: the checked path equals the unchecked digest.
        assert_eq!(commit_checked(&g).unwrap(), canonical_digest(&g));

        // WITNESS: a NaN weight collapses to raw 0, colliding with a zeroed sibling.
        let mut nan_g = g.clone();
        nan_g.edges[0].weight = f32::NAN;
        let mut zero_g = g.clone();
        zero_g.edges[0].weight = 0.0;
        assert_eq!(
            canonical_digest(&nan_g),
            canonical_digest(&zero_g),
            "WITNESS: NaN and 0 collide in the unchecked digest"
        );

        // The fail-closed path rejects the non-finite weight instead of committing it.
        assert_eq!(commit_checked(&nan_g), Err(CommitError::NonFiniteWeight));

        // And rejects a saturating out-of-grid magnitude (1e30 → i64::MAX).
        let mut big_g = g.clone();
        big_g.edges[0].weight = 1e30;
        assert!(matches!(
            commit_checked(&big_g),
            Err(CommitError::OutOfGridWeight(_))
        ));
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
