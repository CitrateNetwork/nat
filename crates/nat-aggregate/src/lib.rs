//! `nat-aggregate` — verifiable gradient aggregation (WS-1, the spine).
//!
//! The on-chain outer-step (DiLoCo barrier): workers run H local steps and emit a
//! **pseudo-gradient** (a delta vector); each outer round aggregates them with a
//! **bucketed coordinate-wise trimmed mean in Q16 fixed-point**. The reduction is a
//! deterministic, bit-reproducible function of the submitted values — the property
//! that lets heterogeneous untrusted validators reconcile the aggregate on-chain with
//! no tolerance window (frontier bet #1), and that pins the result inside the honest
//! band when the trim budget covers the Byzantine count.
//!
//! This crate is the **pure core**. It is the Rust counterpart of the TLA+ pair
//! `nat/formal/GradientAggregation.tla` (determinism) + `GradientAggregationAdversarial.tla`
//! (robustness): the coordinate trimmed mean below uses the **same strict-total-order
//! trim** (value, then worker index) the spec's `Below`/`RankOf`/`TrimmedOf` operators
//! define, so the spec's invariants describe this code. No floats, no hash-map
//! iteration on the aggregation path: every step is integer arithmetic over a sorted
//! vector, so two nodes compute identical bits.
//!
//! Commitment + on-chain challenge (KZG/Merkle via `0x0107/08/09`, the
//! optimistic-challenge contract generalizing `DisputeResolution.sol`) are the
//! citrate-chain integration (AGG-S1 WP-4); this crate produces the aggregate and a
//! deterministic digest an auditor recomputes.

use nat_types::Q16;
use sha2::{Digest, Sha256};

pub mod compress;

/// One worker's pseudo-gradient for an outer round: a fixed-point delta vector. All
/// pseudo-gradients in a round share dimensionality `dim`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PseudoGradient {
    /// The submitting worker's identity (the bucket-seed and the audit trail key).
    pub node_id: String,
    /// The delta, one Q16 coordinate per model parameter (coordinate).
    pub coords: Vec<Q16>,
}

impl PseudoGradient {
    pub fn new(node_id: impl Into<String>, coords: Vec<Q16>) -> Self {
        PseudoGradient { node_id: node_id.into(), coords }
    }
    pub fn dim(&self) -> usize {
        self.coords.len()
    }
}

/// Why an aggregation could not be computed (fail-closed at the boundary).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AggregateError {
    /// No pseudo-gradients were submitted.
    Empty,
    /// Submissions disagreed on dimensionality.
    DimensionMismatch { expected: usize, found: usize },
    /// `2 * trim >= bucket_count`, so the kept band would be empty.
    TrimBudgetTooLarge { trim: usize, bucket_count: usize },
    /// `bucket_count == 0`.
    NoBuckets,
}

impl std::fmt::Display for AggregateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AggregateError::Empty => write!(f, "aggregate: no pseudo-gradients"),
            AggregateError::DimensionMismatch { expected, found } => {
                write!(f, "aggregate: dimension mismatch (expected {expected}, found {found})")
            }
            AggregateError::TrimBudgetTooLarge { trim, bucket_count } => {
                write!(f, "aggregate: trim {trim} too large for {bucket_count} buckets")
            }
            AggregateError::NoBuckets => write!(f, "aggregate: bucket_count must be >= 1"),
        }
    }
}
impl std::error::Error for AggregateError {}

/// The result of an outer-round aggregation: the aggregated pseudo-gradient and a
/// deterministic digest over its raw Q16 bytes (the value an auditor recomputes and
/// the on-chain challenge commits).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AggregateResult {
    pub aggregate: Vec<Q16>,
    pub digest: String,
}

/// Deterministically assign a worker to one of `bucket_count` buckets by hashing
/// `seed || node_id`. Production replaces the seed with a VRF output (so the
/// partition is unpredictable until the round opens); the assignment function is
/// identical and reproducible given the seed. Bucketing makes the coordinate median
/// non-iid-robust without making it gameable.
pub fn bucket_of(seed: &[u8], node_id: &str, bucket_count: usize) -> usize {
    let mut h = Sha256::new();
    h.update(seed);
    h.update([0u8]); // domain separator between seed and id
    h.update(node_id.as_bytes());
    let d = h.finalize();
    let v = u64::from_le_bytes(d[0..8].try_into().expect("sha256 is 32 bytes"));
    (v % bucket_count as u64) as usize
}

/// Coordinate-wise trimmed mean of one coordinate across submissions, in Q16. This is
/// the heart of the reduction and the exact Rust image of the TLA+ operator:
/// sort by the **strict total order** `(value, source_index)` (so equal values break
/// by the smaller index — no tie can make two nodes disagree), drop the lowest `trim`
/// and highest `trim`, and average the kept band with deterministic integer division.
///
/// `values` is `(coordinate_value, source_index)`; `source_index` is the stable order
/// key (bucket index) the spec calls the worker index.
fn coordinate_trimmed_mean(values: &mut [(Q16, usize)], trim: usize) -> Q16 {
    // Strict total order: value ascending, ties broken by the smaller source index.
    values.sort_unstable_by(|a, b| a.0.raw().cmp(&b.0.raw()).then(a.1.cmp(&b.1)));
    let kept = &values[trim..values.len() - trim];
    // Sum the kept band exactly (i128 accumulator: no mid-sum overflow), then divide
    // by the kept count. Integer division is deterministic across platforms.
    let sum: i128 = kept.iter().map(|(q, _)| q.raw() as i128).sum();
    Q16::from_raw((sum / kept.len() as i128) as i64)
}

/// Coordinate-wise mean of a bucket's pseudo-gradients (the within-bucket reduction
/// before the across-bucket trimmed median). All inputs share `dim`.
fn bucket_mean(grads: &[&PseudoGradient], dim: usize) -> Vec<Q16> {
    (0..dim)
        .map(|c| {
            let sum: i128 = grads.iter().map(|g| g.coords[c].raw() as i128).sum();
            Q16::from_raw((sum / grads.len() as i128) as i64)
        })
        .collect()
}

/// Aggregate one outer round: bucket the pseudo-gradients (seed-derived), reduce each
/// bucket to its coordinate-wise mean, then take the **coordinate-wise trimmed mean
/// across the non-empty bucket means**. Returns the aggregated pseudo-gradient and its
/// digest. Deterministic and order-independent: the result depends only on the *set*
/// of (node_id, coords) and the seed, never on submission order.
pub fn aggregate(
    grads: &[PseudoGradient],
    trim: usize,
    bucket_count: usize,
    seed: &[u8],
) -> Result<AggregateResult, AggregateError> {
    if bucket_count == 0 {
        return Err(AggregateError::NoBuckets);
    }
    let first = grads.first().ok_or(AggregateError::Empty)?;
    let dim = first.dim();
    for g in grads {
        if g.dim() != dim {
            return Err(AggregateError::DimensionMismatch { expected: dim, found: g.dim() });
        }
    }

    // Partition into buckets by seed (deterministic). Empty buckets are dropped, so
    // the trim budget is checked against the number of *non-empty* buckets.
    let mut buckets: Vec<Vec<&PseudoGradient>> = vec![Vec::new(); bucket_count];
    for g in grads {
        buckets[bucket_of(seed, &g.node_id, bucket_count)].push(g);
    }
    let bucket_means: Vec<Vec<Q16>> =
        buckets.iter().filter(|b| !b.is_empty()).map(|b| bucket_mean(b, dim)).collect();

    if 2 * trim >= bucket_means.len() {
        return Err(AggregateError::TrimBudgetTooLarge { trim, bucket_count: bucket_means.len() });
    }

    // Coordinate-wise trimmed mean across the bucket means.
    let aggregate: Vec<Q16> = (0..dim)
        .map(|c| {
            let mut col: Vec<(Q16, usize)> =
                bucket_means.iter().enumerate().map(|(i, m)| (m[c], i)).collect();
            coordinate_trimmed_mean(&mut col, trim)
        })
        .collect();

    let digest = digest_of(&aggregate);
    Ok(AggregateResult { aggregate, digest })
}

/// `H(raw Q16 little-endian bytes)` of an aggregate vector — the deterministic
/// commitment an auditor recomputes (and the on-chain challenge anchors). No floats
/// touch this path; the raw `i64`s are the canonical bytes.
pub fn digest_of(v: &[Q16]) -> String {
    let mut h = Sha256::new();
    for q in v {
        h.update(q.raw().to_le_bytes());
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

    fn pg(id: &str, coords: &[f32]) -> PseudoGradient {
        PseudoGradient::new(id, coords.iter().map(|&v| Q16::from_f32(v)).collect())
    }

    // One bucket per worker (bucket_count high + distinct ids) isolates the
    // coordinate trimmed-mean behaviour from the bucketing.
    fn one_per_bucket(grads: &[PseudoGradient], trim: usize) -> AggregateResult {
        aggregate(grads, trim, 64, b"seed-test").expect("aggregate")
    }

    #[test]
    fn determinism_same_inputs_same_digest() {
        let g = vec![pg("a", &[1.0, 2.0]), pg("b", &[1.5, 2.5]), pg("c", &[1.25, 2.25])];
        let r1 = one_per_bucket(&g, 0);
        let r2 = one_per_bucket(&g, 0);
        assert_eq!(r1, r2);
        assert_eq!(r1.digest.len(), 64);
    }

    #[test]
    fn order_independent_under_shuffle() {
        let g = vec![pg("a", &[1.0, 8.0]), pg("b", &[2.0, 7.0]), pg("c", &[3.0, 6.0]), pg("d", &[4.0, 5.0])];
        let mut shuffled = g.clone();
        shuffled.rotate_right(2);
        shuffled.swap(0, 3);
        // Same set, different order -> identical aggregate + digest (the gather is a
        // function of the set, per GradientAggregation.tla::DeterministicAggregation).
        assert_eq!(one_per_bucket(&g, 1), one_per_bucket(&shuffled, 1));
    }

    #[test]
    fn byzantine_outliers_are_trimmed_within_honest_band() {
        // 5 honest workers tightly around 10.0, plus 2 Byzantine extremes. trim=2
        // covers the 2 Byzantine, so the result must stay in the honest band.
        let g = vec![
            pg("h1", &[10.0]),
            pg("h2", &[10.0]),
            pg("h3", &[10.0]),
            pg("h4", &[10.0]),
            pg("h5", &[10.0]),
            pg("byz_lo", &[-9000.0]),
            pg("byz_hi", &[9000.0]),
        ];
        let r = one_per_bucket(&g, 2);
        // Both extremes trimmed -> honest unanimity preserved (== 10.0), exactly the
        // GradientAggregationAdversarial.tla::HonestUnanimityPreserved property.
        assert_eq!(r.aggregate, vec![Q16::from_f32(10.0)]);
    }

    #[test]
    fn trim_budget_too_large_fails_closed() {
        let g = vec![pg("a", &[1.0]), pg("b", &[2.0])];
        // 2 non-empty buckets, trim=1 -> 2*1 >= 2 -> kept band empty -> error.
        let err = aggregate(&g, 1, 64, b"s").unwrap_err();
        assert_eq!(err, AggregateError::TrimBudgetTooLarge { trim: 1, bucket_count: 2 });
    }

    #[test]
    fn dimension_mismatch_fails_closed() {
        let g = vec![pg("a", &[1.0, 2.0]), pg("b", &[1.0])];
        assert_eq!(
            aggregate(&g, 0, 64, b"s").unwrap_err(),
            AggregateError::DimensionMismatch { expected: 2, found: 1 }
        );
    }

    #[test]
    fn empty_fails_closed() {
        assert_eq!(aggregate(&[], 0, 4, b"s").unwrap_err(), AggregateError::Empty);
    }

    #[test]
    fn bucketing_is_deterministic() {
        // Same seed + id -> same bucket, every time and on every platform.
        let b1 = bucket_of(b"round-7", "node-xyz", 16);
        let b2 = bucket_of(b"round-7", "node-xyz", 16);
        assert_eq!(b1, b2);
        assert!(b1 < 16);
        // Different seed generally moves the assignment (not asserted equal/!=, just
        // that the function reads the seed).
        let _ = bucket_of(b"round-8", "node-xyz", 16);
    }

    /// Tiny deterministic LCG — a fixed-seed value source (no external dep) for the
    /// in-process sweep (WP-3 layer-4).
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0 >> 16
        }
    }

    #[test]
    fn sweep_determinism_and_byzantine_robustness() {
        let mut rng = Lcg(0xA66D_E7E1_u64);
        for round in 0..2000u64 {
            // 4..9 honest workers tightly in [9.5, 10.5], plus 0..2 Byzantine extremes.
            let honest = 4 + (rng.next() % 6) as usize;
            let byz = (rng.next() % 3) as usize;
            let trim = byz.max(1); // trim budget covers the Byzantine count (β >= f)
            let mut g = Vec::new();
            for i in 0..honest {
                let v = 9.5 + (rng.next() % 1000) as f32 / 1000.0; // [9.5, 10.5)
                g.push(pg(&format!("h{round}_{i}"), &[v]));
            }
            for i in 0..byz {
                let extreme = if i % 2 == 0 { 50000.0 } else { -50000.0 };
                g.push(pg(&format!("byz{round}_{i}"), &[extreme]));
            }
            // With few workers in 64 buckets, hash collisions can leave <= 2*trim
            // non-empty buckets — the aggregator then correctly FAILS CLOSED, which is
            // a valid outcome, not a bug. Assert the properties only when it succeeds.
            let r1 = match aggregate(&g, trim, 64, b"sweep") {
                Ok(r) => r,
                Err(AggregateError::TrimBudgetTooLarge { .. }) => continue,
                Err(e) => panic!("round {round}: unexpected error {e}"),
            };
            // (1) DETERMINISM: recomputing yields identical bytes.
            let r2 = aggregate(&g, trim, 64, b"sweep").expect("aggregate");
            assert_eq!(r1, r2);
            // (2) ROBUSTNESS: the aggregate stays inside the honest band [9.5, 10.5],
            //     i.e. Byzantine extremes were trimmed (ByzantineCannotFlip...).
            let agg = r1.aggregate[0].to_f32();
            assert!((9.4..=10.6).contains(&agg), "round {round}: agg {agg} escaped honest band");
        }
    }

    // FROZEN GOLDEN BYTES — cross-platform determinism anchor (WP-3). If this digest
    // ever changes, the Q16 aggregation path drifted; that must be a deliberate,
    // reviewed change (the on-chain reconciliation depends on these exact bytes).
    #[test]
    fn frozen_aggregate_digest() {
        let g = vec![
            pg("alpha", &[1.0, -2.0, 3.5]),
            pg("beta", &[2.0, -1.0, 3.0]),
            pg("gamma", &[1.5, -1.5, 3.25]),
            pg("delta", &[1.75, -1.25, 3.1]),
        ];
        let r = aggregate(&g, 1, 64, b"frozen-seed-v1").expect("aggregate");
        assert_eq!(
            r.digest,
            "e79c5a6381c2e761f264d1c64dfdf12016c08ca3494ee909736ec84d00aa59a1",
            "Q16 aggregation digest drifted — review before re-freezing"
        );
    }
}
