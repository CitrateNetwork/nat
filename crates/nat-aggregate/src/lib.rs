//! `nat-aggregate` — verifiable gradient aggregation (WS-1, the spine).
//!
//! The on-chain outer-step (DiLoCo barrier): workers run H local steps and emit a
//! **pseudo-gradient** (a delta vector); each outer round aggregates them with a
//! **bucketed coordinate-wise trimmed mean in Q16 fixed-point**. The reduction is a
//! deterministic, bit-reproducible function of the submitted values — the property that
//! lets heterogeneous untrusted validators reconcile the aggregate on-chain with no
//! tolerance window (frontier bet #1).
//!
//! As of Gate-4 WP-W0 the reduction itself lives in the shared **`citrate-fed-types`**
//! boundary kernel (so `citrate-chain`'s challenge game re-executes the *same code*); this
//! crate **re-exports** it unchanged and keeps the int8 `compress` path (research) plus the
//! frozen-digest regression guard that proves the re-export still reproduces the golden
//! bytes. The reduction is the Rust counterpart of the TLA+ pair
//! `nat/formal/GradientAggregation.tla` (determinism) + `GradientAggregationAdversarial.tla`
//! (robustness): same strict-total-order trim `(value, worker index)`, no floats, no
//! hash-map iteration on the path.

pub mod compress;

// The aggregation contracts moved to the audited boundary kernel (Gate-4 WP-W0). Re-export
// them so `nat_aggregate::aggregate` (and the `PseudoGradient`/digest types) stay
// source-stable for every existing caller while the chain links the identical code.
pub use citrate_fed_types::aggregate::{
    aggregate, bucket_of, digest_of, AggregateError, AggregateResult, PseudoGradient,
};

#[cfg(test)]
mod tests {
    use super::*;
    use nat_types::Q16;

    fn pg(id: &str, coords: &[f32]) -> PseudoGradient {
        PseudoGradient::new(id, coords.iter().map(|&v| Q16::from_f32(v)).collect())
    }

    // One bucket per worker (bucket_count high + distinct ids) isolates the coordinate
    // trimmed-mean behaviour from the bucketing.
    fn one_per_bucket(grads: &[PseudoGradient], trim: usize) -> AggregateResult {
        aggregate(grads, trim, 64, b"seed-test").expect("aggregate")
    }

    #[test]
    fn determinism_same_inputs_same_digest() {
        let g = vec![
            pg("a", &[1.0, 2.0]),
            pg("b", &[1.5, 2.5]),
            pg("c", &[1.25, 2.25]),
        ];
        let r1 = one_per_bucket(&g, 0);
        let r2 = one_per_bucket(&g, 0);
        assert_eq!(r1, r2);
        assert_eq!(r1.digest.len(), 64);
    }

    #[test]
    fn order_independent_under_shuffle() {
        let g = vec![
            pg("a", &[1.0, 8.0]),
            pg("b", &[2.0, 7.0]),
            pg("c", &[3.0, 6.0]),
            pg("d", &[4.0, 5.0]),
        ];
        let mut shuffled = g.clone();
        shuffled.rotate_right(2);
        shuffled.swap(0, 3);
        assert_eq!(one_per_bucket(&g, 1), one_per_bucket(&shuffled, 1));
    }

    #[test]
    fn byzantine_outliers_are_trimmed_within_honest_band() {
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
        assert_eq!(r.aggregate, vec![Q16::from_f32(10.0)]);
    }

    #[test]
    fn trim_budget_too_large_fails_closed() {
        let g = vec![pg("a", &[1.0]), pg("b", &[2.0])];
        let err = aggregate(&g, 1, 64, b"s").unwrap_err();
        assert_eq!(
            err,
            AggregateError::TrimBudgetTooLarge {
                trim: 1,
                bucket_count: 2
            }
        );
    }

    #[test]
    fn dimension_mismatch_fails_closed() {
        let g = vec![pg("a", &[1.0, 2.0]), pg("b", &[1.0])];
        assert_eq!(
            aggregate(&g, 0, 64, b"s").unwrap_err(),
            AggregateError::DimensionMismatch {
                expected: 2,
                found: 1
            }
        );
    }

    #[test]
    fn empty_fails_closed() {
        assert_eq!(
            aggregate(&[], 0, 4, b"s").unwrap_err(),
            AggregateError::Empty
        );
    }

    #[test]
    fn bucketing_is_deterministic() {
        let b1 = bucket_of(b"round-7", "node-xyz", 16);
        let b2 = bucket_of(b"round-7", "node-xyz", 16);
        assert_eq!(b1, b2);
        assert!(b1 < 16);
        let _ = bucket_of(b"round-8", "node-xyz", 16);
    }

    /// Tiny deterministic LCG — a fixed-seed value source (no external dep) for the sweep.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 16
        }
    }

    #[test]
    fn sweep_determinism_and_byzantine_robustness() {
        let mut rng = Lcg(0xA66D_E7E1_u64);
        for round in 0..2000u64 {
            let honest = 4 + (rng.next() % 6) as usize;
            let byz = (rng.next() % 3) as usize;
            let trim = byz.max(1);
            let mut g = Vec::new();
            for i in 0..honest {
                let v = 9.5 + (rng.next() % 1000) as f32 / 1000.0;
                g.push(pg(&format!("h{round}_{i}"), &[v]));
            }
            for i in 0..byz {
                let extreme = if i % 2 == 0 { 50000.0 } else { -50000.0 };
                g.push(pg(&format!("byz{round}_{i}"), &[extreme]));
            }
            let r1 = match aggregate(&g, trim, 64, b"sweep") {
                Ok(r) => r,
                Err(AggregateError::TrimBudgetTooLarge { .. }) => continue,
                Err(e) => panic!("round {round}: unexpected error {e}"),
            };
            let r2 = aggregate(&g, trim, 64, b"sweep").expect("aggregate");
            assert_eq!(r1, r2);
            let agg = r1.aggregate[0].to_f32();
            assert!(
                (9.4..=10.6).contains(&agg),
                "round {round}: agg {agg} escaped honest band"
            );
        }
    }

    // MIGRATION REGRESSION GUARD — the re-exported kernel aggregation must reproduce the
    // exact pre-migration golden bytes (same inputs as the original frozen test and as
    // `citrate-fed-types::aggregate::frozen_aggregate_digest_matches_nat`). If this drifts,
    // the WP-W0 adoption changed a committed-byte path.
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
            r.digest, "e79c5a6381c2e761f264d1c64dfdf12016c08ca3494ee909736ec84d00aa59a1",
            "Q16 aggregation digest drifted post-adoption — the kernel re-export changed bytes"
        );
    }
}
