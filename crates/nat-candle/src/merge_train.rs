//! WP-2 — the differentiable merge, reconciled to the hard Q16.16 provenance
//! merge (ADR-0006).
//!
//! Inference composes survivors with the canonical, non-differentiable
//! `nat_provenance::prune_and_reweight` (hard top-k by score, then Q16.16
//! weighted sum) — that is the product and stays the **single** implementation of
//! the decision. Training needs gradients, so it composes with a *soft* merge:
//! `softmax(scores / τ)` over the zones, then a weighted sum of the per-zone
//! summaries. Both consume the **same per-zone scores**.
//!
//! The reconciliation (the sprint's primary risk, R1): hardening the soft weights
//! — taking the `keep` highest-weighted zones — must reproduce
//! `prune_and_reweight`'s survivor set. It does, because `softmax(·/τ)` is
//! strictly monotonic in the score, so top-k by soft weight == top-k by score ==
//! the canonical survivors. The `reconciliation_*` tests pin this against the
//! canonical decision over a battery, so a future change that fed the soft merge a
//! different signal than the decision would fail loudly. Annealing τ → 0 drives
//! the soft weights toward the hard one-hot, bridging training to the recorded
//! decision.

use candle_core::{Result, Tensor, D};

/// Differentiable soft merge weights: `softmax(scores / τ)` over the last
/// (zone) dim. Large τ → uniform; τ → 0 → one-hot on the top-scoring zone.
/// `scores`: `(batch, n_zones)` → weights `(batch, n_zones)`.
pub fn soft_weights(scores: &Tensor, tau: f64) -> Result<Tensor> {
    debug_assert!(tau > 0.0, "temperature must be positive");
    let scaled = scores.affine(1.0 / tau, 0.0)?;
    candle_nn::ops::softmax(&scaled, D::Minus1)
}

/// Compose per-zone summaries by the soft weights — a differentiable weighted sum.
/// `weights`: `(batch, n_zones)`, `summaries`: `(batch, n_zones, d_out)` →
/// composed `(batch, d_out)`. Gradient flows to both the weights (hence the
/// scores) and the summaries (hence every zone core).
pub fn compose(weights: &Tensor, summaries: &Tensor) -> Result<Tensor> {
    let (b, n) = weights.dims2()?;
    let w = weights.reshape((b, 1, n))?;
    let out = w.matmul(summaries)?; // (b, 1, d_out)
    out.squeeze(1)
}

/// Harden the soft weights to a survivor index set: the `keep` highest-weighted
/// zones, ties broken by ascending index. This is the bridge to the canonical
/// decision — see the `reconciliation_*` tests, which pin it against
/// `nat_provenance::prune_and_reweight`. When the input zones are listed in
/// canonical `ZoneId` order, the index tie-break matches the decision's `ZoneId`
/// tie-break, so the two agree exactly.
pub fn argtopk(weights: &[f32], keep: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..weights.len()).collect();
    idx.sort_by(|&a, &b| {
        weights[b]
            .partial_cmp(&weights[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx.truncate(keep.min(weights.len()));
    idx.sort_unstable();
    idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use nat_provenance::prune_and_reweight;
    use nat_types::{ZoneId, Q16};

    fn softmax_cpu(scores: &[f32], tau: f64) -> Vec<f32> {
        let dev = Device::Cpu;
        let n = scores.len();
        let t = Tensor::from_vec(scores.to_vec(), (1, n), &dev).unwrap();
        soft_weights(&t, tau)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    // The learned zones in canonical order, so an index lines up with a ZoneId and
    // the argtopk index tie-break matches prune_and_reweight's ZoneId tie-break.
    fn zones() -> Vec<ZoneId> {
        ZoneId::LEARNED.to_vec()
    }

    #[test]
    fn reconciliation_matches_canonical_decision_over_a_battery() {
        // For a battery of distinct score vectors, hardening the soft weights
        // reproduces the canonical survivor set exactly. This is the ADR-0006
        // reconciliation — the differentiable path and the recorded decision agree
        // on which zones survive.
        let zs = zones();
        let battery: &[[f32; 5]] = &[
            [0.9, 0.1, 0.5, 0.3, 0.7],
            [0.2, 0.2, 0.9, 0.8, 0.1],
            [0.05, 0.6, 0.55, 0.4, 0.95],
            [1.0, 0.9, 0.8, 0.7, 0.6],
            [0.1, 0.2, 0.3, 0.4, 0.5],
        ];
        for thr in [0.0f32, 0.2, 0.5, 0.8] {
            for row in battery {
                let scored: Vec<(ZoneId, Q16)> = zs
                    .iter()
                    .zip(row.iter())
                    .map(|(&z, &s)| (z, Q16::from_f32(s)))
                    .collect();
                let decision = prune_and_reweight(&scored, Q16::from_f32(thr));
                let keep = decision.survivors.len();

                let w = softmax_cpu(row, 1.0);
                let hardened: Vec<ZoneId> = argtopk(&w, keep).iter().map(|&i| zs[i]).collect();

                assert_eq!(
                    hardened, decision.survivors,
                    "thr={thr} row={row:?}: hardened soft != canonical survivors"
                );
            }
        }
    }

    #[test]
    fn argtopk_tiebreak_matches_canonical_on_ties() {
        // Equal scores → prune keeps the lower ZoneId; argtopk (canonical-order
        // zones) keeps the lower index. They must coincide.
        let zs = zones();
        let row = [0.5f32, 0.5, 0.5, 0.5, 0.5];
        let scored: Vec<(ZoneId, Q16)> = zs.iter().map(|&z| (z, Q16::from_f32(0.5))).collect();
        let decision = prune_and_reweight(&scored, Q16::from_f32(0.5));
        let w = softmax_cpu(&row, 1.0);
        let hardened: Vec<ZoneId> = argtopk(&w, decision.survivors.len())
            .iter()
            .map(|&i| zs[i])
            .collect();
        assert_eq!(hardened, decision.survivors);
    }

    #[test]
    fn annealing_concentrates_mass_on_survivors() {
        // Lower τ puts more soft-weight mass on the canonical survivors — the
        // bridge from the soft training merge toward the hard recorded decision.
        let zs = zones();
        let row = [0.9f32, 0.1, 0.5, 0.3, 0.7];
        let scored: Vec<(ZoneId, Q16)> = zs
            .iter()
            .zip(row.iter())
            .map(|(&z, &s)| (z, Q16::from_f32(s)))
            .collect();
        let survivors = prune_and_reweight(&scored, Q16::from_f32(0.5)).survivors;
        let surv_idx: Vec<usize> = (0..zs.len())
            .filter(|i| survivors.contains(&zs[*i]))
            .collect();

        let mass = |tau: f64| -> f32 {
            let w = softmax_cpu(&row, tau);
            surv_idx.iter().map(|&i| w[i]).sum()
        };
        assert!(
            mass(0.1) > mass(2.0),
            "annealing did not concentrate mass: {} vs {}",
            mass(0.1),
            mass(2.0)
        );
    }

    #[test]
    fn compose_is_differentiable_in_scores_and_summaries() {
        let dev = Device::Cpu;
        // Scores and summaries as leaf variables so we can read their gradients.
        let scores = candle_core::Var::from_tensor(
            &Tensor::from_vec(vec![0.2f32, 0.5, 0.9], (1, 3), &dev).unwrap(),
        )
        .unwrap();
        let summaries = candle_core::Var::from_tensor(
            &Tensor::from_vec(
                (0..3 * 4).map(|i| i as f32 * 0.1).collect(),
                (1, 3, 4),
                &dev,
            )
            .unwrap(),
        )
        .unwrap();
        let w = soft_weights(scores.as_tensor(), 1.0).unwrap();
        let composed = compose(&w, summaries.as_tensor()).unwrap();
        let loss = composed.sqr().unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        for v in [scores.as_tensor(), summaries.as_tensor()] {
            let g = grads.get(v).expect("gradient present");
            let s = g
                .abs()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(s.is_finite() && s > 0.0, "vanishing/absent gradient");
        }
    }
}
