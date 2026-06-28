//! WP-G6 (Rust side) — the consensus-grade LoRA commitment.
//!
//! A generated adapter is registered on-chain (`LoRAFactory.adapterModelCommitment`), so
//! its digest must be deterministic and tamper-detecting. Like the rest of the program,
//! the committed path is **integer**: every factor is quantized onto the Q16 grid and the
//! raw `i64`s are hashed — no float bytes. The digest is **rank-atom-order-independent**
//! (the rank-`K` factorization has no canonical atom order, so we hash the *sorted*
//! multiset of per-atom `(B column, A row)` pairs), which mirrors the permutation-
//! invariance the on-chain verifier needs. The paired TLA+ spec
//! `nat/formal/LoraRegistration.tla` proves the registration protocol built on this digest.

use crate::LoraAdapter;
use citrate_fed_types::lora::{lora_commitment as kernel_lora_commitment, LoraFactors};

/// The Q16-exact, rank-order-independent, tamper-detecting LoRA commitment.
///
/// As of Gate-4 WP-W0 (stage 2) the digest is computed by the shared `citrate-fed-types`
/// kernel, so `citrate-chain` verifies `LoRAFactory.adapterModelCommitment` against the
/// *same code*. The kernel preserves the original domain (`nat-lora-commit-v1`) and
/// serialization, so the committed bytes are byte-identical — the frozen `bd08b278…` test
/// below is the regression guard.
pub fn lora_commitment(a: &LoraAdapter) -> String {
    kernel_lora_commitment(&LoraFactors {
        zone_tag: a.zone as u8,
        rank: a.rank,
        dim_out: a.dim_out,
        dim_in: a.dim_in,
        alpha: a.alpha,
        matrix_a: a.matrix_a.clone(),
        matrix_b: a.matrix_b.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LoraGenerator, SkillAtom, ZoneId};

    fn sample_adapter() -> LoraAdapter {
        let atoms = vec![
            SkillAtom {
                u: vec![1.0, -0.5, 0.25, 0.0],
                v: vec![0.3, -0.2, 0.1, 0.4, -0.1, 0.2],
            },
            SkillAtom {
                u: vec![0.0, 0.5, -0.5, 1.0],
                v: vec![-0.3, 0.2, 0.5, -0.1, 0.0, 0.3],
            },
        ];
        let mut gen = LoraGenerator::new(ZoneId::PF, atoms, 3);
        gen.fit(
            &[
                vec![0.0, 0.0, 0.0],
                vec![1.0, 0.0, 1.0],
                vec![0.0, 1.0, 1.0],
            ],
            &[vec![0.2, 0.1], vec![0.6, 0.3], vec![0.4, 0.5]],
            1e-6,
        );
        gen.generate(&[0.5, 0.5, 0.5])
    }

    #[test]
    fn commitment_is_deterministic() {
        let a = sample_adapter();
        assert_eq!(lora_commitment(&a), lora_commitment(&a));
    }

    #[test]
    fn commitment_detects_factor_tampering() {
        let a = sample_adapter();
        let before = lora_commitment(&a);
        let mut t = a.clone();
        t.matrix_a[0][0] += 0.01;
        assert_ne!(
            before,
            lora_commitment(&t),
            "tampering a factor must flip the digest"
        );
    }

    #[test]
    fn commitment_is_rank_atom_order_independent() {
        let a = sample_adapter();
        // swap the two rank atoms (B columns and A rows together) — same ΔW, must commit equal.
        let mut swapped = a.clone();
        for o in 0..a.dim_out {
            swapped.matrix_b[o].swap(0, 1);
        }
        swapped.matrix_a.swap(0, 1);
        assert_eq!(lora_commitment(&a), lora_commitment(&swapped));
    }

    // FROZEN golden bytes — the consensus-grade ratchet. Regenerate intentionally only
    // when the committed serialization changes (review before re-freezing).
    #[test]
    fn commitment_is_frozen() {
        assert_eq!(
            lora_commitment(&sample_adapter()),
            "bd08b278d9579226abdfaf4b91afee5da82cb80f13ad0d81e612cb4607c30132"
        );
    }
}
