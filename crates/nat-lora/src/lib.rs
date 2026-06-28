//! `nat-lora` — the weight-conditioned LoRA generator (WS-3, frontier bet #3).
//!
//! WS-2 Layer B built the *reader*: a GMN latent (`nat-weightspace`) that diagnoses a
//! model's weight-space cross-architecture. WS-3 builds the *generator* — a hypernetwork
//! that turns that latent into a working low-rank adapter. The claim a one-attempt
//! distillation (WS-2 Layer A) cannot make: **read a peer's weights and emit an adapter
//! that installs the peer's capability, generalizing to peers never seen in meta-training.**
//! The control that proves it: a generator fed a *shuffled* latent must fail.
//!
//! Layer split (meta-plan §3):
//! - The generator + its meta-training + the transfer/ablation benchmark are **research
//!   ML** (f32, measured, gated). See [`LoraGenerator`].
//! - The LoRA **commitment + registration** ([`commit`], [`LoraRegistration`]) is
//!   **consensus-grade** (Q16, frozen bytes, tamper-detecting, paired TLA+
//!   `LoraRegistration.tla`).
//!
//! The emitted adapter is field-aligned to the on-chain `LoRAFactory.LoRAAdapter`
//! (`loraHash`/`baseModelHash`/`rank`/`alpha`/`adapterModelCommitment`) and the Rust
//! `core/learning::LoraAdapter` (`matrix_a [d×r]` / `matrix_b [r×d]`, `apply: W' = W + B·A`).

// Dense linear-algebra index loops (outer products, pivot search) read more clearly with
// explicit indices than with zipped iterators.
#![allow(clippy::needless_range_loop)]

pub mod commit;
mod linalg;

pub use nat_types::ZoneId;
use nat_weightspace::{encoder::GmnEncoder, WeightGraph};

// ---------------------------------------------------------------------------
// WP-G0 — the LoRA primitive + apply
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoraError {
    RankMismatch,
    DimMismatch,
    EmptyFactors,
}

/// A low-rank adapter `ΔW = alpha · (B · A)`, mirroring `core/learning::LoraAdapter`:
/// `matrix_a` is `[rank][dim_in]`, `matrix_b` is `[dim_out][rank]`, and applying it adds
/// the delta to a base weight matrix (`W' = W + ΔW`).
#[derive(Debug, Clone, PartialEq)]
pub struct LoraAdapter {
    pub zone: ZoneId,
    pub rank: usize,
    pub dim_in: usize,
    pub dim_out: usize,
    pub alpha: f32,
    pub matrix_a: Vec<Vec<f32>>, // [rank][dim_in]
    pub matrix_b: Vec<Vec<f32>>, // [dim_out][rank]
}

impl LoraAdapter {
    pub fn validate(&self) -> Result<(), LoraError> {
        if self.matrix_a.is_empty() || self.matrix_b.is_empty() {
            return Err(LoraError::EmptyFactors);
        }
        if self.matrix_a.len() != self.rank {
            return Err(LoraError::RankMismatch);
        }
        if self.matrix_b.iter().any(|r| r.len() != self.rank) {
            return Err(LoraError::RankMismatch);
        }
        if self.matrix_b.len() != self.dim_out {
            return Err(LoraError::DimMismatch);
        }
        if self.matrix_a.iter().any(|r| r.len() != self.dim_in) {
            return Err(LoraError::DimMismatch);
        }
        Ok(())
    }

    /// The dense delta `ΔW = alpha · (B · A)`, shape `[dim_out][dim_in]`.
    pub fn delta(&self) -> Vec<Vec<f32>> {
        let mut d = vec![vec![0.0f32; self.dim_in]; self.dim_out];
        for o in 0..self.dim_out {
            for k in 0..self.rank {
                let bok = self.matrix_b[o][k] * self.alpha;
                if bok == 0.0 {
                    continue;
                }
                for i in 0..self.dim_in {
                    d[o][i] += bok * self.matrix_a[k][i];
                }
            }
        }
        d
    }

    /// `W' = W + ΔW`.
    pub fn apply_matrix(&self, base: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let d = self.delta();
        base.iter()
            .enumerate()
            .map(|(o, row)| row.iter().enumerate().map(|(i, &w)| w + d[o][i]).collect())
            .collect()
    }

    /// WP-G5 — the registration payload, field-aligned to `LoRAFactory.LoRAAdapter`.
    pub fn registration(&self, base_model_digest: &str) -> LoraRegistration {
        LoraRegistration {
            zone: self.zone,
            rank: self.rank,
            dim_in: self.dim_in,
            dim_out: self.dim_out,
            alpha_q16: nat_types::Q16::from_f32(self.alpha).raw(),
            base_model_digest: base_model_digest.to_string(),
            lora_commitment: commit::lora_commitment(self),
        }
    }
}

/// The on-chain registration record this adapter lands as. Field-aligned to
/// `LoRAFactory.sol::LoRAAdapter` (`baseModelHash`, `rank`, `alpha`,
/// `adapterModelCommitment`) — the actual `createLoRA` call is Gate-4-deferred.
#[derive(Debug, Clone, PartialEq)]
pub struct LoraRegistration {
    pub zone: ZoneId,
    pub rank: usize,
    pub dim_in: usize,
    pub dim_out: usize,
    pub alpha_q16: i64,
    pub base_model_digest: String,
    pub lora_commitment: String,
}

/// argmax over each row of `W·h` for a stack of hidden vectors — the decision a readout
/// makes on a probe. Shared by the generator's verification and the benchmark.
pub fn decisions(weight: &[Vec<f32>], hiddens: &[Vec<f32>]) -> Vec<usize> {
    hiddens
        .iter()
        .map(|h| argmax(&linalg::matvec(weight, h)))
        .collect()
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best
}

// ---------------------------------------------------------------------------
// WP-G1 — weight-space conditioning
// ---------------------------------------------------------------------------

/// Encode a peer's weight-graph to the conditioning latent (permutation-invariant,
/// inherited from the GMN encoder). This is the *only* thing the generator reads about a
/// peer — weights in, adapter out.
pub fn condition(encoder: &GmnEncoder, peer: &WeightGraph) -> Vec<f32> {
    encoder.encode(peer)
}

// ---------------------------------------------------------------------------
// WP-G2 — the generator (hypernetwork)
// ---------------------------------------------------------------------------

/// A fixed rank-1 skill atom `D = u ⊗ v` (`u ∈ R^{dim_out}`, `v ∈ R^{dim_in}`). The
/// generator's dictionary of `K` atoms spans the adaptation subspace; meta-training learns
/// only the per-atom *gains* as a function of the peer latent.
#[derive(Debug, Clone, PartialEq)]
pub struct SkillAtom {
    pub u: Vec<f32>,
    pub v: Vec<f32>,
}

/// The weight-conditioned generator. Holds a fixed atom dictionary and a learned linear
/// gain map `latent → gains`; `generate` builds `ΔW = Σ_k gain_k · (u_k ⊗ v_k)` and
/// factors it as a rank-`K` LoRA. Meta-training ([`fit`](LoraGenerator::fit)) learns the
/// gain map from a distribution of peers — i.e. it learns to read weight-space.
#[derive(Debug, Clone)]
pub struct LoraGenerator {
    zone: ZoneId,
    atoms: Vec<SkillAtom>,
    latent_dim: usize,
    /// `[K][latent_dim+1]` (last column is bias). Zeroed until `fit`.
    gain: Vec<Vec<f32>>,
    alpha: f32,
}

impl LoraGenerator {
    pub fn new(zone: ZoneId, atoms: Vec<SkillAtom>, latent_dim: usize) -> Self {
        let k = atoms.len();
        LoraGenerator {
            zone,
            atoms,
            latent_dim,
            gain: vec![vec![0.0; latent_dim + 1]; k],
            alpha: 1.0,
        }
    }

    pub fn num_atoms(&self) -> usize {
        self.atoms.len()
    }

    /// The conditioning-latent dimension this generator expects.
    pub fn latent_dim(&self) -> usize {
        self.latent_dim
    }

    fn aug(latent: &[f32]) -> Vec<f32> {
        let mut z = latent.to_vec();
        z.push(1.0); // bias term
        z
    }

    /// Meta-train the gain map by ridge regression: given peer latents and the per-atom
    /// target gains they should produce, learn `latent → gains`. This is the "learning to
    /// generate adapters from weight-space" step, fit over a task distribution.
    pub fn fit(&mut self, latents: &[Vec<f32>], target_gains: &[Vec<f32>], lambda: f32) {
        assert_eq!(latents.len(), target_gains.len());
        let z: Vec<Vec<f32>> = latents.iter().map(|l| Self::aug(l)).collect();
        self.gain = linalg::ridge_fit(&z, target_gains, lambda); // [K][latent+1]
    }

    /// Predict the per-atom gains for a peer latent.
    pub fn predict_gains(&self, latent: &[f32]) -> Vec<f32> {
        let z = Self::aug(latent);
        self.gain
            .iter()
            .map(|grow| grow.iter().zip(&z).map(|(a, b)| a * b).sum())
            .collect()
    }

    /// Generate the LoRA adapter for a peer latent: `ΔW = Σ_k gain_k (u_k ⊗ v_k)`, factored
    /// as `B[dim_out][K]` (atom outputs) and `A[K][dim_in]` (gain-scaled atom inputs).
    pub fn generate(&self, latent: &[f32]) -> LoraAdapter {
        let gains = self.predict_gains(latent);
        let dim_out = self.atoms[0].u.len();
        let dim_in = self.atoms[0].v.len();
        let k = self.atoms.len();
        // B[o][k] = u_k[o]
        let matrix_b: Vec<Vec<f32>> = (0..dim_out)
            .map(|o| (0..k).map(|kk| self.atoms[kk].u[o]).collect())
            .collect();
        // A[k][i] = gain_k · v_k[i]
        let matrix_a: Vec<Vec<f32>> = (0..k)
            .map(|kk| self.atoms[kk].v.iter().map(|&vi| gains[kk] * vi).collect())
            .collect();
        LoraAdapter {
            zone: self.zone,
            rank: k,
            dim_in,
            dim_out,
            alpha: self.alpha,
            matrix_a,
            matrix_b,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank0_delta_is_identity_and_validate_catches_shape_bugs() {
        let atoms = vec![SkillAtom {
            u: vec![1.0, 0.0],
            v: vec![0.0, 0.0, 0.0],
        }];
        let gen = LoraGenerator::new(ZoneId::PF, atoms, 4);
        // gains zero (unfit) → delta is all-zero → apply is identity.
        let lora = gen.generate(&[0.0, 0.0, 0.0, 0.0]);
        lora.validate().expect("valid shapes");
        let base = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        assert_eq!(lora.apply_matrix(&base), base);

        let mut bad = lora.clone();
        bad.dim_out = 99;
        assert_eq!(bad.validate(), Err(LoraError::DimMismatch));
    }

    #[test]
    fn delta_is_low_rank_outer_product() {
        // one atom u⊗v with gain 2 → delta = 2·u vᵀ
        let atoms = vec![SkillAtom {
            u: vec![1.0, -1.0],
            v: vec![1.0, 2.0],
        }];
        let mut gen = LoraGenerator::new(ZoneId::PF, atoms, 1);
        // teach the gain map to output 2.0 for any latent: gain = [0, 2] (bias 2)
        gen.fit(&[vec![0.0], vec![1.0]], &[vec![2.0], vec![2.0]], 1e-6);
        let lora = gen.generate(&[0.5]);
        let d = lora.delta();
        // delta[o][i] = 2 * u[o] * v[i]
        assert!((d[0][0] - 2.0).abs() < 1e-3);
        assert!((d[0][1] - 4.0).abs() < 1e-3);
        assert!((d[1][0] + 2.0).abs() < 1e-3);
        assert!((d[1][1] + 4.0).abs() < 1e-3);
    }

    #[test]
    fn registration_payload_is_field_aligned() {
        let atoms = vec![SkillAtom {
            u: vec![1.0, 0.0],
            v: vec![1.0, 0.0],
        }];
        let mut gen = LoraGenerator::new(ZoneId::CX, atoms, 2);
        gen.fit(
            &[vec![0.0, 0.0], vec![1.0, 1.0]],
            &[vec![0.3], vec![0.7]],
            1e-6,
        );
        let lora = gen.generate(&[0.5, 0.5]);
        let reg = lora.registration("basemodel-digest-abc");
        assert_eq!(reg.zone, ZoneId::CX);
        assert_eq!(reg.rank, 1);
        assert_eq!(reg.base_model_digest, "basemodel-digest-abc");
        assert_eq!(reg.lora_commitment, commit::lora_commitment(&lora));
        assert_eq!(reg.alpha_q16, nat_types::Q16::from_f32(1.0).raw());
    }
}
