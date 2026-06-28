//! `nat-distill` — federated distillation on a shared probe set (WS-2 Layer A).
//!
//! The **deployable, architecture-agnostic** half of the weight-space program: peers
//! each run a *shared probe set*, and **only the probe-set logits cross the wire**.
//! A deterministic consensus aggregate of those logits (reusing `nat-aggregate`'s Q16
//! trimmed mean) is a teacher that distills capability into a student — moving
//! capability NAT↔transformer *today* without solving weight-space heterogeneity,
//! because nothing here reads weights, only outputs.
//!
//! Validation split (meta-plan §3): the **commitment + aggregation layer is
//! consensus-grade** — probe-logits are committed on the Q16 grid and aggregated
//! deterministically (frozen digests, reuses the AGG-S1 reduction). The **distillation
//! itself is research ML** — f32, gated by a McNemar significance test and a
//! no-regression battery, not a TLA+ invariant.
//!
//! Pipeline (WSE-S1 Layer A WPs):
//! - [`ProbeSet`] — a hash-pinned shared probe set from `nat-eval`'s battery (WP-A0).
//! - [`ProbeRunner`] / [`ProbeLogits`] — architecture-agnostic logit extraction +
//!   Q16 commitment (WP-A1); implemented for `nat_core::NatModel`.
//! - [`aggregate_probe_logits`] — deterministic consensus over peers (WP-A3).
//! - [`Student`] / [`distill`] — soft-label KL distillation toward the teacher (WP-A3).
//! - [`mcnemar_p_value`] / [`promotion_gate`] — the no-regression promotion gate (WP-A4).
//! - [`ZoneAdapter`] — per-zone distillation delta (WP-A5).

use nat_core::NatModel;
use nat_eval::battery::PromptBattery;
use nat_types::{Q16, ZoneId};
use sha2::{Digest, Sha256};

// ---------------------------------------------------------------------------
// WP-A0 — the hash-pinned shared probe set
// ---------------------------------------------------------------------------

/// An ordered, hash-pinned set of probe prompts. Every peer evaluates the identical
/// inputs in the identical order, so the logits are comparable and the consensus is
/// well-defined. The id is a digest of the ordered prompts — two peers derive the same
/// id iff they hold the same probe set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeSet {
    pub prompts: Vec<String>,
}

impl ProbeSet {
    /// Flatten a `nat-eval` battery into an ordered probe set (class order, then prompt
    /// order — both deterministic).
    pub fn from_battery(b: &PromptBattery) -> Self {
        let mut prompts = Vec::new();
        for class in &b.classes {
            for p in &class.prompts {
                prompts.push(p.clone());
            }
        }
        ProbeSet { prompts }
    }

    /// The default L0 probe set.
    pub fn default_l0() -> Self {
        Self::from_battery(&PromptBattery::default_l0())
    }

    pub fn len(&self) -> usize {
        self.prompts.len()
    }
    pub fn is_empty(&self) -> bool {
        self.prompts.is_empty()
    }

    /// `H(ordered prompts)` — the probe-set id all peers must agree on.
    pub fn id(&self) -> String {
        let mut h = Sha256::new();
        for p in &self.prompts {
            h.update((p.len() as u32).to_le_bytes());
            h.update(p.as_bytes());
        }
        hex(&h.finalize())
    }
}

// ---------------------------------------------------------------------------
// WP-A1 — architecture-agnostic logit extraction + Q16 commitment
// ---------------------------------------------------------------------------

/// Anything that produces a logit vector for a prompt. This is the seam that makes
/// distillation architecture-agnostic: a NAT 6-zone model and a standard transformer
/// both implement it, and the protocol only ever sees the logits.
pub trait ProbeRunner {
    fn run(&self, prompt: &str) -> Vec<f32>;
}

/// `nat_core::NatModel` is a probe runner via its forward pass.
impl ProbeRunner for NatModel {
    fn run(&self, prompt: &str) -> Vec<f32> {
        self.forward(prompt, None).output.logits.to_vec()
    }
}

/// One peer's logits over the shared probe set: `[n_prompts][d_out]`, plus the probe
/// id it was computed against. Architecture-opaque — just numbers.
#[derive(Debug, Clone, PartialEq)]
pub struct ProbeLogits {
    pub probe_set_id: String,
    pub logits: Vec<Vec<f32>>,
}

impl ProbeLogits {
    /// Run `runner` over every prompt of `probe`.
    pub fn extract(runner: &dyn ProbeRunner, probe: &ProbeSet) -> Self {
        let logits = probe.prompts.iter().map(|p| runner.run(p)).collect();
        ProbeLogits { probe_set_id: probe.id(), logits }
    }

    pub fn dims(&self) -> (usize, usize) {
        (self.logits.len(), self.logits.first().map(|r| r.len()).unwrap_or(0))
    }

    /// The **consensus-grade commitment**: quantize the logits onto the Q16 grid and
    /// hash the raw integers (no float on the committed path), so two peers that
    /// produced the same logits commit the same digest and an auditor can reconcile
    /// the aggregate input on-chain (Poseidon `0x0107` in deployment).
    pub fn digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(self.probe_set_id.as_bytes());
        for row in &self.logits {
            for &v in row {
                h.update(Q16::from_f32(v).raw().to_le_bytes());
            }
        }
        hex(&h.finalize())
    }
}

// ---------------------------------------------------------------------------
// WP-A3 (a) — deterministic consensus aggregation of peer logits
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DistillError {
    NoPeers,
    ProbeMismatch,
    ShapeMismatch,
    Aggregate(nat_aggregate::AggregateError),
}

/// Aggregate peers' probe-logits into one **consensus teacher**, coordinate-wise, via
/// `nat-aggregate`'s Q16 bucketed trimmed mean (Byzantine-robust + bit-reproducible —
/// the same reduction the spine proves in `GradientAggregation.tla`). Peers must agree
/// on the probe id and the shape. The teacher is returned as `ProbeLogits` (dequantized
/// for the f32 distillation step; the *aggregation* itself was integer/deterministic).
pub fn aggregate_probe_logits(
    peers: &[ProbeLogits],
    trim: usize,
    buckets: usize,
    seed: &[u8],
) -> Result<ProbeLogits, DistillError> {
    let first = peers.first().ok_or(DistillError::NoPeers)?;
    let (n, d) = first.dims();
    for p in peers {
        if p.probe_set_id != first.probe_set_id {
            return Err(DistillError::ProbeMismatch);
        }
        if p.dims() != (n, d) {
            return Err(DistillError::ShapeMismatch);
        }
    }

    // Flatten each peer to one Q16 pseudo-gradient and aggregate coordinate-wise.
    let grads: Vec<nat_aggregate::PseudoGradient> = peers
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let coords =
                p.logits.iter().flatten().map(|&v| Q16::from_f32(v)).collect::<Vec<_>>();
            nat_aggregate::PseudoGradient::new(format!("peer{i}"), coords)
        })
        .collect();

    let agg = nat_aggregate::aggregate(&grads, trim, buckets, seed)
        .map_err(DistillError::Aggregate)?;

    // Reshape the flat aggregate back to [n][d].
    let mut logits = Vec::with_capacity(n);
    for r in 0..n {
        logits.push(agg.aggregate[r * d..(r + 1) * d].iter().map(|q| q.to_f32()).collect());
    }
    Ok(ProbeLogits { probe_set_id: first.probe_set_id.clone(), logits })
}

// ---------------------------------------------------------------------------
// WP-A3 (b) — soft-label distillation into a student
// ---------------------------------------------------------------------------

/// A student over the probe set: a per-prompt logit table it adjusts toward a teacher.
/// (A logit-table student is the architecture-agnostic distillation target; a real
/// deployment backs it with model weights, but the soft-label objective is identical.)
#[derive(Debug, Clone, PartialEq)]
pub struct Student {
    pub logits: Vec<Vec<f32>>,
}

impl Student {
    pub fn from_logits(p: &ProbeLogits) -> Self {
        Student { logits: p.logits.clone() }
    }

    /// argmax prediction per prompt.
    pub fn predictions(&self) -> Vec<usize> {
        self.logits.iter().map(|r| argmax(r)).collect()
    }

    /// Fraction of prompts whose argmax matches `labels`.
    pub fn accuracy(&self, labels: &[usize]) -> f32 {
        let correct = self.predictions().iter().zip(labels).filter(|(p, l)| *p == *l).count();
        correct as f32 / labels.len().max(1) as f32
    }
}

/// One soft-label distillation step: move the student's logits toward the teacher by
/// gradient descent on the temperature-softened KL divergence. The gradient of
/// `KL(softmax(teacher/T) || softmax(student/T))` w.r.t. the student logits is
/// `softmax(student/T) - softmax(teacher/T)`, so this provably reduces the divergence.
pub fn distill_step(student: &mut Student, teacher: &ProbeLogits, temperature: f32, lr: f32) {
    for (s_row, t_row) in student.logits.iter_mut().zip(&teacher.logits) {
        let ps = softmax(s_row, temperature);
        let pt = softmax(t_row, temperature);
        for k in 0..s_row.len() {
            s_row[k] -= lr * (ps[k] - pt[k]);
        }
    }
}

/// Run `steps` distillation steps; returns the mean per-prompt KL divergence
/// (teacher‖student) before and after, so callers can see the transfer happened.
pub fn distill(
    student: &mut Student,
    teacher: &ProbeLogits,
    temperature: f32,
    lr: f32,
    steps: usize,
) -> (f32, f32) {
    let before = mean_kl(student, teacher, temperature);
    for _ in 0..steps {
        distill_step(student, teacher, temperature, lr);
    }
    let after = mean_kl(student, teacher, temperature);
    (before, after)
}

// ---------------------------------------------------------------------------
// WP-A4 — the McNemar promotion gate (no-regression)
// ---------------------------------------------------------------------------

/// Exact two-sided McNemar test on paired correct/incorrect outcomes. `b` = #(model A
/// right, B wrong), `c` = #(A wrong, B right). Returns the exact binomial p-value for
/// the discordant pairs (no chi-square approximation — robust for the small probe set).
pub fn mcnemar_p_value(b: usize, c: usize) -> f64 {
    let n = b + c;
    if n == 0 {
        return 1.0; // no discordant pairs → no evidence of difference
    }
    let k = b.min(c);
    // two-sided exact: 2 * sum_{i=0}^{k} C(n,i) * 0.5^n, capped at 1.
    let mut tail = 0.0f64;
    for i in 0..=k {
        tail += binom(n, i) * 0.5f64.powi(n as i32);
    }
    (2.0 * tail).min(1.0)
}

/// The promotion decision: the distilled student is promoted iff it improves over the
/// baseline on the eval battery with **McNemar p < alpha** AND does **not regress** the
/// held-out battery (a quality ratchet — a distilled model never loses base capability).
pub fn promotion_gate(
    baseline_preds: &[usize],
    candidate_preds: &[usize],
    labels: &[usize],
    heldout_baseline_acc: f32,
    heldout_candidate_acc: f32,
    alpha: f64,
) -> PromotionDecision {
    // discordant pairs on the eval battery
    let mut b = 0; // baseline right, candidate wrong
    let mut c = 0; // baseline wrong, candidate right
    for i in 0..labels.len() {
        let base_ok = baseline_preds[i] == labels[i];
        let cand_ok = candidate_preds[i] == labels[i];
        if base_ok && !cand_ok {
            b += 1;
        } else if !base_ok && cand_ok {
            c += 1;
        }
    }
    let p = mcnemar_p_value(b, c);
    let improved = c > b;
    let significant = p < alpha;
    let no_regression = heldout_candidate_acc + 1e-6 >= heldout_baseline_acc;
    PromotionDecision {
        promote: improved && significant && no_regression,
        p_value: p,
        improved,
        no_regression,
        b,
        c,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PromotionDecision {
    pub promote: bool,
    pub p_value: f64,
    pub improved: bool,
    pub no_regression: bool,
    pub b: usize,
    pub c: usize,
}

// ---------------------------------------------------------------------------
// WP-A5 — per-zone distillation delta (the NAT unique lever)
// ---------------------------------------------------------------------------

/// A distillation delta scoped to a single learned NAT zone — the per-zone lever a
/// dense model cannot offer (paper §7.1). The delta is the student's logit correction
/// for that zone's output slice, tagged for registration as a per-zone LoRA.
#[derive(Debug, Clone, PartialEq)]
pub struct ZoneAdapter {
    pub zone: ZoneId,
    pub delta: Vec<f32>,
}

impl ZoneAdapter {
    /// Build a per-zone adapter from the mean logit correction (post − pre) over the
    /// probe set, for a learned zone. Returns `None` for the non-learned `MX` harness.
    pub fn from_distillation(zone: ZoneId, pre: &Student, post: &Student) -> Option<Self> {
        if !zone.is_learned() {
            return None;
        }
        let d = post.logits.first().map(|r| r.len()).unwrap_or(0);
        let mut delta = vec![0.0f32; d];
        for (post_row, pre_row) in post.logits.iter().zip(&pre.logits) {
            for k in 0..d {
                delta[k] += post_row[k] - pre_row[k];
            }
        }
        let n = post.logits.len().max(1) as f32;
        for v in &mut delta {
            *v /= n;
        }
        Some(ZoneAdapter { zone, delta })
    }
}

// ---------------------------------------------------------------------------
// math helpers (research/f32 layer)
// ---------------------------------------------------------------------------

fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best
}

fn softmax(v: &[f32], temperature: f32) -> Vec<f32> {
    let t = temperature.max(1e-6);
    let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = v.iter().map(|&x| ((x - m) / t).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

fn mean_kl(student: &Student, teacher: &ProbeLogits, temperature: f32) -> f32 {
    let mut total = 0.0;
    for (s_row, t_row) in student.logits.iter().zip(&teacher.logits) {
        let ps = softmax(s_row, temperature);
        let pt = softmax(t_row, temperature);
        // KL(teacher || student) = sum pt * ln(pt/ps)
        for k in 0..s_row.len() {
            if pt[k] > 1e-12 {
                total += pt[k] * (pt[k] / ps[k].max(1e-12)).ln();
            }
        }
    }
    total / student.logits.len().max(1) as f32
}

/// Binomial coefficient C(n, k) as f64 (n is small — the probe set size).
fn binom(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    let mut r = 1.0f64;
    for i in 0..k {
        r = r * (n - i) as f64 / (i + 1) as f64;
    }
    r
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

    // A synthetic "expert" peer: confidently correct on a fixed label per prompt (a
    // stand-in for a model that has the capability the student lacks). Demonstrates the
    // capability transfer end-to-end without the SCALE-S1 training track.
    struct Expert {
        labels: Vec<usize>,
        d: usize,
        noise: f32,
        idx: std::cell::Cell<usize>,
    }
    impl ProbeRunner for Expert {
        fn run(&self, _prompt: &str) -> Vec<f32> {
            let i = self.idx.get();
            self.idx.set(i + 1);
            let mut v = vec![0.0f32; self.d];
            // small deterministic per-(prompt,dim) jitter so peers are not identical
            for (k, vk) in v.iter_mut().enumerate() {
                *vk = self.noise * (((i * 7 + k * 13) % 5) as f32 - 2.0);
            }
            v[self.labels[i % self.labels.len()]] += 5.0; // confident on the true class
            v
        }
    }

    fn expert(labels: &[usize], d: usize, noise: f32) -> Expert {
        Expert { labels: labels.to_vec(), d, noise, idx: std::cell::Cell::new(0) }
    }

    // -- WP-A0 ------------------------------------------------------------

    #[test]
    fn probe_set_id_is_deterministic_and_order_pinned() {
        let p = ProbeSet::default_l0();
        assert!(!p.is_empty());
        assert_eq!(p.id(), ProbeSet::default_l0().id());
        // Reordering changes the id (the probe set is order-pinned).
        let mut rev = p.clone();
        rev.prompts.reverse();
        assert_ne!(p.id(), rev.id());
    }

    // -- WP-A1: real NatModel wiring + Q16 commitment ---------------------

    #[test]
    fn nat_model_is_a_probe_runner_and_commits_deterministically() {
        let probe = ProbeSet::default_l0();
        let model = NatModel::l0();
        let pl1 = ProbeLogits::extract(&model, &probe);
        let pl2 = ProbeLogits::extract(&model, &probe);
        let (n, d) = pl1.dims();
        assert_eq!(n, probe.len());
        assert_eq!(d, 8); // D_OUT
        // The Q16 commitment is deterministic across extractions.
        assert_eq!(pl1.digest(), pl2.digest());
    }

    // -- WP-A3: deterministic aggregation ---------------------------------

    #[test]
    fn aggregation_is_deterministic_and_order_independent() {
        let probe = ProbeSet::default_l0();
        let labels: Vec<usize> = (0..probe.len()).map(|i| i % 4).collect();
        let peers: Vec<ProbeLogits> = (0..5)
            .map(|s| ProbeLogits::extract(&expert(&labels, 8, 0.1 * s as f32), &probe))
            .collect();
        let a = aggregate_probe_logits(&peers, 1, 64, b"seed").unwrap();
        let b = aggregate_probe_logits(&peers, 1, 64, b"seed").unwrap();
        assert_eq!(a.digest(), b.digest());
        // reversing peer order yields the same consensus (a function of the set)
        let mut rev = peers.clone();
        rev.reverse();
        assert_eq!(a.digest(), aggregate_probe_logits(&rev, 1, 64, b"seed").unwrap().digest());
    }

    #[test]
    fn aggregation_rejects_probe_mismatch() {
        let p1 = ProbeLogits { probe_set_id: "x".into(), logits: vec![vec![1.0, 2.0]] };
        let p2 = ProbeLogits { probe_set_id: "y".into(), logits: vec![vec![1.0, 2.0]] };
        assert_eq!(aggregate_probe_logits(&[p1, p2], 0, 4, b"s"), Err(DistillError::ProbeMismatch));
    }

    // -- WP-A3: distillation transfers capability -------------------------

    #[test]
    fn distillation_moves_student_toward_teacher_and_raises_accuracy() {
        let probe = ProbeSet::default_l0();
        let labels: Vec<usize> = (0..probe.len()).map(|i| i % 4).collect();

        // Experts know the labels; their consensus is an accurate teacher.
        let peers: Vec<ProbeLogits> = (0..5)
            .map(|s| ProbeLogits::extract(&expert(&labels, 8, 0.2 * s as f32), &probe))
            .collect();
        let teacher = aggregate_probe_logits(&peers, 1, 64, b"seed").unwrap();

        // A weak student: confidently wrong (predicts class 0 everywhere).
        let weak = ProbeLogits {
            probe_set_id: probe.id(),
            logits: probe.prompts.iter().map(|_| {
                let mut v = vec![0.0f32; 8];
                v[0] = 5.0;
                v
            }).collect(),
        };
        let mut student = Student::from_logits(&weak);
        let acc_before = student.accuracy(&labels);
        let (kl_before, kl_after) = distill(&mut student, &teacher, 2.0, 0.5, 300);
        let acc_after = student.accuracy(&labels);

        assert!(kl_after < kl_before, "distillation reduced KL: {kl_before} -> {kl_after}");
        assert!(acc_after > acc_before, "accuracy rose: {acc_before} -> {acc_after}");
        assert!(acc_after >= 0.9, "student learned the teacher's capability: {acc_after}");
    }

    // -- WP-A4: McNemar gate ----------------------------------------------

    #[test]
    fn mcnemar_significant_improvement_promotes() {
        // 12 prompts the candidate fixes, 1 it breaks -> strongly significant.
        let labels: Vec<usize> = vec![1; 13];
        let baseline: Vec<usize> = vec![0; 13]; // all wrong
        let mut candidate = vec![1; 13]; // fixes 12…
        candidate[0] = 2; // …breaks 1 (wrong)
        let d = promotion_gate(&baseline, &candidate, &labels, 0.8, 0.85, 0.05);
        assert!(d.improved && d.no_regression);
        assert!(d.p_value < 0.05, "p={}", d.p_value);
        assert!(d.promote);
    }

    #[test]
    fn mcnemar_gate_blocks_regression_even_if_significant() {
        let labels: Vec<usize> = vec![1; 13];
        let baseline: Vec<usize> = vec![0; 13];
        let candidate: Vec<usize> = vec![1; 13]; // significant improvement on eval…
        // …but held-out accuracy regressed (0.70 < 0.80) -> blocked by the ratchet.
        let d = promotion_gate(&baseline, &candidate, &labels, 0.80, 0.70, 0.05);
        assert!(d.improved && d.p_value < 0.05);
        assert!(!d.no_regression);
        assert!(!d.promote, "regression must block promotion");
    }

    #[test]
    fn mcnemar_no_improvement_does_not_promote() {
        let labels: Vec<usize> = vec![1; 10];
        let same: Vec<usize> = vec![1; 10];
        let d = promotion_gate(&same, &same, &labels, 0.9, 0.9, 0.05);
        assert!(!d.improved);
        assert!(!d.promote);
    }

    // -- WP-A5: per-zone adapter ------------------------------------------

    #[test]
    fn per_zone_adapter_built_for_learned_zone_rejected_for_mx() {
        let probe = ProbeSet::default_l0();
        let base = Student::from_logits(&ProbeLogits {
            probe_set_id: probe.id(),
            logits: probe.prompts.iter().map(|_| vec![0.0f32; 8]).collect(),
        });
        let mut post = base.clone();
        for row in &mut post.logits {
            row[3] += 1.0; // a correction concentrated on output 3
        }
        let za = ZoneAdapter::from_distillation(ZoneId::PF, &base, &post).unwrap();
        assert_eq!(za.zone, ZoneId::PF);
        assert!((za.delta[3] - 1.0).abs() < 1e-6);
        assert!(ZoneAdapter::from_distillation(ZoneId::MX, &base, &post).is_none());
    }

    // -- consensus-grade discipline: golden ratchet + determinism sweep ---

    /// A fixed synthetic peer set whose aggregate commitment is frozen. This is the
    /// golden-bytes ratchet for the *committed* layer (per the meta-plan validation
    /// split): the f32 distillation is research ML, but the Q16 commitment + the
    /// `nat-aggregate` reduction are consensus-grade and must not drift silently.
    fn golden_peers() -> Vec<ProbeLogits> {
        let labels: Vec<usize> = (0..12).map(|i| i % 4).collect();
        (0..5)
            .map(|s| {
                let logits = (0..12)
                    .map(|p| {
                        let mut v = vec![0.0f32; 8];
                        for (k, vk) in v.iter_mut().enumerate() {
                            // deterministic per-(peer,prompt,dim) signal — no RNG
                            *vk = 0.1 * (((p * 7 + k * 13 + s * 3) % 5) as i32 - 2) as f32;
                        }
                        v[labels[p]] += 5.0;
                        v
                    })
                    .collect();
                ProbeLogits { probe_set_id: "golden-probe-v1".into(), logits }
            })
            .collect()
    }

    #[test]
    fn aggregate_commitment_is_frozen() {
        let agg = aggregate_probe_logits(&golden_peers(), 1, 64, b"agg-golden-v1").unwrap();
        // FROZEN — regenerate intentionally only when the committed reduction changes.
        assert_eq!(
            agg.digest(),
            "f683ad9751701e08f231061a03c76455e9de41b876219f089b33ff78e2feaf6c"
        );
    }

    #[test]
    fn aggregation_determinism_sweep() {
        // The consensus aggregate is a pure function of (peers, trim, buckets, seed).
        // Sweep 2000 distinct seeds; each aggregation must be bit-identical when
        // recomputed — the no-float / no-hashmap-iteration guarantee, exercised across
        // a wide space of bucket assignments. (Order-dependence is intentional: the
        // bucket key is the peer node-id, so the aggregate is a function of the
        // label→value assignment, not the bare value multiset.)
        let peers = golden_peers();
        for round in 0..2000u32 {
            let seed = round.to_le_bytes();
            match aggregate_probe_logits(&peers, 1, 64, &seed) {
                Ok(first) => {
                    let again = aggregate_probe_logits(&peers, 1, 64, &seed).unwrap();
                    assert_eq!(first.digest(), again.digest(), "non-determinism at round {round}");
                }
                // Fail-closed: a seed whose buckets collide below the trim budget is
                // rejected (deterministically) rather than aggregated unsafely.
                Err(DistillError::Aggregate(nat_aggregate::AggregateError::TrimBudgetTooLarge {
                    ..
                })) => continue,
                Err(e) => panic!("unexpected aggregate error at round {round}: {e:?}"),
            }
        }
    }
}
