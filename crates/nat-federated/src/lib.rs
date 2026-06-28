//! `nat-federated` — Gate-4 federated proof.
//!
//! Nodes train toward the shared model and submit **signed contributions**. A
//! gather verifies every signature *before* anything enters the aggregate, sums
//! the reward weights deterministically (Q16.16), and merges the per-node
//! provenance trace-hashes into one auditable hash. That merged hash is then
//! committed on-chain ([`ChainCommit`]) and each accepted node is settled through
//! compute-pool ([`Settlement`]).
//!
//! The four Gate-4 exit criteria (`gates.yaml` gate4) map here:
//!
//! - **g4-gather** — [`gather_and_aggregate`] verifies each signature and drops
//!   invalid contributions before composition. A forged or tampered contribution
//!   can never reach the aggregate or the on-chain commit.
//! - **g4-tolerance** (H-05b) — [`within_tolerance`] checks a federated aggregate
//!   against a centralized baseline within a slack.
//! - **g4-onchain** — the [`ChainCommit`] seam; the real impl anchors `merged_hash`
//!   on citrate-chain and an auditor replays it.
//! - **g4-settlement** — the [`Settlement`] seam; the real impl pays out through
//!   citrate-compute-pool (`compute × quality → reward weight`).
//!
//! ## Honest scope
//!
//! This crate is the **pure, testable core** — signing is pluggable ([`Signer`]/
//! [`Verifier`]) and ships with a *toy keyed-hash* signer for tests. Production
//! swaps in the operator signer (ed25519 / AWS-KMS, already built in the gateway
//! and custody-signed-off). The real multi-node network, the on-chain commit, and
//! the compute-pool payout are the deployment phase (sprint NAT-S3 WP-F3..F6).

use nat_train::StepContribution;
use nat_types::Q16;
use sha2::{Digest, Sha256};

pub mod seam;

/// Domain-separation tag mixed into every signed message, so a NAT federated
/// signature can never be replayed as a signature for any other protocol.
const DOMAIN: &[u8] = b"nat-fed-v1";

// ---------------------------------------------------------------------------
// Signing seam
// ---------------------------------------------------------------------------

/// Signs the canonical message for a node. The production impl is the gateway
/// operator signer (ed25519 / AWS-KMS); [`ToyKeyedSigner`] is the test stand-in.
pub trait Signer {
    /// The node identity this signer speaks for.
    fn node_id(&self) -> &str;
    /// Produce a signature over `msg`.
    fn sign(&self, msg: &[u8]) -> Vec<u8>;
}

/// Verifies a signature attributed to a node. The verifier owns the trusted
/// node→key binding (a roster on the real path); the gather never trusts a
/// `node_id` it cannot verify.
pub trait Verifier {
    /// `true` iff `sig` is a valid signature by `node_id` over `msg`.
    fn verify(&self, node_id: &str, msg: &[u8], sig: &[u8]) -> bool;
}

// ---------------------------------------------------------------------------
// Signed contribution
// ---------------------------------------------------------------------------

/// A node's training-step contribution, bound to the corpus it trained on and the
/// provenance trace it produced, and signed. This is the unit a node submits and
/// the gather verifies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignedContribution {
    /// The submitting node's identity (must be on the verifier's roster).
    pub node_id: String,
    /// The settlement accounting (compute, quality, tokens, step provenance hash).
    pub contribution: StepContribution,
    /// Hash of the corpus manifest the node trained on (what data, under what
    /// license) — binds the reward claim to an auditable shard.
    pub manifest_hash: String,
    /// The merged provenance trace-hash for this node's work this round.
    pub trace_hash: String,
    /// The signature over [`SignedContribution::message`].
    pub signature: Vec<u8>,
}

impl SignedContribution {
    /// The canonical, deterministic bytes a node signs. Integers are written as
    /// fixed-width little-endian and strings are length-prefixed, so two encoders
    /// can never disagree on the boundary between fields (no delimiter ambiguity).
    pub fn signing_message(
        node_id: &str,
        contribution: &StepContribution,
        manifest_hash: &str,
        trace_hash: &str,
    ) -> Vec<u8> {
        let mut m = Vec::new();
        m.extend_from_slice(DOMAIN);
        put_str(&mut m, node_id);
        m.extend_from_slice(&contribution.compute_metered.raw().to_le_bytes());
        m.extend_from_slice(&contribution.data_quality.raw().to_le_bytes());
        m.extend_from_slice(&contribution.tokens.to_le_bytes());
        put_str(&mut m, &contribution.provenance_hash);
        put_str(&mut m, manifest_hash);
        put_str(&mut m, trace_hash);
        m
    }

    /// Build and sign a contribution with `signer` (sets `node_id` from the signer).
    pub fn create(
        signer: &dyn Signer,
        contribution: StepContribution,
        manifest_hash: impl Into<String>,
        trace_hash: impl Into<String>,
    ) -> Self {
        let node_id = signer.node_id().to_string();
        let manifest_hash = manifest_hash.into();
        let trace_hash = trace_hash.into();
        let msg = Self::signing_message(&node_id, &contribution, &manifest_hash, &trace_hash);
        let signature = signer.sign(&msg);
        Self {
            node_id,
            contribution,
            manifest_hash,
            trace_hash,
            signature,
        }
    }

    /// Recompute the canonical signed message from this contribution's own fields.
    /// The verifier hashes *this*, not the wire bytes, so a tampered field changes
    /// the message and fails verification.
    pub fn message(&self) -> Vec<u8> {
        Self::signing_message(
            &self.node_id,
            &self.contribution,
            &self.manifest_hash,
            &self.trace_hash,
        )
    }
}

// ---------------------------------------------------------------------------
// Gather + aggregate (g4-gather)
// ---------------------------------------------------------------------------

/// One accepted node and the reward weight its contribution earned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcceptedContribution {
    pub node_id: String,
    pub reward_weight: Q16,
    pub trace_hash: String,
}

/// The outcome of a gather round: who was accepted (with their weights), who was
/// rejected (and why), the deterministic total reward weight, and the merged
/// trace-hash to commit on-chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatherResult {
    pub accepted: Vec<AcceptedContribution>,
    pub rejected: Vec<Rejection>,
    pub total_reward_weight: Q16,
    /// `H(sorted accepted trace_hashes)`. Deterministic in the *set* of accepted
    /// contributions (order-independent), which is what lets an auditor replay it.
    pub merged_hash: String,
}

/// A dropped contribution and the reason it never entered the aggregate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rejection {
    pub node_id: String,
    pub reason: RejectReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectReason {
    /// Signature did not verify against the roster — forged, tampered, or unknown node.
    BadSignature,
}

/// Verify every signature, drop the invalid ones, then aggregate. **Verification
/// happens before composition** (g4-gather): a contribution whose signature does
/// not verify contributes nothing to `total_reward_weight` and is absent from
/// `merged_hash`.
///
/// The total is a Q16.16 sum (deterministic, federated-reconcilable) and the
/// merged hash is over the *sorted* accepted trace-hashes, so reordering inputs
/// yields the identical result.
pub fn gather_and_aggregate(
    contribs: &[SignedContribution],
    verifier: &dyn Verifier,
) -> GatherResult {
    let mut accepted = Vec::new();
    let mut rejected = Vec::new();

    for c in contribs {
        if verifier.verify(&c.node_id, &c.message(), &c.signature) {
            accepted.push(AcceptedContribution {
                node_id: c.node_id.clone(),
                reward_weight: c.contribution.reward_weight(),
                trace_hash: c.trace_hash.clone(),
            });
        } else {
            rejected.push(Rejection {
                node_id: c.node_id.clone(),
                reason: RejectReason::BadSignature,
            });
        }
    }

    let total_reward_weight = accepted.iter().map(|a| a.reward_weight).sum();
    let merged_hash = merge_trace_hashes(accepted.iter().map(|a| a.trace_hash.as_str()));

    GatherResult {
        accepted,
        rejected,
        total_reward_weight,
        merged_hash,
    }
}

/// `H(sorted trace_hashes joined by '\n')`. Sorting makes the merge a function of
/// the accepted *set*, not the arrival order — the determinism an on-chain replay
/// depends on (MergeDeterminism `DeterminismTheorem`).
pub fn merge_trace_hashes<'a>(hashes: impl Iterator<Item = &'a str>) -> String {
    let mut v: Vec<&str> = hashes.collect();
    v.sort_unstable();
    let mut h = Sha256::new();
    for (i, t) in v.iter().enumerate() {
        if i > 0 {
            h.update(b"\n");
        }
        h.update(t.as_bytes());
    }
    hex(&h.finalize())
}

// ---------------------------------------------------------------------------
// Tolerance (g4-tolerance / H-05b)
// ---------------------------------------------------------------------------

/// H-05b: does a federated aggregate match the centralized baseline within a
/// relative slack? Returns `true` iff `|federated - centralized| <= tol * |centralized|`.
/// The `tol` fraction is quantized onto the Q16 grid so the whole comparison stays
/// on integers (federated-reconcilable).
pub fn within_tolerance(federated: Q16, centralized: Q16, tol: f32) -> bool {
    let diff = (federated.raw() - centralized.raw()).unsigned_abs() as u128;
    let mag = centralized.raw().unsigned_abs() as u128;
    let tol_raw = Q16::from_f32(tol).raw().unsigned_abs() as u128;
    // bound = mag * tol  (Q16 multiply: (mag * tol_raw) >> 16)
    let bound = (mag * tol_raw) >> 16;
    diff <= bound
}

// ---------------------------------------------------------------------------
// On-chain commit + settlement seams (g4-onchain / g4-settlement)
// ---------------------------------------------------------------------------

/// Error from the deployment seams (chain / settlement). The pure core never
/// errors; only the real impls (network, chain, pool) can.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FederationError(pub String);

impl std::fmt::Display for FederationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "federation error: {}", self.0)
    }
}
impl std::error::Error for FederationError {}

/// Commits the merged trace-hash on-chain (g4-onchain). The real impl writes to
/// citrate-chain (the agent-runtime recorder already anchors hashes); an auditor
/// replays the gather and checks the committed hash reproduces.
pub trait ChainCommit {
    /// An opaque receipt (e.g. a tx hash) the caller can present for audit.
    fn commit_trace_hash(&self, merged_hash: &str) -> Result<String, FederationError>;
}

/// Settles an accepted contribution through compute-pool (g4-settlement). NAT
/// proposes the reward *weight*; compute-pool converts weight → payout under its
/// tokenomics. The real impl calls citrate-compute-pool.
pub trait Settlement {
    fn settle(&self, node_id: &str, reward_weight: Q16) -> Result<(), FederationError>;
}

/// Drive a verified gather all the way through commit + settlement. Commits the
/// merged hash once, then settles each accepted node. Stops at the first seam
/// error (the deployment impls decide retry/idempotency).
pub fn finalize_round(
    result: &GatherResult,
    chain: &dyn ChainCommit,
    settlement: &dyn Settlement,
) -> Result<String, FederationError> {
    let receipt = chain.commit_trace_hash(&result.merged_hash)?;
    for a in &result.accepted {
        settlement.settle(&a.node_id, a.reward_weight)?;
    }
    Ok(receipt)
}

// ---------------------------------------------------------------------------
// Toy keyed-hash signer (TEST STAND-IN — not for production)
// ---------------------------------------------------------------------------

/// A deterministic keyed-hash signer for tests and the L0 simulated gather:
/// `sig = H(key || msg || key)`. **Not** a real signature scheme (no public-key
/// separation) — production uses the operator ed25519 / AWS-KMS signer. It is
/// enough to exercise the verify-before-compose path: tamper any field and the
/// recomputed message no longer matches the signature.
pub struct ToyKeyedSigner {
    node_id: String,
    key: Vec<u8>,
}

impl ToyKeyedSigner {
    pub fn new(node_id: impl Into<String>, key: impl Into<Vec<u8>>) -> Self {
        Self {
            node_id: node_id.into(),
            key: key.into(),
        }
    }
}

fn keyed_hash(key: &[u8], msg: &[u8]) -> Vec<u8> {
    let mut h = Sha256::new();
    h.update(key);
    h.update(msg);
    h.update(key);
    h.finalize().to_vec()
}

impl Signer for ToyKeyedSigner {
    fn node_id(&self) -> &str {
        &self.node_id
    }
    fn sign(&self, msg: &[u8]) -> Vec<u8> {
        keyed_hash(&self.key, msg)
    }
}

/// The verifier counterpart to [`ToyKeyedSigner`]: holds the trusted node→key
/// roster and recomputes the keyed hash. An unknown node fails closed.
#[derive(Default)]
pub struct ToyRosterVerifier {
    roster: std::collections::BTreeMap<String, Vec<u8>>,
}

impl ToyRosterVerifier {
    pub fn new() -> Self {
        Self::default()
    }
    /// Register a node's key on the roster (chainable).
    pub fn with_node(mut self, node_id: impl Into<String>, key: impl Into<Vec<u8>>) -> Self {
        self.roster.insert(node_id.into(), key.into());
        self
    }
}

impl Verifier for ToyRosterVerifier {
    fn verify(&self, node_id: &str, msg: &[u8], sig: &[u8]) -> bool {
        match self.roster.get(node_id) {
            Some(key) => {
                let expect = keyed_hash(key, msg);
                // constant-time-ish: lengths equal then byte compare (toy path).
                expect.len() == sig.len() && expect.iter().zip(sig).all(|(a, b)| a == b)
            }
            None => false, // unknown node: fail closed
        }
    }
}

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

/// Length-prefix a string into a byte buffer (u32 LE length + bytes), so field
/// boundaries are unambiguous in the canonical message.
fn put_str(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

/// Lowercase hex of a byte slice (no external hex dep on the deterministic path).
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

    fn contrib(compute: f32, quality: f32, prov: &str) -> StepContribution {
        StepContribution {
            compute_metered: Q16::from_f32(compute),
            data_quality: Q16::from_f32(quality),
            tokens: 1024,
            provenance_hash: prov.into(),
        }
    }

    fn signer(id: &str) -> ToyKeyedSigner {
        ToyKeyedSigner::new(id, format!("key-{id}").into_bytes())
    }

    fn roster(ids: &[&str]) -> ToyRosterVerifier {
        let mut v = ToyRosterVerifier::new();
        for id in ids {
            v = v.with_node(*id, format!("key-{id}").into_bytes());
        }
        v
    }

    #[test]
    fn valid_contributions_are_accepted_and_aggregated() {
        let v = roster(&["a", "b"]);
        let cs = vec![
            SignedContribution::create(&signer("a"), contrib(4.0, 0.5, "pa"), "ma", "ta"),
            SignedContribution::create(&signer("b"), contrib(2.0, 1.0, "pb"), "mb", "tb"),
        ];
        let r = gather_and_aggregate(&cs, &v);
        assert_eq!(r.accepted.len(), 2);
        assert!(r.rejected.is_empty());
        // 4*0.5 + 2*1.0 = 4.0
        assert_eq!(r.total_reward_weight, Q16::from_f32(4.0));
        assert_eq!(r.merged_hash.len(), 64);
    }

    #[test]
    fn forged_signature_is_rejected_before_aggregation() {
        // 'mallory' is on the roster but signs with the wrong key.
        let v = roster(&["a", "mallory"]);
        let wrong_key = ToyKeyedSigner::new("mallory", b"not-mallorys-key".to_vec());
        let cs = vec![
            SignedContribution::create(&signer("a"), contrib(4.0, 0.5, "pa"), "ma", "ta"),
            SignedContribution::create(&wrong_key, contrib(1000.0, 1.0, "pm"), "mm", "tm"),
        ];
        let r = gather_and_aggregate(&cs, &v);
        assert_eq!(r.accepted.len(), 1);
        assert_eq!(r.rejected.len(), 1);
        assert_eq!(r.rejected[0].node_id, "mallory");
        // The forged 1000.0 contribution never entered the total.
        assert_eq!(r.total_reward_weight, Q16::from_f32(2.0));
    }

    #[test]
    fn tampering_a_field_after_signing_fails_verification() {
        let v = roster(&["a"]);
        let mut c = SignedContribution::create(&signer("a"), contrib(4.0, 0.5, "pa"), "ma", "ta");
        // Inflate the metered compute after signing — the recomputed message no
        // longer matches the signature.
        c.contribution.compute_metered = Q16::from_f32(9000.0);
        let r = gather_and_aggregate(std::slice::from_ref(&c), &v);
        assert!(r.accepted.is_empty());
        assert_eq!(r.rejected[0].reason, RejectReason::BadSignature);
    }

    #[test]
    fn unknown_node_fails_closed() {
        let v = roster(&["a"]); // 'z' is not on the roster
        let cs = vec![SignedContribution::create(
            &signer("z"),
            contrib(4.0, 0.5, "pz"),
            "mz",
            "tz",
        )];
        let r = gather_and_aggregate(&cs, &v);
        assert!(r.accepted.is_empty());
        assert_eq!(r.rejected.len(), 1);
    }

    #[test]
    fn merged_hash_is_order_independent() {
        let v = roster(&["a", "b"]);
        let a = SignedContribution::create(&signer("a"), contrib(1.0, 1.0, "pa"), "ma", "ta");
        let b = SignedContribution::create(&signer("b"), contrib(1.0, 1.0, "pb"), "mb", "tb");
        let r1 = gather_and_aggregate(&[a.clone(), b.clone()], &v);
        let r2 = gather_and_aggregate(&[b, a], &v);
        // Reordering the inputs yields the identical committed hash + total.
        assert_eq!(r1.merged_hash, r2.merged_hash);
        assert_eq!(r1.total_reward_weight, r2.total_reward_weight);
    }

    #[test]
    fn tolerance_accepts_within_and_rejects_outside() {
        let cent = Q16::from_f32(10.0);
        // within 5%: 10.3 vs 10.0 -> 3% ok
        assert!(within_tolerance(Q16::from_f32(10.3), cent, 0.05));
        // outside 5%: 11.0 vs 10.0 -> 10% not ok
        assert!(!within_tolerance(Q16::from_f32(11.0), cent, 0.05));
        // exact match is always within any non-negative tolerance
        assert!(within_tolerance(cent, cent, 0.0));
    }

    #[test]
    fn finalize_round_commits_then_settles_each_accepted() {
        use std::cell::RefCell;

        struct RecChain(RefCell<Vec<String>>);
        impl ChainCommit for RecChain {
            fn commit_trace_hash(&self, h: &str) -> Result<String, FederationError> {
                self.0.borrow_mut().push(h.to_string());
                Ok(format!("tx:{h}"))
            }
        }
        struct RecSettle(RefCell<Vec<(String, Q16)>>);
        impl Settlement for RecSettle {
            fn settle(&self, node: &str, w: Q16) -> Result<(), FederationError> {
                self.0.borrow_mut().push((node.to_string(), w));
                Ok(())
            }
        }

        let v = roster(&["a", "b"]);
        let cs = vec![
            SignedContribution::create(&signer("a"), contrib(4.0, 0.5, "pa"), "ma", "ta"),
            SignedContribution::create(&signer("b"), contrib(2.0, 1.0, "pb"), "mb", "tb"),
        ];
        let r = gather_and_aggregate(&cs, &v);
        let chain = RecChain(RefCell::new(Vec::new()));
        let settle = RecSettle(RefCell::new(Vec::new()));
        let receipt = finalize_round(&r, &chain, &settle).unwrap();

        assert_eq!(receipt, format!("tx:{}", r.merged_hash));
        assert_eq!(chain.0.borrow().len(), 1); // committed exactly once
        assert_eq!(settle.0.borrow().len(), 2); // settled each accepted node
    }
}
