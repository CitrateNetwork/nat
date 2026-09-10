---
created: 2026-06-22T00:00:00Z
closed: 2026-06-22T00:00:00Z
branch: docs/nat-s3-federated
author: Larry Klosowski (@SaulBuilds) + Claude Opus 4.8 (1M context)
status: completed
sprint: NAT-S3
---

# NAT-S3 — Gate 4: federated proof

The L3 milestone (`gates.yaml` gate4): nodes train toward the shared model and
submit **signed contributions**; a gather verifies the signatures, aggregates the
reward weights, commits the merged trace-hash **on-chain**, and **settles** through
compute-pool. This sprint scaffolds the testable core now and documents the path to
the real multi-node + on-chain deployment.

Frame: `PLANSET/07` Sprint 2 + `docs/SETTLEMENT_SEAM.md`. Builds on: the settlement
seam (`nat_train::StepContribution`, `reward_weight = compute × quality`), the
provenance trace + hash (`nat-provenance`), and the operator signer (already shipped
in the gateway: AWS-KMS / ed25519, custody-signed-off).

## The four exit criteria (gates.yaml)

| id | desc | this sprint |
|----|------|-------------|
| g4-gather | multi-node async signed gather; signatures verify before composition | ✅ **scaffolded** — `nat-federated::gather_and_aggregate` verifies each signed contribution and drops invalid ones before aggregating |
| g4-tolerance | federated result matches centralized within tolerance (H-05b) | ✅ **harness** — `within_tolerance(federated, centralized, tol)`; the real run compares a federated aggregate to a centralized baseline |
| g4-onchain | on-chain trace-hash commit + auditor replay reproduces output hash | gated — `ChainCommit` trait; the impl calls `citrate-chain` (the agent-runtime's recorder already anchors hashes) |
| g4-settlement | contribution settles via compute-pool (compute × quality → reward weight) | gated — `Settlement` trait; the impl calls `citrate-compute-pool` |

## Work packages

| WP | Subject | Status |
|----|---------|--------|
| WP-F1 | **Signed contribution + gather** — `SignedContribution` (node id + StepContribution + manifest/trace hashes + sig), a canonical sign-message, `gather_and_aggregate` (verify → drop invalid → aggregate `reward_weight` on Q16, merge the trace-hashes). Pluggable `Signer`/`Verifier`. | ✅ scaffolded (`nat-federated`) |
| WP-F2 | **Tolerance harness** — `within_tolerance` (H-05b): a federated aggregate must match a centralized baseline within a slack. | ✅ scaffolded |
| WP-F3 | **On-chain commit** — implement `ChainCommit` against `citrate-chain` (the agent-runtime recorder anchors hashes today); auditor replay reproduces the output hash. | gated (chain) |
| WP-F4 | **Settlement** — implement `Settlement` against `citrate-compute-pool`; end-to-end "compute × quality → reward weight → payout". | gated (compute-pool) |
| WP-F5 | **Real multi-node run** — N nodes train on disjoint shards, sign, gather over a wall-clock async network (the L0 simulated gather becomes real, same deadline discipline). | gated (nodes) |
| WP-F6 | **Production signer** — swap the toy HMAC signer for the operator signer (ed25519 / AWS-KMS, already built in the gateway). | gated (custody) |

## Architecture

```
node_i: train(shard_i) → StepContribution_i + trace_hash_i
        sign( node_i || contribution_i || manifest_hash || trace_hash_i ) → sig_i
        submit SignedContribution_i
gather: verify(sig_i) ∀i  → keep valid                       (g4-gather)
        aggregate reward_weight = Σ compute_i × quality_i      (Q16, deterministic)
        merged_hash = H(sorted valid trace_hashes)
        ChainCommit.commit(merged_hash)                        (g4-onchain)
        Settlement.settle(node_i, weight_i) ∀ accepted         (g4-settlement)
verify: federated aggregate ≈ centralized baseline (± tol)     (g4-tolerance / H-05b)
```

The verify-before-compose order is the security property: a bad signature can never
enter the aggregate or the on-chain commit. Determinism (Q16, sorted hashes) is what
lets an auditor replay the merged hash.

## Honest scope

This sprint delivers the **gather/verify/aggregate/tolerance core** (testable without
a network or chain) plus the `ChainCommit`/`Settlement` seams. The **real** Gate-4 —
multiple wall-clock nodes, the on-chain commit against `citrate-chain`, settlement
through `citrate-compute-pool`, and the production signer — is gated on that infra and
is the deployment phase (WP-F3..F6). H-05b (federated ≈ centralized) is a statistical
claim proven only by a real run.

## REPORT — close-out (2026-06-22)

**Status: COMPLETE as a scaffold sprint.** The two in-scope WPs (WP-F1 gather/verify/
aggregate, WP-F2 tolerance harness) are delivered and tested in `nat-federated`. The
four infra-dependent WPs (WP-F3..F6) are **carried, gated on owner-provisioned infra**
— they were never in this sprint's scope, which was explicitly "scaffold the testable
core + document the path." **`gates.yaml` gate4 stays `met:false` across all four
criteria; this close-out flips nothing.** Honest posture: the security order and the
determinism are done and checkable today; the proof-at-scale is not, and H-05b carries
no number until a real run produces one.

### What landed (evidence — `nat-federated`, 7 tests green)
- **WP-F1** signed gather, verify-before-compose: `gather_and_aggregate` verifies every
  signature *before* anything enters the aggregate or the committed hash. Adversarial
  coverage — `forged_signature_is_rejected_before_aggregation`,
  `tampering_a_field_after_signing_fails_verification`, `unknown_node_fails_closed`
  (all fail closed) + `valid_contributions_are_accepted_and_aggregated`. Reward total
  is a Q16 sum; merged trace-hash is over the *sorted* accepted hashes
  (`merged_hash_is_order_independent`) — a function of the accepted set, not arrival
  order, so an auditor's replay is deterministic.
- **WP-F2** H-05b tolerance harness: `within_tolerance(federated, centralized, tol)`
  on the Q16 grid (`tolerance_accepts_within_and_rejects_outside`).
- **Seams:** `ChainCommit` / `Settlement` traits driven by `finalize_round`
  (commit-once-then-settle-each-accepted; `finalize_round_commits_then_settles_each_accepted`).
  Signing is pluggable behind `Signer`/`Verifier`; a toy keyed-hash signer stands in
  for the production operator signer the gateway already ships.

### Gate / hypothesis state at close (unchanged — honest)
- `gates.yaml` gate4: **all four exit criteria `met:false`**, each carrying a
  `scaffold:` note recording exactly what the tests prove. `status: pending`.
- `hypotheses.md`: **H-05b open** (statistical L3 claim; the harness exists, the number
  comes only from a real run). H-05a remains supported (TLA+ + merge determinism test).

### Carried forward (infra-gated — DO NOT start without owner go-ahead)
- **WP-F3** on-chain commit — real `ChainCommit` against `citrate-chain`.
- **WP-F4** settlement — real `Settlement` against `citrate-compute-pool`.
- **WP-F5** real multi-node wall-clock gather over N nodes on disjoint shards.
- **WP-F6** swap the toy signer for the production operator signer (ed25519 / AWS-KMS,
  custody-signed-off in the gateway).
These are the deployment phase and the source of the only real H-05b number; they are
backlog item #6 (infra-gated) in the track handoff.
