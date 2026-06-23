---
created: 2026-06-22T00:00:00Z
branch: docs/nat-s3-federated
author: Larry Klosowski (@SaulBuilds) + Claude Opus 4.8 (1M context)
status: active
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

## Close-out

- REPORT.md citing the real-run H-05b number; update `gates.yaml` gate4 + `hypotheses.md`
  H-05b; move to `completed/`.
