---
created: 2026-06-28T00:00:00Z
closed: 2026-06-28T00:00:00Z
branch: chore/repin-fed-types-dbbfbff
author: Larry Klosowski (@SaulBuilds) + Claude Opus 4.8 (1M context)
status: completed
sprint: I64-S1 (nat re-pin)
companion: ../../../../../citrate-federation/.agentile/planset/2026-06-28-i64-s1-q16-unification-and-deterministic-reroll.md
---

# nat — re-pin to the Tier-1-audited kernel (dbbfbff)

nat's federated half (`nat-lora`, `nat-aggregate`, `nat-federated`) consumes
`citrate-fed-types` for its Q16 + commitment primitives. When the kernel passed its
Tier-1 audit and the remediation merged, nat had to move to that exact rev so the
model side and the chain side share *one* audited kernel — not two copies drifting.
Merged on nat `main` (47e0262, PR #45).

> Scope fence held: this touches only the federated-infra crates. `nat-candle`
> (training / corpus / CUDA — the SCALE-S1 track) was not touched.

## What changed

- **Pin bumped** to `citrate-fed-types @ dbbfbff` (the remediated rev).
- **Adapted to one fail-closed API change.** The audit made `lora_commitment` return a
  `Result` (H2 — it now refuses shape-mismatched matrices instead of digesting garbage).
  nat-lora already guarantees the shape invariant at construction, so `commit.rs` wraps
  the call with `.expect("LoraAdapter invariant: matrices match declared rank/dim_out/
  dim_in")`. This is the honest seam: the kernel refuses untrusted shapes; nat asserts
  the invariant it already enforces, so the `expect` is unreachable-by-construction, not
  a swallowed error.

## Why this is the right shape

The kernel's job is to fail closed on *anyone's* bad input. nat's job, as a trusted
caller that builds its own matrices, is to uphold the invariant the kernel now checks.
The `.expect` documents exactly that contract at the call site — if it ever fires, a
nat-side invariant broke, which is precisely the signal you want, loud.

## Close

nat `main` builds + tests green on the new pin; the federated path now shares the same
audited Q16 the chain pins to in I64-S1. The kernel↔chain parity proof
(`citrate-chain/core/federated`) and this re-pin together mean model, chain, and kernel
agree on Q16 by construction.
