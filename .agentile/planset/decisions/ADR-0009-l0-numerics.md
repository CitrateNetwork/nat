# ADR-0009 — L0 numerics behind the ZoneCore trait

**Status:** accepted · **Date:** 2026-06-18 · **Remediation:** #6

## Decision
L0 zone cores are small, deterministic, dependency-light implementations (a
linear-recurrence SSM and a softmax-combine attention) written behind a
`ZoneCore` trait. Burn/Candle-backed cores slot in at L1 by implementing the same
trait — nothing above `nat-core::cores` knows whether a core is toy or trained.

## Why
Decision (owner): Rust reference only. Pulling a full training framework into L0
would inflate compile time and the L0 timeline without serving L0's actual goal,
which is "wire up the forward pass and prove the provenance log emits" (Master
Plan rung L0), not "train." Keeping L0 numerics minimal keeps the pass honest and
fast while the trait preserves the L1 upgrade path with no rework above the seam.

## Honest note
This is *not* a claim that Rust-from-scratch training is cheap. It is the
acknowledgement (remediation #6) that the L0 timeline is "wire the pass," and the
real training-stack cost lands at L1 (WP-1.1/1.2), budgeted there, not hidden.
