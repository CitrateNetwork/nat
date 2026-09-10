# ADR-0007 — Integrate with compute-pool for settlement (do not reinvent)

**Status:** accepted · **Date:** 2026-06-18 · **Remediation:** #1

## Decision
The participant economic layer — reward as a function of compute contributed ×
data quantity/quality — is settled by **`citrate-compute-pool`**, not by NAT.
NAT's responsibility ends at *emitting* the inputs: a metered-compute receipt, a
data-quality score (from the pipeline's QUALITY_SCORE stage), and a provenance
hash. compute-pool owns the reward math, tokenomics, and payout.

## Rejected
- A standalone settlement mechanism inside NAT — duplicates an audited Tier-1
  system (compute-pool already ships a compute marketplace, tokenomics
  simulation, reward settlement, training-worker + pool-coordinator).

## Why
The memory graph showed the economic layer the owner emphasized is already built
in compute-pool + federation. Reinventing it would split the audit surface and
diverge two reward implementations. NAT scores; compute-pool settles.

## Seam
`docs/SETTLEMENT_SEAM.md` specifies the interface; `nat_train::StepContribution`
is the accounting type, with `reward_weight() = compute_metered × data_quality`
as the *signal* compute-pool converts to payout. The boundary is: NAT decides the
weight formula; compute-pool decides the payout.
