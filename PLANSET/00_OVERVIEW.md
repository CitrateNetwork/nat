# NAT Planset — Overview

**RFC-CIT-NAT-0001** · Draft v0.1 · Owner: Larry Klosowski (@saulbuilds) · Entity: Mozi Satori / Citrate Network

This is the repo-resident planset for the Citrate Neuroarchitectural Transformer.
It is the refined form of the source design set (`~/Downloads/NAT-Model-Specs/`),
brought into the repo, wired to the code, and extended with a sprint plan and a
critique-and-remediations record.

## Read in this order

1. **`01_MASTER_PLAN.md`** — what it is, why, scope, the scale ladder, the five
   gates, risks, prior-art posture.
2. **`02_ARCHITECTURE.md`** — the comprehensive design. Six zones, hybrid
   routing, the pruning merge, the provenance trace, the MCP harness, SSM
   integration, serialization, the novelty wedge.
3. **`03_FORMAL_SCAFFOLD.md`** — the TLA+/Gherkin source scaffold. Realized in
   `../formal/` (three checkable modules) and `../features/` (Gherkin by gate).
4. **`04_DATA_OPS.md`** — data shape by zone, sources, cleaning pipeline,
   federated data strategy, honest compute math.
5. **`05_RESEARCH_METHOD.md`** — the Agentile method. Materialized in
   `../.agentile/planset/` (gates, hypotheses, ADRs, case studies).
6. **`06_VISUAL_DESIGN_BRIEF.md`** — the training console brief for Claude Design.
7. **`07_SPRINTS_AND_WPS.md`** — the sprint plan: the first full pass (Gate 1 +
   Gate 2) and what each work package delivered, plus the road to Gates 3–5.
8. **`08_CRITIQUE_AND_REMEDIATIONS.md`** — the seven pushbacks raised in review
   and how each is carried as an ADR, a tightened claim, or a build decision.
9. **`09_JOURNAL.md`** — the first-person record of how this was designed.
   Conception record material for counsel.

## The one number to watch

**Hypothesis H-01:** zone partitioning does not reduce capability per parameter
versus a dense baseline of equal size. Unproven and load-bearing. The scale
ladder (L0/L1 on the Spark) exists to test it cheaply before the 10B commit. If
it fails, honest posture says change course. Tracked in
`../.agentile/planset/hypotheses.md`.

## Where the code is

The Rust reference implementation lives in `../crates/`. The first build target —
the L0 zone-partitioned forward pass that emits a provenance trace — is **done
and green** against the Gate-2 acceptance suite (`../features/gate2_*.feature`,
realized as `crates/nat-core/tests/gate2_*.rs`). See `07_SPRINTS_AND_WPS.md`.

## The economic layer

NAT does not reinvent reward settlement. It emits a metered-compute receipt, a
data-quality score, and a provenance hash that **`citrate-compute-pool`** settles
into participant rewards (it already ships a compute marketplace, tokenomics
simulation, and reward settlement). Participant economic advantage is a function
of **compute contributed × data quantity/quality submitted**. The interface is
`../docs/SETTLEMENT_SEAM.md`; the contribution accounting type is
`nat_train::StepContribution`.
