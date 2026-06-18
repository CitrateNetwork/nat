# ADR-0002 — SSM cores in temporal zones

**Status:** accepted · **Date:** 2026-06-18

## Decision
Cerebellar (CB) and Sensorimotor (SM) zones use State Space Model cores for
linear-time recurrence and native temporal dynamics. Prefrontal (PF),
Hippocampal (HP), and Codec (CX) keep attention. Each SSM zone carries a thin
attention head for cross-zone communication.

## Rejected
- Attention everywhere — quadratic cost; no native temporal state.
- SSM everywhere — loses content-addressable retrieval where HP/PF need it.

## Why
Temporal zones benefit from explicit evolving state (logged like any other zone
signal); reasoning zones benefit from flexible look-back. Gluing SSM↔attention is
a known training risk (Master Plan risk register) — mitigation: stabilize SSM
zones in isolation, then unfreeze cross-zone heads on a schedule. Encoded in
`nat_types::ZoneId::default_core` and `nat-core::cores`.
