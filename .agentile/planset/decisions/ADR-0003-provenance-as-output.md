# ADR-0003 — Provenance as a forward-pass output

**Status:** accepted · **Date:** 2026-06-18

## Decision
The zone-activation trace, confidence scores, prune decisions, inter-zone flows,
and tool routing are emitted as structured, deterministically-serializable output
on every pass — hashable for on-chain commitment and replay.

## Rejected
- Post-hoc interpretability tooling — cannot prove what actually ran; it
  reconstructs, it does not record.

## Why
This is the wedge against opacity and the basis for on-chain auditability (claim
C-2). Structure-as-interpretability: knowing which zone produced a contribution
is a property of the architecture, not of a tool. Realized in `nat-provenance`.
See ADR-0006 for the faithfulness scoping.
