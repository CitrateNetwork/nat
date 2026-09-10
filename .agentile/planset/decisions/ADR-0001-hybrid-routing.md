# ADR-0001 — Hybrid routing

**Status:** accepted · **Date:** 2026-06-18

## Decision
Fixed inter-zone topology, with context-aware learned modulation of edge
strengths and zone activation. The router modulates a fixed graph; it cannot
create edges the topology does not declare.

## Rejected
- Pure hard-wiring — too rigid; no per-prompt adaptation.
- Pure learned MoE routing — loses interpretability; topology becomes opaque.

## Why
Keeps the topology auditable (an auditor and a TLA+ model both read it directly)
while letting signal strength adapt per prompt. This is claim C-1 and the
property `nat-core::router` enforces structurally (it only iterates declared
edges). "A little right brain, a little left brain, determined by the prompt."
