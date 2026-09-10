# ADR-0004 — Auxiliary sidecar format

**Status:** accepted · **Date:** 2026-06-18

## Decision
GGUF/ONNX remain the tensor container. A sidecar (`.nat.json` or embedded GGUF
metadata KV) carries the zone graph, routing topology, training recipes, and
composition rules. A sidecar-unaware runtime runs the tensors opaquely; a
sidecar-aware runtime runs the full zone-partitioned pass.

## Rejected
- Forking GGUF — breaks the Ollama onramp; kills ecosystem adoption.

## Why
Backwards compatibility is the onramp (claim C-3). Realized in `nat-sidecar`.

## Amendment (remediation #7)
"Runs opaquely in Ollama" applies to a **flattened-dense** export, not to a
literal parallel-heterogeneous zone graph — GGUF has no layout for parallel
SSM+attention zones. The sidecar's `export_kind` (`ZonePartitioned` |
`FlattenedDense`) records which form the paired tensor container holds. The
sidecar is always the source of truth for the zone graph.
