# ADR-0005 — The H-01 dense-baseline protocol

**Status:** accepted · **Date:** 2026-06-18 · **Remediation:** #2

## Decision
The H-01 ablation (does zone partitioning cost capability per parameter?) is only
valid if the dense baseline is pinned. The baseline MUST share with the NAT run:
identical parameter count (±1%), token budget, training data and order (same
shard manifest + seed), tokenizer, optimizer + schedule, and compute budget. The
**only** difference is the zone partitioning + routing + merge. Any other delta
invalidates the comparison.

## Why
"Dense baseline of equal params" proves nothing if the two runs differ on data,
seed, or compute. H-01 is the bet-deciding metric; a sloppy baseline would let a
favorable-but-meaningless number through.

## Gate binding
This is a **hard Gate-3 exit blocker** (`gates.yaml` g3-h01), not a reported
metric. If the ablation is not run to this protocol, Gate 3 does not pass.
