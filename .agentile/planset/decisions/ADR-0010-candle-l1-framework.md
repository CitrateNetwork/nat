# ADR-0010 — Candle as the L1 tensor framework

**Status:** accepted · **Date:** 2026-06-21 · **Decides:** WP-1.1

## Decision
The L1 training stack uses **Candle** (huggingface/candle) for zone cores and
training. Candle-backed cores live in the `nat-candle` crate and implement the
existing `nat_core::cores::ZoneCore` trait, so they drop in behind the trait with
no change above (ADR-0009). The same op graph runs on CPU now and on a CUDA
device at L1 by swapping `Device::Cpu` for a GPU device.

## Rejected
- **Burn** — more mature autodiff and a training-first design, but heavier and a
  larger commitment. Candle was chosen for the reasons below.
- **Pulling the framework into `nat-core`** — would slow the fast L0 default build
  for every developer. Kept in a separate crate so the L0 path stays Candle-free.

## Why
- **HF ecosystem + GGUF-native.** Candle reads/writes GGUF directly, which helps
  the serialization/export path (Architecture §10, critique #7) rather than
  fighting it.
- **Lighter compile** than Burn, keeping iteration tolerable.
- **CPU + CUDA via a device swap** — the cores are GPU-ready without code changes.
- **The `ZoneCore` seam isolates the choice** — if Candle's training ergonomics
  disappoint at scale, only `nat-candle` changes, not the architecture.

## Evidence
`nat-candle` ships `CandleSsmCore` (linear recurrence as a single lower-triangular
matmul) and `CandleAttentionCore` (matmul + softmax), both deterministic and
trait-conformant, plus `train_tiny_zone_head`, a smoke test proving forward +
autodiff backward + AdamW reduce loss toward zero on CPU. This de-risks critique
#6 (the Rust training stack) before the expensive run.

## Honest note
Candle's training/autodiff story is less battle-tested than Burn's. The smoke test
covers the basic loop; the real stress is L1 training at 1–2B params, where this
ADR may be revisited if the ergonomics or performance don't hold.
