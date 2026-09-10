# CS-00: L0 zone-partitioned forward pass

**Gate:** Gate 2 · **Rung:** L0 · **Dates:** 2026-06-18 .. 2026-06-18
**Authors:** Larry Klosowski (architect), Claude Code (build)

## Question
Does a six-zone partitioned forward pass run end to end and emit a provenance
trace that validates against the Gate-2 acceptance criteria?

## Setup
Rust workspace, toolchain 1.96.0, CPU. Six zones (SM/CB SSM, HP/PF/CX attention,
MX non-learned), hidden width D=96 (16/zone), D_OUT=8, default topology
(Architecture §5.1), prune_threshold 0.8, gather deadline 100 (logical ms). Toy
deterministic cores (ADR-0009). No training (L0 wires the pass). Config: the
default L0 sidecar (`Sidecar::default_l0`). Rerun: `cargo test --workspace`.

## What we measured
The Gate-2 acceptance suite (`features/gate2_*.feature`, realized as
`crates/nat-core/tests/gate2_*.rs`): all learned zones produce output with
confidence+latency; router emits a length-6 activation and modulates only
declared edges; merge scores→prunes→reweights with pruned zones recorded; trace
is complete, hashable, and decision-faithful; async gather times out a straggler
and stays consistent; the MCP harness never side-effects before the gate and
always reaches RETURN. Target: all green.

## Result
49 tests green (40 unit + 9 Gate-2 acceptance). Clippy clean under `-D warnings`.
The trace hashes reproducibly (same bytes twice) and the output hash is stable
across re-runs (bit-faithful at L0 because the merge is Q16.16 and the toy cores
are deterministic). `verify_decision_faithful` holds on every sampled prompt.

## What surprised us
Two things. First, the `f32`-derives-`Eq` compile error on `MergeParams` was a
useful early flag that the deterministic boundary must be explicit — `f32` does
not even admit `Eq`, which is exactly why the merge runs on Q16.16, not float.
Second, routing differentiation (H-02) showed up at L0 *despite* the router being
hand-wired, not trained: math vs narrative prompts already drive different zone
mixes (`nat_eval::routing_divergence > 0`). That is encouraging but not evidence
— a trained router is the real test (L1).

## Decision
Keep the `ZoneCore` trait seam (ADR-0009) — it made the toy/real boundary clean.
Promote the repo's Tier-1 classification (Gate-2 green is the promotion trigger,
`AUDIT_TIER.md`). Proceed to Sprint 1 (Gate 3), where H-01 — the bet — gets its
first real test.

## Open threads
- H-01 (capability/param) — untestable at L0; the load-bearing question opens at
  L1 under the ADR-0005 baseline protocol.
- H-03b (bit-faithful) — holds at L0; revisit under float cores at L1.
- TLC has not been run (no JRE in the bootstrap env); the three modules are
  written and checkable. First open item against `gates.yaml` g1-formal.
