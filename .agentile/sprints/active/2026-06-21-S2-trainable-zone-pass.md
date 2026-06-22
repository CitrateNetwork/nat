---
created: 2026-06-21T00:00:00Z
branch: docs/s2-trainable-zone-pass
author: Larry Klosowski (@SaulBuilds) + Claude Opus 4.8 (1M context)
status: active
sprint: NAT-S2
---

# NAT-S2 — The trainable end-to-end zone pass (Gate 3 on the DGX)

The L1 build: make the whole `NatModel` pass differentiable so gradients flow from
an output loss back through merge → zones → router → embedding, train it for real
on the GB10, and then run the **conclusive H-01 ablation** with the real model as
the partitioned arm. This is the federation backlog **item #4** (handoff §6) and is
what unblocks the bet-deciding verdict.

It discharges the still-open L1 parts of the planset: **WP-1.2** (real training
loop), **WP-1.3** (conclusive H-01), and **WP-1.5** (trained router for H-02).
Canonical sprint frame: `PLANSET/07_SPRINTS_AND_WPS.md` § Sprint 1.

Already landed this DGX cycle (precursors, not part of this sprint's WPs): the GPU
device swap (`nat-candle`, PR #8, `candle-cuda` live on the GB10) and the H-01
ablation taken to GPU + multi-seed with *structural-analog* arms (`nat-ablation`,
PR #9). This sprint replaces the analogs with the real model.

## Why now — what blocks training today

1. **The autodiff graph dies at the zone boundary.** `ZoneCore::forward(&[f32]) ->
   CoreOutput([f32;8])` returns plain arrays; even the Candle cores `.to_vec1()`
   back out. No tensor graph survives zones → merge → output, so nothing backprops.
2. **The merge is non-differentiable** — hard top-k prune (drops ~80%) + Q16.16
   compose. Zero gradient to pruned zones; quantization is not differentiable.
3. **The router is hand-wired** (`route()` is a fixed signal→activation map;
   ADR-0001 says L1 replaces it with a learned gate).
4. **The embedding is hand-crafted** (`embed()` non-learned); no task / loss /
   optimizer threads the whole pass.

## Governing constraints (do not relitigate; extend)

- **ADR-0006 (decision- vs bit-faithful).** The provenance/inference path **keeps**
  the hard Q16.16 decision-faithful merge — that is the product. Training runs a
  **parallel differentiable** merge. The two must be *reconciled*: hardening the
  soft weights must reproduce `nat_provenance::prune_and_reweight`'s survivor set,
  so we train the structure we ship. **This reconciliation is the sprint's main
  correctness risk** (see Risks).
- **ADR-0008 (zone staging).** The first trainable build and the first conclusive
  H-01 run on the **3-zone {HP, PF, CX}** subset (data-rich attention zones), not
  all six. SM/CB widen later when their data earns it.
- **Determinism (AGENT_ENTRY / ADR-0009).** Merge + reward stay on `nat_types::Q16`;
  never introduce float into the trace-hash or merge path. The trace stays the
  product — anything changing what a pass records touches `nat-provenance` (Tier-1).
- **ADR-0010.** Candle is the L1 framework.

## Work packages (red-test-first)

| WP | Subject | Red-test (acceptance, written first) | Discharges | Status |
|----|---------|--------------------------------------|------------|--------|
| WP-1 | **Tensor-native trainable spine** — a parallel forward keeping Candle `Tensor`s end to end (cores expose `forward_tensor`; the `[f32;8]` `ZoneCore` API stays for the trace path). | A loss at the output produces **nonzero gradient on every zone's params**; an AdamW step reduces the loss. | WP-1.2 | ✅ **done** — `nat-candle::trainable` (`TrainableZonePass`, `TensorCore` + tensor-native `AttnCore`/`SsmCore`, seeded-reproducible). `every_param_has_gradient` + `training_reduces_loss` green on CPU **and GPU** (`scripts/dgx-gpu.sh test`). |
| WP-2 | **Differentiable merge reconciled to the hard one** — temperature-softmax over zone scores → weighted compose (f32, differentiable), annealable toward top-k. | (a) gradient flows to scores; (b) **hardening the soft weights reproduces `prune_and_reweight`'s survivor set** on a prompt battery (decision-faithful bridge, ADR-0006). | WP-1.2 | planned |
| WP-3 | **Learned router gate** (ADR-0001) over input features, modulating **only declared edges**. | (a) gate is trainable (grad flows); (b) **no undeclared edge ever receives weight** (C-1 invariant, property-tested); (c) `nat_eval` routing-divergence on the trained gate **beats the L0 hand-wired baseline**. | WP-1.5 (H-02) | planned |
| WP-4 | **Learned embedding + real task + the loop** — trainable embedding, a task (scaled synthetic-structured first, then `nat-data` shards), GPU AdamW loop, 3-zone, seeded-reproducible, emitting `StepContribution`. | held-out loss drops; a `StepContribution` with `reward_weight = compute × quality` is emitted per step; a checkpoint round-trips (save → load → identical forward). | WP-1.2 | planned |
| WP-5 | **Swap the real NatModel into the H-01 ablation** — replace the analog `PartitionedArm` with the trainable 3-zone NatModel and `DenseArm` with a real equal-param dense transformer; keep ADR-0005 + `guard_not_toy`. | ablation runs with the **real** NatModel arm on GPU, param-matched ≤ tolerance, multi-seed; reports cap/param. The conclusive g3-h01 read. | WP-1.3 (g3-h01) | planned |
| WP-6 *(stretch)* | **Deterministic-inference mode** for bit-faithful `output_hash` (ADR-0006 optional mode; H-03b). | re-running inference reproduces `output_hash` bit-for-bit under the mode. | H-03b | stretch |

## Sequencing & dependencies

```
WP-1 ──┬── WP-2 ──┐
       └── WP-3 ──┤
            WP-4 ──┴── WP-5   (conclusive H-01)
```

WP-1 is the foundation. WP-2 and WP-3 build on it independently. WP-4 needs WP-1+2+3
(the full differentiable pass + a task). WP-5 needs WP-4 (the trainable model) and is
the payoff that closes the Gate-3 blocker. WP-6 is optional and may slip to a
follow-up.

## Risks

- **R1 — soft/hard merge divergence (primary).** If the trained soft weights and the
  recorded hard top-k disagree on survivors, we optimize one structure and ship
  another, breaking the decision-faithful claim. *Mitigation:* anneal the softmax
  temperature toward hard selection during training; assert decision agreement on a
  battery as a WP-2 gate; consider a straight-through estimator if annealing is
  insufficient. Bind the agreement check to `verify_decision_faithful`.
- **R2 — GPU non-determinism.** Float reduction order on CUDA can vary run-to-run.
  *Mitigation:* target **decision-faithful + seeded-init reproducibility** (as the
  ablation already does); bit-faithful is explicitly the optional WP-6 mode per
  ADR-0006 — do not over-claim it.
- **R3 — data readiness.** Real-corpus ingestion (`nat-data` + a real tokenizer) is
  a parallel thread; WP-4 starts on a scaled synthetic-structured task to prove the
  loop, then swaps in real shards. Log the swap; don't silently ship synthetic.

## Exit criteria (Gate-3 bindings)

- [ ] **g3-train** — the 3-zone NatModel trains end-to-end on the GB10 (`candle-cuda`),
      loss drops on held-out data, reproducible from a seed; emits `StepContribution`.
- [ ] **g3-routing (H-02)** — the trained router's routing-divergence beats the L0
      baseline on `nat_eval`'s labeled battery, above the significance threshold.
- [ ] **g3-h01 (BLOCKER)** — the conclusive ablation runs the **real** NatModel arm
      vs an equal-param dense transformer under ADR-0005, multi-seed, on GPU, and
      reports cap/param. **If partitioned < dense, H-01 is refuted — say so.**
- [ ] decision-faithfulness preserved: hardened training merge == provenance merge on
      the battery; Gate-2 acceptance suite still green; merge/trace path still Q16.16.
- [ ] full gate green: `cargo test --workspace`, clippy `-D warnings`, fmt; GPU suite
      via `scripts/dgx-gpu.sh test`.

## Out of scope (carried)

- **g3-gguf** (GGUF `FlattenedDense` export, Ollama load) — that is item #3 / WP-1.4,
  a separate sprint.
- **SM/CB zones** — widened past the 3-zone subset only once their data earns it
  (ADR-0008).
- **Gate-4 federated** (signed multi-node gather, on-chain commit) — Sprint 2/§07.

## Honest posture

The H-01 verdict from WP-5 is the real one — the structural-analog read (PR #9) was
explicitly necessary-not-final. If the conclusive run refutes H-01 at equal params,
the right move is to say so and change course; the scale ladder exists to learn that
on the GB10 before any L2 commit.

## Close-out (to fill at sprint close)

- REPORT.md citing the ticked exit lines + the conclusive H-01 number; move to
  `completed/2026-06/`.
- Update `gates.yaml` (g3-train / g3-routing / g3-h01) and `hypotheses.md`
  (H-01 → supported|refuted, H-02 → supported, H-03a continuity) with evidence.
- Next: WP-1.4 (GGUF round-trip) toward the rest of Gate 3.
