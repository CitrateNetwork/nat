---
created: 2026-06-21T00:00:00Z
closed: 2026-06-22T00:00:00Z
branch: docs/s2-trainable-zone-pass
author: Larry Klosowski (@SaulBuilds) + Claude Opus 4.8 (1M context)
status: completed
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
| WP-2 | **Differentiable merge reconciled to the hard one** — temperature-softmax over zone scores → weighted compose (f32, differentiable), annealable toward top-k. | (a) gradient flows to scores; (b) **hardening the soft weights reproduces `prune_and_reweight`'s survivor set** on a prompt battery (decision-faithful bridge, ADR-0006). | WP-1.2 | ✅ **done** — `nat-candle::merge_train` (`soft_weights`/`compose`/`argtopk`); spine composes via per-zone score heads → `softmax(·/τ)` → weighted sum → readout. Reconciliation over a battery + spine-level decision bridge + differentiability + annealing tests green (CPU + GPU). |
| WP-3 | **Learned router gate** (ADR-0001) over input features, modulating **only declared edges**. | (a) gate is trainable (grad flows); (b) **no undeclared edge ever receives weight** (C-1 invariant, property-tested); (c) `nat_eval` routing-divergence on the trained gate **beats the L0 hand-wired baseline**. | WP-1.5 (H-02) | ✅ **done** — `nat-candle::router::LearnedRouter` (feat→hidden→sigmoid gate; edges copied from the sidecar so only declared edges can be weighted). Trainable + declared-edges-invariant tests in nat-candle; H-02 comparison in `nat-eval` (dev-dep nat-candle): **trained separation 11.70 vs L0 baseline 4.25**. `separation_ratio` exposed for the shared metric. |
| WP-4 | **Learned embedding + real task + the loop** — trainable embedding, a task (scaled synthetic-structured first, then `nat-data` shards), GPU AdamW loop, 3-zone, seeded-reproducible, emitting `StepContribution`. | held-out loss drops; a `StepContribution` with `reward_weight = compute × quality` is emitted per step; a checkpoint round-trips (save → load → identical forward). | WP-1.2 | ✅ **done** — `nat-candle::train_loop::NatTrainModel` wires embedding → router (WP-3) → spine (WP-1) → merge (WP-2, score = activation × confidence) → readout; one optimizer over all vars, 3-zone, seeded. `held_out_loss_drops_end_to_end` + `emits_step_contributions_with_reward_weight` + `checkpoint_round_trips` green on CPU + GPU. StepContribution per step (`reward_weight = compute × quality`). Task is scaled-synthetic; real `nat-data` shards are the next data thread. |
| WP-5 | **Swap the real NatModel into the H-01 ablation** — replace the analog `PartitionedArm` with the trainable 3-zone NatModel and `DenseArm` with a real equal-param dense transformer; keep ADR-0005 + `guard_not_toy`. | ablation runs with the **real** NatModel arm on GPU, param-matched ≤ tolerance, multi-seed; reports cap/param. The conclusive g3-h01 read. | WP-1.3 (g3-h01) | ✅ **done** — `nat-ablation::real` (`run_real_ablation[_seeds]`): real `NatTrainModel` arm vs equal-param `DenseTransformerArm`, param-matched (search refuses on mismatch), multi-seed, `guard_not_toy`. GPU run (5 seeds, params 3882=3882): **H-01 HOLDS on the mean** (nat 4.37 ≥ dense 3.88 cap/param) but **only 3/5 seeds** — a marginal hold on the synthetic task; real-corpus is the final word. |
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

- [x] **g3-train** — the 3-zone NatModel trains end-to-end on the GB10 (`candle-cuda`),
      held-out loss drops, reproducible from a seed; emits `StepContribution` (WP-4).
      *Caveat: synthetic task; real-corpus `nat-data` is the next data thread.*
- [x] **g3-routing (H-02)** — the trained router beats the L0 baseline on `nat_eval`'s
      battery (**11.70 vs 4.25**, WP-3). *Caveat: in-sample; held-out is the final read.*
- [~] **g3-h01 (BLOCKER)** — the conclusive ablation runs the **real** NatModel arm vs
      an equal-param dense transformer under ADR-0005, multi-seed, on GPU (WP-5). First
      read: **HOLDS on the mean** (nat 4.37 ≥ dense 3.88) but only **3/5 seeds** — a
      *marginal* hold on the synthetic task. **Not yet decisive**; real-corpus data at
      larger scale is the final word. Honest posture: reported, not over-claimed.
- [x] decision-faithfulness preserved: hardened training merge == provenance merge on
      the battery (WP-2); Gate-2 suite still green; merge/trace path still Q16.16.
- [x] full gate green: `cargo test --workspace` (112), clippy `-D warnings`, fmt; GPU
      suite via `scripts/dgx-gpu.sh test`.

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

## REPORT — close-out (2026-06-22)

**Status: COMPLETE.** All five WPs delivered; WP-6 (deterministic-inference mode)
deferred as the stretch it always was. The sprint's mandate — make the whole
`NatModel` pass differentiable and train it end-to-end on the GB10 — is met, and it
produced the *first* real H-01/H-02 reads. Those first reads were honestly marginal
(H-01 3/5 on synthetic; H-02 in-sample); the **decisive** verdicts are not this
sprint's to claim — they landed downstream (see "What this sprint did NOT decide").

### What landed (evidence)
- **WP-1** tensor-native trainable spine — `nat-candle::trainable`
  (`every_param_has_gradient`, `training_reduces_loss`, CPU+GPU). Discharges WP-1.2.
- **WP-2** differentiable merge reconciled to the hard top-k — `nat-candle::merge_train`
  (hardening reproduces `prune_and_reweight`'s survivor set on the battery — the
  ADR-0006 decision-faithful bridge). Discharges WP-1.2.
- **WP-3** learned router gate, declared-edges-only invariant property-tested —
  `nat-candle::router::LearnedRouter`; H-02 separation **11.70 vs L0 4.25** (in-sample).
  Discharges WP-1.5.
- **WP-4** learned embedding + GPU AdamW loop emitting `StepContribution`
  (`reward_weight = compute × quality`); checkpoint round-trips —
  `nat-candle::train_loop::NatTrainModel`. Discharges WP-1.2.
- **WP-5** real `NatTrainModel` arm vs equal-param dense transformer in the H-01
  ablation, param-matched, multi-seed, GPU, `guard_not_toy` — `nat-ablation::real`.
  Discharges WP-1.3. **First read: HOLDS on the mean (nat 4.37 ≥ dense 3.88) but only
  3/5 seeds — marginal on the synthetic task.**

### Gate / hypothesis state at close
- `gates.yaml`: **g3-train met:true**, **g3-routing met:true**, **g3-h01 met:true**
  — all three carry their evidence and the honest scale caveats. (The `met:true` on
  g3-h01 / g3-routing reflects the *downstream* decisive/held-out reads, not WP-5's
  marginal synthetic one — see below; this REPORT does not re-flip anything.)
- `hypotheses.md`: H-01 supported, H-02 supported, H-03a supported by construction.

### What this sprint did NOT decide (handed downstream, honest posture)
- **The decisive H-01 (5/5 seeds, real corpus)** is **DATA-S1 WP-D6**'s result
  (`run_real_corpus_ablation`: nat 2.88–2.91 < dense 2.97–2.99), not WP-5's. WP-5's
  3/5 synthetic read is exactly why DATA-S1 went and got real data.
- **The held-out H-02** is the later `nat-eval::h02_heldout` read (trained 3.10 vs L0
  2.63 on unseen prompts, PR #29 `feat/nat-h02-heldout`), not WP-3's in-sample 11.70.

### Carried forward
- WP-6 deterministic-inference / bit-faithful `output_hash` (H-03b) — stretch, open.
- g3-gguf (GGUF round-trip + Ollama-class load) — shipped separately (PR #33,
  round-trip half done; execution half still gated, gates.yaml g3-gguf met:false).
- SM/CB zone widening — done at L scale in DATA-S1 WP-D11 (5-zone ladder).
