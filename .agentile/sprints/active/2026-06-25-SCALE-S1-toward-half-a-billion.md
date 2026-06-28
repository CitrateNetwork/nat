---
created: 2026-06-25T00:00:00Z
branch: main
author: Larry Klosowski (@SaulBuilds) + Claude Opus 4.8 (1M context)
status: active
sprint: SCALE-S1
---

# SCALE-S1 — Toward half a billion parameters

H-01 is no longer the question. On corpus-v4 (31M tokens) the per-position BPE-4096
architecture holds the bet **5/5 at 4M and 8M params**, and the NAT-over-dense gap
**widens with scale** (0.188 → 0.251 bits/byte, within-recipe, all seeds stable after
the warmup+clip fix). The bet now wants the one thing it hasn't had: **scale**, on the
order of **half a billion parameters (~500M)** — the rung between L1 (done) and L2
(~10B, `g5-l2`, owner-gated).

This sprint is the program to get there. It is honest about the binding constraint:
**500M needs billions of license-clean tokens; we have 31M — a ~100–300× gap.** Closing
that gap (not the architecture) is the work. Everything else serves it or de-risks the
run that consumes it.

Frame: `PLANSET/04_DATA_OPS.md` · `docs/DGX_HANDOFF.md` §5.3 · the research loop
(`.agentile/research-loop/INTENT.md`). Pairs with **HERMES-S1** (the agent that
collects + refines at volume).

## The token-budget reality (why this is a program, not a knob)

A from-scratch LM at 500M params wants ~5–10 tokens/param *minimum* to not be trivially
data-bound — **2.5–5B tokens**. corpus-v4 is 31M. So the ladder cannot simply turn the
`target_params` knob up: at 16M+ params on 31M tokens the model starts memorizing the
corpus, and a memorized H-01 read means nothing. **Corpus volume gates every rung above
~8M.** Two secondary facts shape the design:

- **Tokenizer.** At BPE-4096 the embedding+readout dominate a *small* model's budget (the
  current "core-only signal" caveat). That self-resolves as `d` grows — at 500M the cores
  dominate — but a bigger vocab (16k) is still needed for **token efficiency** (fewer
  steps to cover billions of tokens) and held-out quality.
- **Run length.** A 500M run is days. The architecture we're scaling (`AutoregLm`) has
  **no checkpoint/resume** today (only `NatTrainModel` does). The host crash mid-this-very-
  -sprint is the proof: crash-safety is a hard prerequisite, not a nicety.

## Work packages

Three workstreams. **A** (infra) and **B** (data) run in parallel; **C** (ladder) consumes
both. Ordered within each by dependency.

### Workstream A — Crash-safe, scalable training infra
| WP | Subject | Acceptance | Status |
|----|---------|-----------|--------|
| WP-S1 | **Checkpoint/resume for `AutoregLm` + `AutoregDenseLm`** | `save(dir)`/`load(dir)` via safetensors; periodic checkpoint inside the train loop; a resumed run continues from the last step; round-trip + resume-equivalence tests | 🟡 **weight-level done (2026-06-25)** — `save`/`load` + `train_minibatched_checkpointed` (per-epoch `model.safetensors`+`meta.json`) on both arms, 2 round-trip tests green (39 total). **Remainder:** wire resume into the run/example path (read `meta.json`, skip done epochs) + serialize AdamW state for *bit-identical* continuation (today a resume restarts the optimizer + LR warmup). |
| WP-S2 | **bf16 mixed precision + gradient accumulation** | configurable dtype + accum steps; large effective batch fits the GB10 unified pool at d≫500; numerics within tolerance of f32 on a small model | 🟡 **designed (2026-06-25)** — implementation plan + obstacle in the design note below. Impl + numerics/throughput validation **gated on a free GPU** (the 16M/32M run holds the card); not landed blind. |
| WP-S3 | **Throughput profile of the candle-cuda path** | per-step wall-clock vs `d`/batch; identify the bottleneck (attention? matmul? host transfer) and a target tokens/sec at 500M | 🟡 **harness built (2026-06-25)** — `crates/nat-candle/examples/bench_throughput.rs` (tokens/sec at a target param scale; CPU-smoked). **GPU run pending** (don't compete with the live ladder); pair with `nvidia-smi` for peak memory. |

### Workstream B — Data pipeline to billions of license-clean tokens
| WP | Subject | Acceptance | Status |
|----|---------|-----------|--------|
| WP-S4 | **Data-volume scoping** | a written, defensible inventory: realistic license-clean token yield per source (Gutenberg, Wikipedia, permissive code, HF datasets) + the connector work each needs; verdict on whether 2.5–5B is reachable | ✅ **done (2026-06-25)** — `research-loop/DATA_VOLUME_SCOPING.md`. Verdict: **reachable with margin** (Gutenberg-full + Wikipedia + permissive code clear 2.5–5B several×); the program is **engineering, not sourcing**. Empirical counting run still owed before committing storage. |
| WP-S5 | **Streaming/sharded pipeline** | the pipeline processes inputs without holding the whole corpus in memory; bounded RAM at 20GB+ input; deterministic shards preserved | ☐ |
| WP-S6 | **Volume connectors** — full-Gutenberg sweep, HF-datasets (permissive subsets), Wikipedia (CC-BY-SA) | each emits `RawDoc` JSONL through the fail-closed license gate; license screening at scale; provenance immutable | ☐ |
| WP-S7 | **Model-based quality filter as a fail-closed GATE** | upgrade the L0 heuristic scorer (`run_pipeline_with_scorer` is a score today) to a gate that quarantines below-threshold docs; tuned on a labeled sample | ☐ |
| WP-S8 | **Corpus storage + offsite backup** | a multi-billion-token corpus has a durable home (not gitignored-local-only) + a backup/restore runbook; the v4 rebuild-after-crash doesn't recur at 100× the size | ☐ |

### Workstream C — Ladder, evals, and the 500M run
| WP | Subject | Acceptance | Status |
|----|---------|-----------|--------|
| WP-S9 | **corpus-v5 + BPE-16k** | 5–10× the volume haul folded through the pipeline; fresh BPE-16k; manifest + quality recorded | ✅ **done (2026-06-25)** — 1500 PD Gutenberg books + v3/v4 pillars → **392,499 docs / 167.2M tokens (5.4× v4) / quality 0.857**, 7,316 quarantined (dedup+PII, 0 license). **BPE-16384 @ 2.53 bytes/tok ≈ 383M BPE tokens.** `scripts/build-corpus-v5.sh`. Unblocks 16M→32M rungs; 64M wants corpus-v6. NOT yet durably stored (WP-S8 open). |
| WP-S10 | **Ladder rungs 16M → 32M → 64M** | param-matched NAT vs dense at each, on corpus-v5, held-out bits/byte, gap reported per rung (honest: narrowing = report it) | 🟡 **16M+32M done (2026-06-27)** — both HOLD 3/3. 16M (f32) gap **0.191**, 32M (bf16) gap **0.176** → **flat/slightly narrowing, NOT widening** (corpus-v4 widened; BPE-16k embedding-dominated → flatter per-total-param gap; the −0.015 is confounded by the f32→bf16 change + 3-seed noise — re-run 16M in bf16 for a clean read). 64M wants corpus-v6. |
| WP-S11 | **Eval battery beyond bits/byte** | domain-split held-out perplexity + a few small downstream tasks; a capability read that isn't just LM loss | ☐ |
| WP-S12 | **The 500M run: 128M → 256M → 512M** | each rung param-matched (sampled seeds at the top to bound compute), checkpointed, gap holding; `g3b` exit | ☐ (north star) |

## Sources & licensing posture (unchanged, hard rules)

The fail-closed `ALLOWED_LICENSES` gate (CC0/CC-BY/CC-BY-SA/MIT/Apache-2.0/BSD-3-Clause/
public-domain) holds at every scale. New source *domains* require owner approval (the
Hermes approval queue) before bulk fetch. Provenance (`source`/`license`/`fetch_date`/
`raw_hash`) stays immutable; PII and non-permissive licenses are quarantined, never
trained. **At billions of tokens, license screening must be enforced per-doc in the
streaming path, not as a post-hoc pass.**

## Exit criteria

- [ ] A multi-day `AutoregLm` run survives a kill -9 and resumes from its last checkpoint
      with bit-identical continuation (WP-S1).
- [ ] A written scoping verdict on whether 2.5–5B license-clean tokens is reachable, with
      per-source numbers (WP-S4).
- [ ] corpus-v5 ≥ several hundred million tokens, fail-closed-clean, durably stored (WP-S9, S8).
- [ ] H-01 read at ≥64M params on corpus-v5 (WP-S10) — the next honest ladder point.
- [ ] **`g3b`: a 500M-param H-01 read** — partitioned vs equal-param dense, held-out,
      checkpointed, with the verdict reported as-is (WP-S12).

## Honest posture

The widening gap (0.024 → 0.251 across the ladder so far) is encouraging and it is an
**extrapolation** — three-to-five small points, none above 8M, none above 31M tokens. A
500M run on billions of tokens could flatten or reverse it. If it does, that is the
result and we change course; the whole point of building the ladder cheaply is to find
that out before committing L2 (10B) compute. We are **not** claiming 500M trains today —
we're naming the rung, the data gap it requires, and the infra that makes the run safe to
attempt.

## Dependencies / sequencing note (owner-set 2026-06-25)

Order: **(1)** this plan + the `g3b` gate · **(2)** WP-S1 checkpoint/resume · **(3)** WP-S4
data-volume scoping · **then** WP-S9 corpus-v5 and onward. A and B parallelize after that;
C consumes them.

## WP-S2/S3 design note — bf16 mixed precision (2026-06-25)

Investigation findings (so the implementation is a known quantity, not a guess), and **why
it wasn't landed blind**: the bf16 numerics and the throughput payoff can only be validated
on the GPU, which is busy running the 16M/32M ladder (~15h). Landing numerically-delicate
changes to softmax/SSM/optimizer paths without GPU validation is exactly the kind of move
this lab's honest posture rejects. So WP-S3's **measurement harness** is built first
(`bench_throughput.rs`) and the implementation below is specified, to be landed *with* GPU
validation the moment the card frees.

**The obstacle (the non-obvious part).** bf16 is *not* a one-line VarBuilder flip. The seeded
init (`seed.rs::seeded_linear` / `seeded_scalar_var` / `seeded_uniform`) builds every weight
as **f32** via `Tensor::from_vec(Vec<f32>)` and inserts it into the `VarMap` directly,
bypassing the VarBuilder dtype — and `seeded_linear` then builds the `Linear` via `linear(.., vb.pp(..))`
which reads those vars back. So bf16 weights require **threading a dtype through the seeded
helpers** (cast the f32 init → dtype before `Var::from_tensor`) AND confirming candle's
`VarBuilder::get` / `VarMap` behavior on a bf16 dtype (open question to resolve by running).

**The plan (matmuls in bf16 for the win; f32 for the sensitive ops):**
1. `seed.rs`: dtype-aware init (additive `*_dt` variants; existing f32 call sites unchanged).
2. `AutoregLm` / `AutoregDenseLm`: add a `dtype` field; `new()` → f32 (unchanged, default),
   `new_with_dtype(cfg, dtype)` builds weights + VarBuilder in `dtype`.
3. **f32-protect the sensitive ops** (no-ops when already f32, so the f32 path stays
   bit-identical and the existing 39 tests still pass):
   - `CausalAttn`: upcast scores → f32 for `+mask` and softmax, downcast weights → compute dtype.
   - `CausalSsm`: compute the decay matrix (`log_a.exp().log()`, `tk·rate`.exp()) in f32, cast
     the result → compute dtype for the matmul (the exp/log is the precision-sensitive part).
   - per-position merge softmax + `cross_entropy`: cast logits → f32.
   - Causal masks stay f32 (added in f32 space) — no dtype threading needed there.
4. Optimizer: start with **pure bf16 storage** (max memory saving; AdamW moments in bf16),
   guarded by the warmup+grad-clip from WP-S1's fix. If a GPU run shows instability, escalate
   to **f32 master weights + bf16 compute** (more memory, more stable) — that decision is a
   GPU-validation output, not a guess.
5. **Gradient accumulation**: accumulate grads over N micro-batches before `opt.step` (already
   have the explicit `backward()`/`clip`/`step` split from WP-S1) for a large effective batch.

**Validation plan (when the GPU frees):** CPU tolerance test (bf16 loss ≈ f32 within ε on a
tiny model) → `bench_throughput` f32-vs-bf16 at 8M/16M/64M (tok/s + nvidia-smi peak memory) →
a short bf16 training run to confirm stability → only then flip the ladder's upper rungs to bf16.
