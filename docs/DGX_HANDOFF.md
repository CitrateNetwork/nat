# DGX Handoff — `nat` (Citrate Neuroarchitectural Transformer)

**Audience:** whoever picks this up on the DGX with **zero prior context.**
**Goal of this doc:** get you from `git clone` to running the real Gate-3 / L1
work (training real cores, running the H-01 ablation for real) without needing
anyone to explain anything.

Read this top to bottom once. It is self-contained.

---

## 0. TL;DR — what you are inheriting

`nat` is a Rust workspace implementing the **Citrate Neuroarchitectural
Transformer** (RFC-CIT-NAT-0001): a transformer whose hidden representation is
partitioned into six named **zones**, each with its own core (attention or
state-space), communicating over a **fixed topology** modulated per-input by a
learned router, merged by an **attention-scored, noise-pruned merge**, emitting a
**hashable provenance trace** of which zones fired and why.

- **Gate 2 is green:** the L0 (toy-core) forward pass runs end to end and emits a
  validated provenance trace.
- **Sprint 1 (CPU, GPU-free) is landed:** the Candle training stack, the data
  pipeline + quality scoring, the eval/routing harness, the H-01 ablation harness,
  and the reproducibility floor all exist and are tested.
- **Your job (Gate 3 / L1, needs the GPU):** train real cores at ~1–2B params,
  run the **H-01 ablation** for a conclusive verdict, GGUF round-trip, and the
  federated proof.

**The one number that decides everything: H-01** — does zone partitioning cost
capability per parameter versus a dense baseline of equal size? If it does, honest
posture says change course. The ablation harness (`nat-ablation`) exists to answer
this; you run it for real on the DGX.

---

## 1. Get it building (5 minutes)

```sh
git clone https://github.com/CitrateNetwork/nat.git
cd nat
# Toolchain is pinned in rust-toolchain.toml (1.96.0). rustup auto-installs it.
cargo build --workspace
cargo test  --workspace        # expect all green (see §3 for the count)
```

No GPU needed for the above — everything runs on CPU. If `cargo` isn't present:
install rustup (`https://rustup.rs`), then the pinned toolchain self-installs.

### Run the headline things right now

```sh
# The H-01 ablation (the bet), CPU/synthetic scale — illustrative only:
cargo run -p nat-ablation --example ablation

# Confirm CI passes in a clean Linux container (needs Docker / `colima start`):
scripts/ci-local.sh
```

---

## 2. The architecture in one screen

Pipeline (forward pass, `nat-core`):

```
prompt
  → featurize   (class signals + a hidden embedding)
  → router      (per-zone activation + edge strengths over the FIXED topology)
  → zones       (parallel cores: SSM for SM/CB, attention for HP/PF/CX)
  → gather      (deadline discipline; a slow zone → timed_out, never blocks)
  → merge       (score → prune bottom ~80% → re-weight → compose, on Q16.16)
  → MX harness  (non-learned state machine; gates any tool use, fail-closed)
  → (Output, Trace)
```

The six zones: `SM` Sensorimotor (SSM), `CB` Cerebellar (SSM), `HP` Hippocampal
(attn), `PF` Prefrontal (attn), `CX` Codec (attn), `MX` MCP-harness (non-learned).
Five learned + one non-learned executive harness.

**Two invariants you must not break:**
- The **merge and reward math run on `nat_types::Q16` (Q16.16 fixed-point), never
  `f32`.** This is what makes federated results reconcile and on-chain provenance
  verify. Do not introduce float into the merge/trace-hash/reward path.
- **The provenance trace is the product**, not a debug aside. Anything that
  changes what a pass records is Tier-1 (`AUDIT_TIER.md`).

Full design: `PLANSET/02_ARCHITECTURE.md`. Formal specs: `formal/` (three TLA+
modules). Acceptance criteria: `features/` (Gherkin by gate).

---

## 3. The crate map (10 crates)

| Crate | What it owns |
|-------|--------------|
| `nat-types` | `ZoneId`/`CoreType`/`Status`/`Verification` + the **Q16.16** type |
| `nat-provenance` | the trace schema, its deterministic hash, the canonical merge decision, decision-faithful replay |
| `nat-mcp` | the non-learned MCP harness state machine (the safety story) |
| `nat-sidecar` | the `.nat.json` zone-graph that wraps a GGUF/ONNX container |
| `nat-core` | the forward pass: zones, router, gather, **deterministic merge**, **pluggable cores** |
| `nat-candle` | **Candle-backed cores (CPU, GPU-ready)** — the L1 training stack |
| `nat-data` | the data pipeline INGEST→…→MANIFEST: **quality scoring**, zone tagging, dedup, deterministic shards |
| `nat-train` | the **reproducibility floor** (config hash, hardware, rerun command) + the settlement contribution type |
| `nat-eval` | routing-differentiation (H-02) + faithfulness (H-03) harness |
| `nat-ablation` | the **H-01 ablation harness** (partitioned vs equal-param dense) |

`cargo test --workspace` count at handoff time: **87 tests, clippy clean, fmt
clean** (re-run to confirm; the number only goes up).

---

## 4. THE CRITICAL GUARANTEE — no toy cores on the DGX

The L0 cores are deliberately *toy* (tiny, deterministic, dependency-light) — they
validate the architecture, they do **not** train. The real cores are Candle
(`nat-candle`). **A real run must never silently use the toy cores**, and we made
that *verifiable*, not a matter of discipline:

- `NatModel` takes a pluggable `CoreFactory`. `NatModel::l0()` uses **toy** cores;
  `nat_candle::candle_model(sidecar)` uses **real Candle** cores.
- **Every provenance trace records `trace.backend`** — `"toy-l0"` or
  `"candle-cpu"` (later `"candle-cuda"`). Inspect it to know which ran.
- `model.uses_toy_cores()` is the hard guard.
- The ablation harness calls `nat_ablation::guard_not_toy(model.uses_toy_cores())`
  and **refuses** to measure a toy-backed model.

**Before any DGX measurement, assert you are on real cores:**

```rust
let model = nat_candle::candle_model(sidecar);
assert!(!model.uses_toy_cores());          // or: nat_ablation::guard_not_toy(...)?;
assert_eq!(model.backend(), "candle-cpu"); // "candle-cuda" once you wire the GPU device
```

---

## 5. Taking it to the GPU (the actual Gate-3 work)

### 5.1 Candle on CUDA — DONE (2026-06-21)
The device swap is implemented and verified live on the GB10. The cores no longer
hardcode `Device::Cpu`; device selection lives in one place,
`nat-candle/src/device.rs`:

- `device()` returns CPU by default, or the first CUDA device when the crate is
  built with the `cuda` feature (`Device::cuda_if_available(0)`, fail-honest CPU
  fallback). The cores and the trainer read it; nothing else changed, because the
  op graph (matmul, softmax, the SSM lower-triangular matmul) is device-agnostic.
- `backend_label()` is **honest by construction** — derived from the device that
  actually came up, so `trace.backend` records `"candle-cuda"` only on a real GPU
  run and `"candle-cpu"` otherwise. It can never claim a GPU run that did not
  happen (the §4 "record reality" guarantee, extended to cpu-vs-cuda).
- The `cuda` cargo feature wires `candle-core/candle-nn` CUDA on. Off by default so
  the standard build + CI stay CPU-only.

**Building the GPU path (two non-obvious toolchain facts):**

1. **CUDA version.** candle 0.8.4 pins cudarc 0.13.9, whose build script
   hard-rejects any CUDA toolkit **newer than 12.8**. A box with only CUDA 13
   cannot build the GPU path. Install the 12.8 toolkit side-by-side (does **not**
   touch the driver): `sudo apt-get install -y cuda-toolkit-12-8`.
2. **GPU arch.** nvcc 12.8 knows `sm_120` but not the GB10's `sm_121` (needs nvcc
   ≥ 12.9). Compile virtual `compute_120` PTX (`CUDA_COMPUTE_CAP=120`) and let the
   CUDA-13 driver JIT it to sm_121 at load.

Both are encapsulated in **`scripts/dgx-gpu.sh`**:

```sh
scripts/dgx-gpu.sh build    # cargo build -p nat-candle --features cuda
scripts/dgx-gpu.sh test     # GPU test suite
scripts/dgx-gpu.sh probe    # asserts the GPU device is actually live (candle-cuda)
```

(The aarch64 `+fp16` rustflag that `gemm-f16` needs to build at all is supplied
automatically by `.cargo/config.toml`.)

### 5.2 The H-01 ablation, for real (the bet)
`nat-ablation` runs partitioned-vs-dense Candle arms under the ADR-0005 protocol
(equal-params enforcement, identical training, capability-per-param, repro hash,
no-toy guard). **Done since 2026-06-21 (steps 1, 3, 4):**

- Runs **on the GPU** — `run_ablation` reads `nat-candle`'s device source of truth
  and records the real backend; `scripts/dgx-gpu.sh run -p nat-ablation --features
  cuda --example ablation` runs it on the GB10 (`candle-cuda`).
- **`AblationConfig::scaled()`** is the larger config (in=256, out=128,
  dense_hidden=1024, 2000 steps), param-matched to ~0.1%.
- **Multi-seed averaging** — `run_ablation_seeds(cfg, &seeds)` returns a
  `MultiSeedReport` (mean cap/param + holds-fraction). Init is **reproducible** via
  a seeded PRNG (`seeded_linear`), because candle's CPU backend cannot `set_seed`.
- First scaled GPU read (5 seeds): **H-01 HOLDS** on the mean (partitioned 1.66e0 ≥
  dense 1.55e0), holds on 4/5 seeds. *Necessary, not final* — see the caveat below.

**Still open to make the verdict conclusive (step 2 — gated on backlog item #4):**

2. Replace the structural-analog `PartitionedArm` (independent zone projections →
   concat → head) with the **full `NatModel`** (real Candle cores + routing +
   merge), and `DenseArm` with a real equal-param dense transformer. This needs the
   trainable end-to-end zone pass (item #4) to exist first. Until then the harness
   measures the partitioning *structure* at scale, not the full model.

Keep the **ADR-0005 protocol** (`.agentile/planset/decisions/ADR-0005`): identical
token budget, data, seed, optimizer, compute — only the partitioning differs. The
harness **refuses** unequal-params runs; don't bypass it. **If partitioned < dense
at equal params, H-01 is refuted — say so and change course.**

### 5.3 Real training data
`nat-data` is the pipeline. Feed it real corpora (`RawDoc`s with permissive
licenses — it screens and quarantines non-permissive ones). It emits
deterministic, manifested shards and a corpus **`aggregate_quality`** score. That
score is the `data_quality` term in the reward seam (§7). The quality scorer in
`nat-data/src/quality.rs` is L0 heuristics — upgrade it with model-based filters
at scale (Data Ops §4).

---

## 6. What is NOT done yet (your backlog, in priority order)

1. ~~**GPU device swap** in `nat-candle` (§5.1)~~ — **DONE 2026-06-21**, verified
   live on the GB10 (`candle-cuda`). See §5.1 + `scripts/dgx-gpu.sh`.
2. **Real H-01 ablation** (§5.2) — *the bet*. **Partially done 2026-06-21**: runs
   on the GPU at scale with multi-seed averaging + reproducible init (HOLDS, 4/5
   seeds). Remaining: swap in the full `NatModel` arm — gated on item #4 below.
3. **WP-1.4 — GGUF `FlattenedDense` export** + sidecar (`nat-sidecar::ExportKind`).
   Retires the "runs opaquely in Ollama" claim (critique #7). Candle is
   GGUF-native, so this is a clean fit.
4. **A real end-to-end training loop** wiring gradients through the whole zone
   pass (today `nat-candle` proves the *stack* with a tiny head; the full
   trainable `NatModel` is the L1 build).
5. **TLC** — run the three `formal/*.tla` modules through TLC (needs a JRE; was
   not run in bootstrap). This is the open Gate-1 item `g1-formal`.
6. **Federated proof** (Gate 4) — multi-node signed gather; on-chain provenance.

Gate/exit criteria are machine-readable in `.agentile/planset/gates.yaml`. Open
hypotheses are in `.agentile/planset/hypotheses.md` (**H-01 is the load-bearing
bet**).

---

## 7. How NAT fits the federation (economics)

NAT does **not** implement reward settlement. It emits, per training step, a
`nat_train::StepContribution { compute_metered, data_quality, tokens,
provenance_hash }` and a deterministic `reward_weight = compute_metered ×
data_quality`. **`citrate-compute-pool`** (a sibling federation repo that already
ships a compute marketplace + tokenomics + settlement) converts that weight into
participant payout. The interface is `docs/SETTLEMENT_SEAM.md`. Do not reinvent
settlement in this repo (ADR-0007).

Participant economic advantage = **compute contributed × data quantity/quality** —
NAT scores it, compute-pool settles it.

---

## 8. Decisions and critiques you should not relitigate

- **ADRs** (`.agentile/planset/decisions/`): 0001 hybrid routing · 0002 SSM in
  temporal zones · 0003 provenance-as-output · 0004 sidecar · 0005 the H-01
  baseline protocol · 0006 decision-faithful vs bit-faithful · 0007 integrate with
  compute-pool · 0008 zone staging · 0009 L0 numerics · **0010 Candle as the L1
  framework**.
- **The seven Gate-1 review critiques and their remediations:**
  `PLANSET/08_CRITIQUE_AND_REMEDIATIONS.md`. Extend these; don't re-argue them.

---

## 9. CI and verification

GitHub Actions is currently blocked by an **enterprise Actions budget** issue (org
cap was raised; the enforcement clock lags). It will go green on its own once the
block lifts; the workflow config in `.github/workflows/ci.yml` is correct.

**To verify CI yourself right now**, independent of GitHub:

```sh
scripts/ci-local.sh      # runs fmt + clippy + tests + cargo-deny in rust:1.96 Linux container
```

Needs a Docker engine (`brew install colima && colima start`, or Docker Desktop).

---

## 10. Where to read more (in order)

1. `README.md` — orientation.
2. `PLANSET/00_OVERVIEW.md` → the numbered planset (master plan, architecture,
   formal scaffold, data ops, research method, design brief, sprints, critiques,
   journal).
3. `.agentile/AGENT_ENTRY.md` — the rules of the house (red-test-first,
   determinism, honest posture).
4. `.agentile/planset/gates.yaml` + `hypotheses.md` — what must be true to advance.

## 11. Owners / contact

- Maintainer of record: Larry Klosowski (@saulbuilds).
- Security: see `CitrateNetwork/.github` SECURITY.md.
- This repo is **Tier-1** (`AUDIT_TIER.md`): full audit before first stable tag.

---

*Honest posture is the standing discipline here. If the bet (H-01) fails, the
right move is to say so and change course. The scale ladder exists precisely so
you can find that out cheaply before committing the full run.*
