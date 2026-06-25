---
created: 2026-06-22T00:00:00Z
branch: docs/data-s1-hermes-plansets
author: Larry Klosowski (@SaulBuilds) + Claude Opus 4.8 (1M context)
status: active
---

# Research Loop — INTENT

The single, append-only source of truth for **what we want NAT to learn next** and
**what data to gather**. The owner adds intent here each day; Hermes (HERMES-S1) and
Claude execute against it; the daily standup (below / in the Logseq journal) reports
back. The loop closes weekly when the growing corpus retrains the model and the
H-01/H-02 read updates the next day's intent.

## How to use (daily, 2 minutes)

1. **Add a dated intent block** under "Intent log" — domains to grow, what NAT is
   weak at, sources to approve or avoid, any priority shift.
2. Hermes reads the latest intent, runs its cycle, and appends a **standup** under
   "Standups".
3. Approve/deny anything in the **approval queue** (new source domains, licenses).
4. Weekly: retrain + re-run H-01/H-02 on the grown corpus; record the number; let it
   shape tomorrow's intent.

## Standing priorities (edit as they change)

- Seed the data-rich zones **{HP narrative, PF reasoning/math, CX code}** (ADR-0008).
- Mirror the eval battery: math · narrative · code · sensory.
- **The values spine** (see `READING_LIST.md`): rules, language, meaning
  (Wittgenstein rule-following + private language → "the rules of the room"), logic
  (Boole → Frege → Russell → Belnap), computation (Turing → Church → Shannon), the
  craft of code (SICP, the Rust Book, permissive repos), and expression (Strunk,
  Whitman, Montaigne). The goal: a good coder with great logic, creative and
  expressive, who follows the rules of the room they are in.
- Permissive licenses only (fail-closed gate). Provenance immutable. Copyrighted
  ideas enter only via authored CC0 explainers (we own the framing).

## Current bottleneck (from the latest result)

- 2026-06-22 (updated): the **overfit is resolved by volume**. With ~170K tokens of
  PD text the byte-LM holds **4.05 bits/byte on held-out, flat across 8 epochs** (no
  climb-back). Next limiter: corpus *breadth + size* — add code (CX) and more
  reasoning/logic, then re-run the **conclusive H-01/H-02 on real data** (DATA-S1
  WP-D6). Connectors beyond Gutenberg (HF, permissive code) are the path to volume.

## Intent log

> Append a block per day. Newest at the top.

### 2026-06-22 — BPE tokenizer (WP-D5) — ~31% better bits/byte
- `nat-data::bpe::Bpe` (byte-level, deterministic, dep-free) + `nat-corpus train-bpe`.
  Compression: **1.99 bytes/token @ vocab 1024**, 2.38 @ 4096.
- BPE autoregressive LM (vocab 1024, 128K params): held-out **2.37 bits/byte** vs the
  byte autoregressive LM's 3.42 — a **~31% reduction** (each position covers ~2 bytes).
  bits/byte is the fair metric (normalizes out the vocab). BPE is the scalable choice
  toward L2.
- **Next:** model-based quality gate (WP-D5 part 2); BPE at larger vocab/scale.

### 2026-06-22 — scale ladder toward L2 (WP-D11) — the architecture scales
- `byte_lm{,_medium,_large}` configs + `scale_ladder` example. On the 1.12M-token
  corpus (held-out bits/byte): **S 20718p 3-zone 4.097 → M 56534p 3-zone 4.054 →
  L 114956p 5-zone 3.953**. Monotonic improvement with size; the 5-zone L rung (first
  real-data training of the SM/CB **SSM** zones, ADR-0008) is best.
- Evidence that the zone architecture scales on real data → justifies L2 compute.
- **Next toward L2 (architectural):** WP-D7 per-position autoregressive LM (predict at
  every position, not one byte per fixed context — the efficiency step), then BPE
  (WP-D5) + code-normalize (WP-D8), then bigger + committed compute.

### 2026-06-22 — CONCLUSIVE H-01 on real data (WP-D6) — the bet HOLDS
- `run_real_corpus_ablation`: real NatModel vs equal-param dense transformer
  (20718≈20701), both **mini-batch-trained on the 1.12M-token corpus**, held-out
  cap/param, 5 seeds (GPU).
- **H-01 HOLDS, 5/5 seeds**: NAT held-out loss 2.88–2.91 < dense 2.97–2.99 —
  partitioning beats the dense baseline per parameter on real text. (Synthetic was a
  marginal 3/5; real data is decisive.) gates.yaml g3-h01 → **met**; H-01 → supported.
- Caveat: small byte-LM 3-zone scale (~20K params). **Next**: H-02 held-out, then
  scale (bigger model + BPE + WP-D8 code-normalize) toward the L2 read.

### 2026-06-22 — mini-batch loop (WP-D10) — corpus now exploited
- `NatTrainModel::train_minibatched` (seeded shuffle + `index_select` batches). The
  loop now consumes **160K train windows** of the 1.12M-token corpus (was a fixed
  24K-window full-batch slice).
- Result (GPU): held-out **4.02 bits/byte, monotonically improving across 8 epochs,
  no overfit** (4.21→4.13→…→4.017). The full-batch slice had overfit (2.95→3.04).
- **The data is now used.** Bottleneck cleared. Next data lever: more volume +
  WP-D6 (re-run the conclusive H-01/H-02 on real data with this loop).

### 2026-06-22 — values-spine PD fetch (Claude, manual; pre-Hermes)
- **Sourced** (PD, `scripts/fetch-values-spine.sh`): Russell ×3 (Problems of
  Philosophy, Intro to Math Philosophy, Analysis of Mind), Strunk (Elements of
  Style), Whitman (Leaves of Grass), Carroll (Alice), Montaigne (Essays — 1058
  passages), Emerson, Aesop, Austen + the 17 CC0 explainers.
- **Refined** → **1,120,711 tokens**, 779 shards, quality 0.852, **0 quarantined**.
- **Couldn't get cleanly**: Boole *Laws of Thought* (15114) + Wittgenstein
  *Tractatus* (5740) are **PDF / page-image / LaTeX only** on Gutenberg (no text/HTML)
  → need a `from-pdf` connector (DATA-S1 WP-D9). Russell IMP (41654) recovered via
  HTML-strip.
- **Trained** (GPU): best held-out **4.26 bits/byte** — BUT the loop only consumed a
  30K-window slice (full-batch), so it mildly overfit that slice rather than using
  the full 1.12M tokens. **The bottleneck has shifted from data to the training
  loop**: it needs mini-batch SGD over shuffled windows (DATA-S1 WP-D10) to exploit
  the corpus.
- **Next**: WP-D10 (mini-batch the full corpus) → then the corpus's value shows.

### 2026-06-22 — the values spine (owner)
- Add **Wittgenstein on private language + rule-following** — critical for the model
  to grasp rulemaking *without* a community or spirit behind the rule (it can't be
  done): meaning needs a public practice. This is the root of "follow the rules of
  the room." **Done as CC0 explainers** (PI is copyrighted); Tractatus (PD) to fetch.
- Add **Alan Turing** (and the computation lineage) — align the model with technology
  + design, not only philosophy. CC0 explainers done; PD primary sources (Turing
  life+70 as of 2025, Boole, Lovelace) to fetch.
- Goal restated: **a good coder, with great logic, that is creative and expressive,
  while following the general rules of the room they are in.** Curated targets +
  license status in `READING_LIST.md`.
- Next sourcing: Tractatus (5740), Boole *Laws of Thought* (15114), Strunk *Elements
  of Style* (37134), Alice (11), Whitman (1322); SICP (CC-BY-SA) + Rust Book
  (MIT/Apache) once a markdown-text connector exists.

### 2026-06-22 — kickoff
- Grow all four zones from the seed; prioritize PF (reasoning/math) + the Belnap/logic
  stream, since the architecture leans on the verification lattice.
- Approve: Project Gutenberg (PD), permissive HF text/code datasets.
- Hold for review: anything CC-BY-SA mixing, any code license not MIT/Apache/BSD.

## Approval queue

> New source domains / licenses Hermes is waiting on. Owner: allow once / deny.

- _(none yet)_

## Standups

> Hermes appends here (or in the Logseq daily journal). Newest at the top.

### 2026-06-25 — the divergence fix landed: corpus-v4 ladder HOLDS 5/5 at BOTH 4M and 8M (Claude, manual)
- **Follow-up from 2026-06-24 closed.** Added linear LR warmup (first 5% of steps) + global
  grad-norm clip at 1.0 in a shared `train_minibatched_impl` (both arms identical — strengthens
  ADR-0005), then re-ran 8M and 4M on corpus-v4 under the unified recipe (`candle-cuda`, 5 seeds).
- **8M: HOLDS 5/5.** The diverged seed 2 came back **1.989 b/byte (was 3.314)** — instability
  gone. NAT mean 1.990 vs dense 2.241; cap/param NAT 4.07e-8 > dense 3.62e-8.
- **4M: HOLDS 5/5**, NAT 1.996 vs dense 2.184 — unchanged from the pre-fix 4M (gap 0.183→0.188),
  so the recipe rescues the broken rung without distorting the stable one.
- **Coherent corpus-v4 ladder (one recipe, all seeds stable): gap widens 4M 0.188 → 8M 0.251.**
  Clean within-corpus, within-recipe read — no diverged seed, no cross-corpus confound. Honest
  mechanism: NAT's absolute loss is ~flat across the rungs while the *dense* arm degrades
  (2.184→2.241), i.e. dense fails to convert 2× params into capability where NAT holds.
- **Next**: re-run the lower rungs (248K/1M/2M) on corpus-v4 under this recipe for a fully
  unified ladder; push to 16M if the corpus ceiling allows; then committed-compute L2.

### 2026-06-24 — H-01 ladder pushed to 4M + 8M on corpus-v4 (16× volume): HOLDS, gap keeps widening (Claude, manual)
- **What**: re-ran the WP-D7 H-01 ladder (`h01_autoreg_bpe`, per-position `AutoregLm`,
  BPE-4096, NAT 5-zone vs param-matched dense Transformer) on **corpus-v4** — a strict
  superset of corpus-v3 at **30,986,801 tokens / 74,236 docs** (≈16× the 1.9M-token
  corpus-v3), built via `scripts/fetch-corpus-volume.sh` + `scripts/build-corpus-v4.sh`
  and a fresh BPE-4096 (`bpe-4096-v4.json`, 2.230 bytes/token). Genuinely on GPU
  (`candle-cuda`), 5 seeds/rung, held-out bits/byte, param-match <0.02%. Extends the
  prior ladder (248K→1M→2M, all HOLDS 5/5) to the higher rungs the new volume supports
  without overfitting. *(Re-run: the original launch died in a host crash; corpus-v4
  itself built fine and was reused — only the interrupted run's stdout was lost.)*
- **New rungs (mean held-out bits/byte, within-rung NAT vs dense):**

  | params | NAT d | NAT b/byte | dense b/byte | gap | verdict |
  |-------:|------:|-----------:|-------------:|----:|:--------|
  | 3,993,978 | 295 | 2.000 | 2.183 | 0.183 | HOLDS 5/5 |
  | 7,992,811 | 476 | 2.425 | 2.631 | 0.206 | HOLDS 4/5 |

- **Finding**: H-01 holds at both new rungs and the NAT-over-dense gap **keeps widening**
  across the whole ladder (2M 0.141 → 4M 0.183 → 8M 0.206). The 8M figure *includes* a
  diverged seed dragging the NAT mean up; **excluding it the 8M gap is ~0.42 b/byte**
  (4 stable seeds: NAT 1.97–2.50 vs dense 2.48–2.70). Zone partitioning's per-parameter
  advantage continues to grow with scale on a 16×-larger corpus — the direction the L2
  bet needs, now confirmed at 4× the prior param scale.
- **The 8M caveat (honest)**: 8M **seed 2 diverged** — NAT 3.314 b/byte, *worse* than its
  own dense arm (2.649) and ~1.3 above the other NAT seeds. That is an **optimization
  failure on one seed, not the architecture losing**: the fixed `lr=0.003` is too hot at
  d=476. **Follow-up: add LR warmup + grad-clip to `train_minibatched`, then re-confirm
  the 8M rung at 5/5.**
- **Honest scope (unchanged)**: still a scale-UP toward L2, NOT L2. Only 2 new points,
  ≤8M params; at BPE-4096 the embedding+readout dominate the budget, so the hold is a
  per-parameter signal in the cores, not a whole-model claim. True L2 (~10B, committed
  compute, gate `g5-l2`) stays owner-gated. **corpus-v4 is gitignored — local-only; back
  it up** (it cost a multi-hour rebuild after the crash).
- **Next**: LR-stability fix → re-confirm 8M 5/5; push the ladder to 16M if the corpus
  ceiling allows (held-out still improving at 8M, so likely room); then committed-compute
  L2 read.

### 2026-06-23 — H-01 on the WP-D7 architecture at scale: HOLDS, gap WIDENS (Claude, manual)
- **First H-01 read on the architecture we actually intend to scale.** Every prior H-01
  used the single-output byte-LM (`NatTrainModel`, vocab 256, ~20K params). This tests the
  **per-position autoregressive LM** (`AutoregLm`, WP-D7) on **BPE-4096**, NAT 5-zone
  (SM/CB SSM + HP/PF/CX attention) vs a new **param-matched per-position dense Transformer**
  (`AutoregDenseLm`: causal attention + FFN, no partitioning; identical embedding+readout).
  Genuinely on GPU (`candle-cuda`, ~92% util). 5 seeds, held-out bits/byte, param-match <0.02%.
- **Size ladder (corpus-v3, BPE-4096, mean held-out bits/byte):**

  | params | NAT d | NAT b/byte | dense b/byte | gap | verdict |
  |-------:|------:|-----------:|-------------:|----:|:--------|
  | 248,235 | 28 | 2.086 | 2.110 | 0.024 | HOLDS 5/5 |
  | 1,005,603 | 100 | 1.890 | 1.996 | 0.106 | HOLDS 5/5 |
  | 1,992,978 | 175 | 1.845 | 1.986 | 0.141 | HOLDS 5/5 |

- **Finding**: H-01 **holds 5/5 at every scale**, and the NAT-over-dense gap **WIDENS with
  size** (0.024 → 0.106 → 0.141 bits/byte). Zone partitioning's per-parameter advantage
  *grows* as the model scales — the direction the L2 bet needs. At ~50× the L1 param scale
  the bet not only survives the per-position + BPE architecture, it strengthens.
- **Honest scope**: a scale-UP toward L2, NOT L2. True L2 (~10B, committed compute, gate
  `g5-l2`) stays owner-gated. Only 3 points, all ≤2M params on ~788K BPE tokens (24k train
  windows) — a widening trend, not a 10B extrapolation; the 2M point is near this corpus's
  honest ceiling (held-out still improving, so not yet overfit-bound, but data-limited). At
  BPE-4096 the embedding+readout dominate, so partitioning governs a minority of params — the
  hold is a per-parameter signal in the cores. **Next data lever is VOLUME** (more tokens) to
  push the ladder further without overfitting; then committed-compute L2.
- **WP-D7 status**: the per-position LM was already built (PR #25); this *exercises and
  validates* it at scale with the settled vocab 4096 and gives it its H-01 baseline.

### 2026-06-23 — param-matched vocab sweep + the GPU was never on (WP-D5, Claude, manual)
- **Param-matched sweep** (`param_matched_bpe_sweep` example, **genuinely on GPU** —
  `is_cuda=true` verified): fix a ~500K param budget, binary-search width `d` per vocab so
  every model has equal params; only the embedding-vs-compute split varies. Held-out
  bits/byte (8 epochs, 24k/6k):

  | vocab | d | params | bytes/tok | bits/byte |
  |------:|--:|-------:|----------:|----------:|
  | 1024 | 135 | 498,232 | 1.970 | 2.351 |
  | 2048 |  95 | 500,896 | 2.216 | 2.236 |
  | 4096 |  56 | 501,323 | 2.433 | 2.180 |
  | 8192 |  29 | 493,858 | 2.624 | 2.157 |

- **Finding**: at *equal params*, bits/byte still falls monotonically with vocab — so the
  tokenizer effect is **real**, not just the param-count confound. BUT diminishing returns
  are sharp: 1024→2048 buys 0.116, 2048→4096 0.055, **4096→8192 only 0.023**. The knee is
  ~4096 (mirrors the compression curve). The `d` column shows the tradeoff: vocab 8192
  starves its cores to a 29-wide model to feed the embedding table — near the point where
  shrinking `d` starts to cost more than the tokenizer buys. **Recommended default: vocab
  ~4096** — ~95% of the bits/byte benefit, balanced compute width.
- **OPERATIONAL — prior "GPU" runs were silently on CPU.** `Device::cuda_if_available`
  returns CPU on any CUDA error and candle falls back without complaint; ollama had two
  models pinned (`qwen2.5:72b` 48 GB + `llama3.1` 5.5 GB) holding ~53 GB of the GB10's
  unified pool, so **CUDA context creation OOM'd → CPU fallback**. `nvidia-smi` looked idle
  (1% util) because the memory was *reserved, not computing*. After `ollama stop` (both),
  `is_cuda=true`, util 1%→92%, power 15W→50W. **Correction**: the same-day vocab-1024 and
  vocab-8192 BPE-LM runs below ran on **candle-cpu**, not GPU as their entries say — the
  bits/byte numbers stand (same F32 candle ops, device-independent to ~3 dp) but the device
  label was wrong. Lesson: run `scripts/dgx-gpu.sh probe` (it asserts `is_cuda`, panics on
  fallback) BEFORE claiming a GPU run; free the unified pool first.
- **Next**: per-position autoregressive LM (WP-D7); computation-lineage PD primaries.

### 2026-06-23 — BPE vocab sweep on corpus-v3 + batched-eval OOM fix (WP-D5, Claude, manual)
- **Vocab sweep, compression** (bytes/token on corpus-v3): **1.97 @1024 → 2.43 @4096 →
  2.62 @8192.** Diminishing returns: the 4× step (1024→4096) bought +0.46 B/tok; the 2×
  step (4096→8192) only +0.19 (tokens fell just 7.3% for 4096 extra merges). On this
  corpus the knee is ~4096; 8192 is into the rare-symbol long tail (code identifiers /
  Scheme tokens that don't repeat enough to earn a slot).
- **BPE-LM held-out bits/byte** (GPU path, 8 epochs, 24k/6k split):
  - vocab 1024 (127,699 params): **3.106 → 2.505**, monotonic, no overfit.
  - vocab 8192 (822,995 params): **2.463 → 2.096**, monotonic, no overfit.
- **Honest read (confounded)**: 8192's 2.096 < 1024's 2.505, but the vocab-8192 LM is a
  **6.4× bigger model** — the embedding + output tables scale with vocab (695K of the
  extra params are vocab-tied). So the lower bits/byte is mostly *more parameters*, not a
  cleaner tokenizer; you'd have to hold params fixed to isolate the tokenizer effect. The
  un-confounded read stays "the recipe descends monotonically, overfit-free."
- **Bug fixed**: the BPE-LM eval did one full-val forward (`loss_on`), materializing a
  `(6000, 64, vocab)` logit tensor — ~12.6 GB at vocab 8192, which **OOM'd the GPU and
  took the box down**. Added `AutoregLm::loss_on_batched` (64-seq minibatched eval, row-
  weighted mean = exact same number; unit-tested vs `loss_on`); example now uses it.
  Also: these small models are CPU-bound in this candle build — the 822K run took ~2h.
- **Next**: per-position autoregressive LM (WP-D7); param-matched vocab comparison if the
  tokenizer effect needs isolating; computation-lineage PD primaries (Lovelace, Turing).

### 2026-06-22 — BPE retrained at corpus-v3 scale (WP-D5, Claude, manual)
- **Retrained the BPE tokenizer on `corpus-v3`** (1.91M tokens / 5064 docs / 12.03 MB),
  training on the exact post-pipeline shard text (reconstructed the v3 RawDoc JSONL from
  the 1688 shards — network-free, byte-exact: 5064 docs, 1,914,943 tokens reproduced).
  Compression: **1.97 bytes/token @ vocab 1024** (was 1.99), **2.43 @ 4096** (was 2.38).
- **BPE autoregressive LM on `corpus-v3`** (GPU `candle-cuda`, vocab 1024, 127,699 params,
  seq_len 64, 24k/6k split, 8 epochs): held-out **3.106 → 2.505 bits/byte, monotonic, no
  overfit climb-back**. BPE + LM both encode the same v3 shards (self-consistent).
- **Honest read**: the 4096 ratio *rose* (2.38→2.43) and the LM bits/byte is **+0.135 vs
  the prose-only 2.37** — both the harder distribution, not a regression. corpus-v3 carries
  Rust + SICP **Scheme** code; a larger merge budget spreads thinner and code raises the
  entropy floor, exactly as the H-01 grow noted. bits/byte is **not comparable across
  different corpora** — the durable claim is "the BPE recipe scales cleanly to the harder
  corpus (monotone descent, no overfit)," not "the model got better."
- **Next**: BPE at larger vocab/scale on v3; per-position autoregressive LM (WP-D7);
  computation-lineage PD primaries (Lovelace, Turing).

### 2026-06-22 — SICP + H-01 re-confirmed on the grown corpus (Claude, manual)
- **Sourced**: **SICP** (Abelson & Sussman) from `sarabander/sicp` — the book HTML is
  explicitly **CC-BY-SA-4.0** (owner approved the CC-BY-SA fetch). Tag-stripped the 39
  section xhtml → `from-text` → **461 passages**. License-tagged CC-BY-SA-4.0; the
  fail-closed gate accepted it (0 license quarantines).
- **Refined**: folded SICP into `corpus-v3`: **1,914,943 tokens** (was 1.70M; base was
  1.12M) / 5064 docs / 1688 shards / aggregate_quality 0.827. Quarantined 315
  (exact_dup ×264, pii ×43, near_dup ×4, too_short ×4).
- **Re-ran the CONCLUSIVE H-01 ablation on `corpus-v3`** (GPU `candle-cuda`, real
  NatModel vs equal-param dense, params 20718≈20701, held-out cap/param, 5 seeds):
  **H-01 HOLDS, 5/5 seeds.** Mean cap/param nat **1.575e-5** > dense **1.537e-5**;
  per-seed NAT loss 3.058–3.074 < dense 3.138–3.148.
- **Why this matters (honest)**: the original decisive read was on a 1.12M prose-heavy
  corpus. corpus-v3 is bigger *and harder* — code (Rust Book + 3 crates) and SICP raise
  the entropy (losses rose from ~2.9 to ~3.1, as expected). The **NAT-over-dense gap
  survives the harder distribution**, so the hold is not an artifact of easy/prose-only
  text. Still L1 small scale (~20K params); **L2 remains the open question** — a larger
  run could still refute, and we'd say so.
- **Next**: BPE on corpus-v3; computation-lineage PD primaries (Lovelace, Turing);
  scale the ablation toward L2 when compute is committed.

### 2026-06-22 — code & craft cycle: the CX zone (Claude, manual; pre-Hermes)
- **Intent addressed**: the latest bottleneck — *"add code (CX) and more reasoning/
  logic"*. Pillar III (a good coder) was the most under-served zone; this cycle grows it.
- **Sourced** (permissive, no approval needed — MIT/Apache):
  - **The Rust Book** (`rust-lang/book`, MIT/Apache) — markdown prose on the craft +
    idioms of code; the literal "rules of the room" for a coder → `from-text` (550 passages).
  - **Idiomatic crates** `dtolnay/anyhow`, `rust-itertools/itertools`, `serde-rs/serde`
    (all MIT/Apache) — real Rust source for the CX lexical signal → `from-code`
    (37+76+237 files → 1546 passages).
- **Refined**: combined with the values-spine inputs → one unified corpus (`corpus-v2`):
  **1,698,676 tokens** (was 1.12M; **+52%**) / 4615 kept docs / 1539 shards /
  aggregate_quality **0.831**. Quarantined 303 (exact_dup ×264 — code boilerplate/
  headers; pii_detected ×32 — emails in code; near_dup ×4; too_short ×3). **0 license
  quarantines — the fail-closed allow-list passed every source clean.**
- **Trained** (GPU `candle-cuda`, byte-LM 3-zone, 20718 params, 160K/40K split):
  held-out **4.442 → 4.242 bits/byte across 8 epochs, monotonic, no overfit** (uniform 8.0).
- **Honest read (H-01 in view)**: 4.242 is **+0.22 bits/byte vs the prose-only 1.12M
  corpus's 4.02** — *expected and not a regression*. Code is higher-entropy than prose for
  a tiny byte-LM, and bits/byte is **not comparable across different corpora** (the held-out
  distribution is now broader/harder). The honest signal is "trains cleanly, monotonic, no
  overfit on a 52%-bigger, genuinely harder distribution." A true H-01 re-read means
  re-running the **NAT-vs-dense ablation on `corpus-v2`** — that is the next rung, not this
  daily grow.
- **Built**: `scripts/fetch-code-craft.sh` (recipe committed; data gitignored; Hermes-
  automatable, capsules corpus-fetch/normalize).
- **Next**: re-run the conclusive H-01 ablation on `corpus-v2`; BPE on the grown corpus;
  computation-lineage PD primaries (Lovelace, Turing); SICP (CC-BY-SA, on the allow-list)
  pending owner confirmation of the kickoff "hold CC-BY-SA for review".

### 2026-06-22 — first real cycle (Claude, manual; pre-Hermes)
- **Sourced**: Gutenberg (PD) — 1342 Pride & Prejudice (narrative/HP), 5827 Problems
  of Philosophy (logic/language/PF), 41654 Intro to Mathematical Philosophy (math/PF).
- **Refined**: `nat-corpus from-gutenberg` → 404 passages → pipeline → **403 kept,
  135 shards, 170,379 tokens, aggregate_quality 0.854**, 1 quarantined (PII false-pos).
- **Trained**: byte-LM 3-zone on the corpus (GPU) → held-out **4.05 bits/byte**, flat
  across 8 epochs — the seed's overfit is **gone**.
- **Built**: the Gutenberg connector (`nat-data::gutenberg`) + `from-gutenberg` CLI
  (caught a CRLF-splitting bug → regression-tested).
- **Next**: a code (CX) connector + more logic/reasoning volume → re-run conclusive
  H-01/H-02 on real data.
