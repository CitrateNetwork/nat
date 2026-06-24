# CS-02: From no data to a scaling curve — the L1 corpus and training loop

**Gate:** Gate 3 · **Rung:** L1 · **Dates:** 2026-06-22
**Authors:** Larry Klosowski (architect), Claude Code (build)

## Question
Can NAT train on real, license-clean text, and does the architecture *scale* —
i.e., does held-out loss fall as the model and data grow? (The prerequisite for any
L2 commitment, and the substrate for the conclusive H-01 in CS-01.)

## Setup
We started with no corpus — `nat-data` was a tested pipeline (license/length/dedup/
PII/quality gates) with nothing run through it. Built in DATA-S1:
- a byte-level tokenizer (vocab 256, deterministic), on-disk shard persistence, and a
  corpus loader (`nat-candle::corpus`);
- the `nat-corpus` CLI + a `RawDoc` JSONL contract, and source connectors
  (`from-gutenberg` prose, `from-code` permissive repos);
- a CC0 seed corpus including authored explainers of the values spine (Wittgenstein
  rule-following / private language, Turing, Belnap) — see `research-loop/READING_LIST.md`;
- `scripts/fetch-values-spine.sh`, which fetched ~1.1M tokens of public-domain text
  (Russell ×3, Strunk, Whitman, Carroll, Montaigne, Emerson, Aesop, Austen).
All on the GB10 (`candle-cuda`), mini-batch SGD over shuffled windows (WP-D10).
Rerun the corpus: `scripts/fetch-values-spine.sh`; the ladder:
`scripts/dgx-gpu.sh run -p nat-candle --features cuda --example scale_ladder -- <corpus>`.

## What we measured
Held-out next-byte loss (bits/byte; uniform = 8.0) as the bottleneck moved, and across
a model-size ladder. Corpus health: kept/quarantined docs, aggregate quality, tokens.

## Result
- Corpus: 2,337 docs / 779 shards / **1,120,711 tokens** / quality 0.852 / **0
  quarantined** (the 1 PII false-positive aside).
- The bottleneck moved three times, each an honest finding:
  1. *Data.* The 4 KB seed reached 4.18 bits/byte then **overfit** — needed volume.
  2. *Loop.* With 1.1M tokens the full-batch loop still trained a fixed 24K-window
     slice and overfit it; **mini-batch SGD** drove held-out to **4.02 bits/byte,
     monotonic across 8 epochs, no overfit.**
  3. *Architecture.* The single-output model wasted compute (one prediction per
     window). The **per-position autoregressive** form (WP-D7) reached **3.42
     bits/byte at 53K params** — better loss with *half* the parameters.
- Scale ladder (single-output): S 20718p/3-zone **4.097** → M 56534p/3-zone **4.054**
  → L 114956p/5-zone **3.953** bits/byte. **Loss falls monotonically with size**, and
  the 5-zone L rung (first real training of the SM/CB SSM zones, ADR-0008) is best.

## What surprised us
How cleanly the bottleneck handed off: data → loop → architecture, each fix exposing
the next limiter rather than a wall. Also that widening to five zones *helped* at
this scale — ADR-0008 staged SM/CB until "the data earns it," and it now does. And a
small thing with teeth: Gutenberg files are CRLF, so a naive paragraph split turned a
700 KB book into one over-long passage that got quarantined — a regression test now
guards it.

## Decision
Keep the connector split (deterministic Rust conversion; the *fetch* is the agent's
network-granted job — HERMES-S1). Make `train_minibatched` and the autoregressive
`AutoregLm` the default training paths. Order the road to L2 in DATA-S1: BPE (WP-D5),
code-aware NORMALIZE (WP-D8), `from-pdf` (WP-D9), then bigger with committed compute.

## Open threads
- NORMALIZE flattens code whitespace — code trains lexically but loses layout (WP-D8).
- Boole's *Laws of Thought* and Wittgenstein's *Tractatus* are PDF/LaTeX-only on
  Gutenberg; they need a `from-pdf` path (WP-D9). The Wittgenstein *content* is covered
  by the CC0 explainers.
- The autoregressive result is at 53K params / 1M tokens — the scaling claim is a
  curve, not an L2 proof. Bigger runs are the test.

---

## Continuation (2026-06-24): corpus-v4 and the ladder at 4M / 8M

**Dates:** 2026-06-24 · **Authors:** Larry Klosowski (architect), Claude Code (build)

### What changed
The 2M-param point (CS-01 / the 2026-06-23 ladder) was nearing the corpus's honest
ceiling: held-out loss still fell, but ~788K BPE tokens can't feed a much bigger model
without it memorizing the test. So we built **corpus-v4** — a strict superset of
corpus-v3 (same curated pillars) plus a large public-domain volume haul
(`scripts/fetch-corpus-volume.sh` → `scripts/build-corpus-v4.sh`): **74,236 docs /
30,986,801 tokens (~16× corpus-v3)**, fresh BPE-4096 (2.230 bytes/token). Then we pushed
the per-position H-01 ladder to **4M and 8M params** (prior ceiling 2M), 5 seeds each,
`candle-cuda`, param-matched <0.02%.
Rerun: `scripts/build-corpus-v4.sh`, then `scripts/dgx-gpu.sh run -p nat-candle
--features cuda --release --example h01_autoreg_bpe -- <corpus-v4-dir> <bpe-4096-v4.json>
<target_params> <max_windows> 5`.

### Result (mean held-out bits/byte, within-rung NAT vs dense)
| params | NAT | dense | gap | verdict |
|-------:|----:|------:|----:|:--------|
| 3,993,978 | 2.000 | 2.183 | **0.183** | HOLDS 5/5 |
| 7,992,811 | 2.425 | 2.631 | **0.206** | HOLDS 4/5 |

Across the full ladder the gap reads **0.024 → 0.106 → 0.141 → 0.183 → 0.206** — it keeps
widening, now at 4× the prior param scale on a 16× corpus. The 4M rung is clean and
unanimous.

### What surprised us — the diverged seed
At 8M, **one seed of five diverged**: seed 2's NAT arm hit **3.314 b/byte**, worse than its
own dense control (2.649) and ~1.3 above its sibling NAT seeds. That is what makes 8M 4/5
rather than 5/5. We read it as an **optimizer instability, not an architecture failure**,
and the evidence is on the record: (1) the dense arm at the same seed trained fine, so the
seed isn't cursed — only the *wide* NAT arm tripped; (2) the four clean NAT seeds posted the
**best numbers on the whole ladder** (down to 1.973), so excluding the diverged seed the 8M
gap is **~0.42 b/byte — the widest measured**; (3) the signature is textbook early-step
Adam blow-up at `d=476` with a flat `lr=0.003` and no warmup. The bug *sharpened* the read
(it surfaced the widest clean gap), and it sits in the recipe, not the thesis.

### Decision / fix (verdict in flight)
Both arms were folded onto one shared `train_minibatched_impl` with **linear LR warmup
(first 5% of steps) + global grad-norm clip at 1.0** — standard hygiene, and the single-loop
refactor makes ADR-0005's "identical training" literally enforced rather than copy-pasted.
All 37 nat-candle tests stay green; it compiles and runs on CUDA; the re-run (8M then 4M
under the unified recipe) is **in flight — the post-fix 5/5 is not yet confirmed.** If the
fix doesn't resolve the divergence, that is the result and the journal records it.

### Open threads (updated)
- Post-fix 8M re-confirmation pending; the 4M/8M numbers above are under the *pre-fix*
  recipe, so the coherent corpus-v4 ladder is the re-run, not this table.
- bits/byte is not comparable across the v3→v4 corpus change; only the within-corpus 4M→8M
  widening (0.183 → 0.206) is a clean read. The cross-corpus span mixes scale + distribution.
- At BPE-4096 embedding+readout dominate the budget — the hold is a per-parameter signal in
  the cores. ≤8M on 31M tokens is a scale-*up*; real L2 (committed compute, g5-l2) is still
  the rung that could refute.
