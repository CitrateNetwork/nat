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
