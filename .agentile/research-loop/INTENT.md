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
