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
