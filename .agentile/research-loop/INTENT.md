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
- **Logic & language philosophy + Belnap FOUR** — aligns with the provenance
  verification lattice; keep this stream flowing.
- Permissive licenses only (fail-closed gate). Provenance immutable.

## Current bottleneck (from the latest result)

- 2026-06-22 (updated): the **overfit is resolved by volume**. With ~170K tokens of
  PD text the byte-LM holds **4.05 bits/byte on held-out, flat across 8 epochs** (no
  climb-back). Next limiter: corpus *breadth + size* — add code (CX) and more
  reasoning/logic, then re-run the **conclusive H-01/H-02 on real data** (DATA-S1
  WP-D6). Connectors beyond Gutenberg (HF, permissive code) are the path to volume.

## Intent log

> Append a block per day. Newest at the top.

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
