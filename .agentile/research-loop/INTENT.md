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

- 2026-06-22: byte-LM on the 4KB seed reaches **4.18 bits/byte** then **overfits** —
  the limiter is **data volume**, not architecture. Need orders of magnitude more
  permissive text across the four zones before the conclusive H-01/H-02 is decisive.

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

- _(none yet — first cycle pending HERMES-S1 WP-H6)_
