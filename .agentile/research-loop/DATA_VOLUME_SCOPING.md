# Data-Volume Scoping — can we reach billions of license-clean tokens?

**Sprint:** SCALE-S1 WP-S4 · **Date:** 2026-06-25 · **Author:** Claude Opus 4.8 (1M)
**Question:** A ~500M-param from-scratch LM wants **~2.5–5B tokens** (≈5–10 tokens/param,
the floor to not be trivially data-bound). corpus-v4 is **31M tokens**. Is the
**~100–300× gap** closable with *permissively-licensed* data, and what does each source cost
to ingest?

> **Estimates, not measurements.** Token counts below are order-of-magnitude, derived from
> public corpus sizes and our own BPE-4096 ratio (**2.23 bytes/token** on corpus-v4 — code/
> prose mixed). The real number is whatever our pipeline counts after license + dedup +
> quality gates. WP-S4's actual deliverable is a **counting run** (ingest a sample of each
> source, report kept tokens) before we commit storage. Treat the verdict as "is the order
> of magnitude there," which it robustly is.

## Per-source inventory

| Source | License | Est. clean tokens | Connector status | Work to ingest at volume |
|--------|---------|------------------:|------------------|--------------------------|
| **Project Gutenberg** (full English) | PD | **~3–8B** | ✅ `from-gutenberg` (one id) + sample haul in v4 | **Full sweep**: iterate the ~60k English ids, not a few hundred; rate-limit/mirror; we already ingest cleanly, just at 1% of the catalog |
| **English Wikipedia** | CC-BY-SA | **~3–4B** | ❌ none | New `from-wikipedia` connector (HF `wikimedia/wikipedia` dump or the XML dump → wikitext strip). Owner already set the CC-BY-SA precedent (SICP) |
| **Permissive code** (MIT/Apache/BSD) | permissive | **10B+** available (we'd cap) | ✅ `from-code` (one repo) | A bulk path over a curated permissive set (e.g. the permissive slice of The Stack v2, or a vetted GitHub MIT/Apache list) with **per-file SPDX screening** at scale |
| **HF permissive text datasets** | CC0/CC-BY/PD | **~1–3B** | ❌ none | New `from-hf` connector (streaming `datasets`); whitelist permissive subsets (PD-books, CC-licensed corpora) |
| **US Government / legal** (PD) | PD / CC0 | **~1–3B** | partial (`from-text`) | Federal works are PD; Caselaw Access Project (CC0). Needs a fetch+normalize recipe, not new core |
| **PubMed Central OA — permissive subset** | mixed (CC-BY/CC0 only) | **~1–2B** | ❌ | High value (reasoning/PF) but **license is per-article** — only the CC-BY/CC0 subset; screening is the work, and the risk |
| arXiv | mostly non-permissive | — | — | **Excluded by default**: arXiv's default license is not on the allow-list. Only explicitly CC-BY/CC0 papers qualify — not worth the screening cost yet |

**Headline:** Gutenberg (full) + Wikipedia + a capped permissive-code slice **alone clear
the 2.5–5B target several times over.** Availability is not the constraint.

## What we actually need, by rung

| Rung | Params | Tokens @ ~10/param | Source mix that covers it |
|-----:|-------:|-------------------:|---------------------------|
| 16M | 16M | ~160M | corpus-v4 (31M) is short → **corpus-v5** (full Gutenberg ≈ several hundred M) |
| 64M | 64M | ~640M | corpus-v5 (Gutenberg full + Wikipedia) |
| 128M | 128M | ~1.3B | + permissive code slice |
| 256M | 256M | ~2.6B | + HF permissive + gov/legal |
| **512M** | 512M | **~5B** | all of the above, deduped, quality-gated |

So the ladder doesn't need 5B on day one — each rung's data lands incrementally. **corpus-v5
(≈300–800M tokens) unblocks 16M→64M immediately**; the multi-billion push is staged behind it.

## The real constraints (not availability)

1. **License screening at scale must be per-doc and in-path.** At 31M tokens a post-hoc gate
   is fine; at billions it must run inside the streaming pipeline (WP-S5/S6), fail-closed, or
   one mislicensed dump poisons the corpus. The allow-list (`ALLOWED_LICENSES`) is the
   contract; the work is enforcing it per-record on heterogeneous sources.
2. **Cross-source dedup.** Gutenberg ↔ Wikipedia ↔ code share boilerplate; the same text
   recurs across HF dumps. The MinHash/LSH near-dup (commit 7618d6c) is near-linear and
   scales, but it must run **across the whole corpus**, not per-shard, or duplicates inflate
   the token count and bias the LM.
3. **Quality at volume.** The L0 heuristic scorer (`run_pipeline_with_scorer`) is a *score*,
   not a gate. At billions of tokens, low-quality web/code noise needs a fail-closed
   **model-based filter** (WP-S7) or the corpus's effective quality drops as it grows.
4. **Storage + backup.** ~5B tokens ≈ **20–40 GB** of shards. That cannot stay
   gitignored-local-only (this sprint's host crash already cost a v4 rebuild). Needs a durable
   home + restore runbook (WP-S8) **before** the big ingest, not after.

## Verdict

**Reachable, with margin.** 2.5–5B license-clean tokens is *available* from Gutenberg +
Wikipedia + permissive code without touching anything legally marginal. The program is
therefore **engineering, not sourcing**: build the bulk/streaming connectors (WP-S6), enforce
license + dedup + quality in-path at scale (WP-S5/S7), and give the corpus a durable home
(WP-S8). None of it is research-risky; all of it is buildable.

## Recommended acquisition order (cheapest high-yield first)

1. **Full Gutenberg sweep** — connector exists, pure PD, no screening risk, ~billions of
   tokens. Single highest-yield, lowest-risk move. → **corpus-v5**.
2. **Wikipedia (CC-BY-SA)** — one new connector, ~3–4B tokens, owner-precedented license.
3. **Permissive code slice** — bulk `from-code` + SPDX screening; caps at whatever we need.
4. **HF permissive + gov/legal** — fills the 256M→512M rungs; more screening per token.

## Open question for the owner

PubMed Central OA (CC-BY/CC0 subset) is high-value for the PF/reasoning zone but needs
per-article license screening. **Worth building the screening for, or defer past 512M?** And:
confirm Wikipedia CC-BY-SA is approved for bulk (the SICP precedent suggests yes, but it's a
new *domain*, so it belongs in the Hermes approval queue).
