---
created: 2026-06-22T00:00:00Z
branch: docs/data-s1-hermes-plansets
author: Larry Klosowski (@SaulBuilds) + Claude Opus 4.8 (1M context)
status: active
sprint: DATA-S1
---

# DATA-S1 — Real corpus: from pipeline to a decisive H-01/H-02

NAT-S2 proved the whole trainable pipeline but left two honest gaps: the H-01 hold
is **marginal on synthetic data**, and H-02 is **in-sample**. Both close the same
way — **real, growing, permissively-licensed corpus**. This sprint wires real data
into training (done) and scales it (the Hermes/loop job) until the conclusive read
is decisive.

Frame: `PLANSET/04_DATA_OPS.md` + `docs/DGX_HANDOFF.md` §5.3. Pairs with
**HERMES-S1** (the agent that collects + refines) and the **research loop**
(`.agentile/research-loop/INTENT.md`).

## Target domains (owner-set 2026-06-22)

Seed the **data-rich zones {HP, PF, CX}** (ADR-0008) and mirror the eval battery:

| Domain | Zone | Sources (permissive only) |
|--------|------|---------------------------|
| General English prose / narrative | HP | Project Gutenberg (PD), CC-BY/CC0 text |
| Math / structured reasoning | PF | permissive math/QA/proof corpora, CC0 explainers |
| Code | CX | MIT/Apache/BSD source (e.g. permissively-licensed slices of The Stack) |
| Sensory / descriptive | SM | PD literature (descriptive passages) |
| **Logic & language philosophy + Belnap FOUR** | PF/CX | SEP-adjacent CC content, PD logic texts, authored CC0 explainers |

Belnap's four-valued logic is load-bearing, not decorative: its **both/neither**
values map onto the provenance verification lattice (Pass / Fail / Unverified /
contradicted) — training on it aligns the model with the architecture's own story.

## Work packages

| WP | Subject | Acceptance | Status |
|----|---------|-----------|--------|
| WP-D1 | **Byte-level tokenizer** (vocab 256, deterministic) | encode/decode round-trip; IDs in vocab | ✅ done (`nat-data::tokenizer`) |
| WP-D2 | **Corpus persistence** (versioned on-disk shards + manifest) | write→read round-trips; manifest hash matches | ✅ done (`nat-data::persist`) |
| WP-D3 | **Corpus loader** → next-byte windows | well-formed `(ids, targets)`; valid byte IDs | ✅ done (`nat-candle::corpus`) |
| WP-D4 | **Next-byte LM objective + first real run** | held-out byte loss beats uniform on real text | ✅ done — seed corpus: **4.18 bits/byte** best (vs 8.0); overfits ~4KB → needs volume |
| WP-D5 | **BPE tokenizer + model-based quality** | learned vocab behind the same interface; perplexity/classifier quality gate (replaces L0 heuristic) | planned |
| WP-D6 | **Scale corpus → decisive H-01/H-02** | ≥ N tokens across the 4 zones; re-run conclusive ablation + held-out H-02 on real data | ✅ **H-01 done** — `run_real_corpus_ablation` (both arms mini-batched on the 1.12M-token corpus, held-out cap/param). **H-01 HOLDS 5/5 seeds** (NAT loss 2.88–2.91 < dense 2.97–2.99 at equal params) — decisive vs the synthetic 3/5. Caveat: small byte-LM scale. H-02 held-out still to do. |
| WP-D7 | **Per-position autoregressive LM** (architecture) | causal per-position next-token heads; full LM loss | planned (out of DATA-S1 if it grows) |
| WP-D8 | **Code-aware NORMALIZE** | preserve newlines/indentation so code structure survives the pipeline (CX zone) | planned |
| WP-D9 | **`from-text`/`from-markdown`/`from-pdf` connector** | ingest CC text (SICP CC-BY-SA, Rust Book) + PDF/LaTeX-only PD (Boole, Tractatus) | planned |
| WP-D10 | **Mini-batch SGD over the full corpus** | the loop samples shuffled mini-batches from all shards, not a fixed 30K-window full-batch slice — so the ~1M-token corpus is actually used | ✅ **done** — `NatTrainModel::train_minibatched` (seeded Fisher-Yates shuffle, `index_select` batches). On the 1.12M-token corpus: 160K train windows, held-out **4.02 bits/byte monotonically improving across 8 epochs, no overfit** (vs the full-batch slice's 2.95→3.04 climb). |

**Values-spine corpus built (2026-06-22):** `scripts/fetch-values-spine.sh` →
**1,120,711 tokens / 779 shards / quality 0.852 / 0 quarantined** (Russell ×3, Strunk,
Whitman, Carroll, Montaigne, Emerson, Aesop, Austen + 17 CC0 explainers). Boole +
Tractatus are PDF/TeX-only on Gutenberg (→ WP-D9). First train consumed only a 30K-window
slice (full-batch) → 4.26 bits/byte best, mild overfit on the slice — **WP-D10 is now the
gate to exploiting the corpus.**

**Values spine (owner intent 2026-06-22):** beyond the domain table, the corpus
targets a curated set — Wittgenstein (rule-following / private language → "the rules
of the room"), Turing + the computation lineage, the craft of code (SICP, the Rust
Book, permissive repos), and expressive writing (Strunk, Whitman). Full list + license
status: `research-loop/READING_LIST.md`. Copyrighted ideas enter only via authored CC0
explainers (6 added to the seed corpus: Wittgenstein ×3, Turing ×2, code-as-rule-following).

## Sources & licensing posture (hard rules)

- **License allow-list is a fail-closed gate** (`nat-data::ALLOWED_LICENSES`): CC0,
  CC-BY, CC-BY-SA, MIT, Apache-2.0, BSD-3-Clause, public-domain. Anything else is
  **quarantined, never trained on** — already enforced.
- **Provenance is immutable**: every doc records `source`, `license`, `fetch_date`,
  `raw_hash`. Raw is never mutated; drops are quarantined with a reason.
- **PII is a gate, not a warning** — quarantined.
- New source *domains* require human approval (the Hermes approval queue) before
  Hermes may fetch from them at scale.

## Exit criteria

- [x] Real text trains end to end on the GB10 (byte-LM, held-out beats uniform).
- [ ] BPE + model-based quality (WP-D5).
- [ ] Corpus large + balanced enough that the **conclusive H-01 is decisive** (not a
      3/5-seed marginal) and **H-02 holds out-of-sample** (WP-D6).
- [ ] Every shard's provenance + license auditable from the manifest.

## Honest posture

The seed result is a *path proof*, not a capability claim — it overfits 4KB. No
real-data H-01/H-02 verdict is claimed until WP-D6. If, at volume, partitioning
still doesn't beat dense per-param, H-01 is refuted — say so and change course.

## Close-out

- REPORT.md with the real-data H-01/H-02 numbers; update `gates.yaml` (g3-h01) and
  `hypotheses.md` (H-01/H-02) accordingly; move to `completed/`.
