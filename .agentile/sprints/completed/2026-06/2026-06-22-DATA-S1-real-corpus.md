---
created: 2026-06-22T00:00:00Z
closed: 2026-06-22T00:00:00Z
branch: docs/data-s1-hermes-plansets
author: Larry Klosowski (@SaulBuilds) + Claude Opus 4.8 (1M context)
status: completed
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
| WP-D5 | **BPE tokenizer** (+ model-based quality, deferred) | learned vocab behind the same `encode`/`decode`/`vocab_size` interface | ✅ **BPE done** — `nat-data::bpe::Bpe` (byte-level, deterministic, dep-free; train/encode/decode/save/load) + `nat-corpus train-bpe` CLI. On the corpus: **1.99 bytes/token @ vocab 1024**. BPE autoregressive LM: **2.37 bits/byte** vs the byte LM's 3.42 — a **~31% reduction** (more context/position). Model-based quality: `quality::NgramModel` (byte-bigram perplexity → [0,1]) + non-breaking `run_pipeline_with_scorer` hook; clean ranks above gibberish. |
| WP-D6 | **Scale corpus → decisive H-01/H-02** | ≥ N tokens across the 4 zones; re-run conclusive ablation + held-out H-02 on real data | ✅ **H-01 done** — `run_real_corpus_ablation` (both arms mini-batched on the 1.12M-token corpus, held-out cap/param). **H-01 HOLDS 5/5 seeds** (NAT loss 2.88–2.91 < dense 2.97–2.99 at equal params) — decisive vs the synthetic 3/5. Caveat: small byte-LM scale. **H-02 held-out DONE** — `nat-eval::h02_heldout` (PR #29): trained router separates prompt classes it never saw, **3.10 vs L0 2.63** on a train/held-out split of the extended battery (held-out on the prompt-class battery, H-02's natural domain). |
| WP-D7 | **Per-position autoregressive LM** (architecture) | causal per-position next-token heads; full LM loss | ✅ **done** — `nat-candle::autoreg::AutoregLm` (causal attention + causal-SSM zones over the full sequence; per-position soft merge; next-token loss). On the corpus: **3.42 bits/byte at 53K params** vs the single-output model's 3.95 @ 115K — better loss, **half the params** (each seq → seq_len−1 predictions). The L2-efficiency unlock. |
| WP-D8 | **Code-aware NORMALIZE** | preserve newlines/indentation so code structure survives the pipeline (CX zone) | ✅ **done** — `nat-data::normalize` now preserves line structure + indentation (per-line: keep indent, collapse internal ws; collapse blank-line runs; first line left-trimmed). Prose unaffected; code keeps its layout. |
| WP-D9 | **`from-text` + LaTeX-strip connector** | ingest CC/PD text + LaTeX-only PD | ✅ **done** — `nat-data::text` (passage split) + `nat-data::latex::strip` + `nat-corpus from-text --strip latex`. Live: **Boole 15114 (339 passages) + Tractatus 5740 (146)** → 188K tokens, quality 0.825. SICP/markdown ingest via the same `from-text` (no strip). |
| WP-D10 | **Mini-batch SGD over the full corpus** | the loop samples shuffled mini-batches from all shards, not a fixed 30K-window full-batch slice — so the ~1M-token corpus is actually used | ✅ **done** — `NatTrainModel::train_minibatched` (seeded Fisher-Yates shuffle, `index_select` batches). On the 1.12M-token corpus: 160K train windows, held-out **4.02 bits/byte monotonically improving across 8 epochs, no overfit** (vs the full-batch slice's 2.95→3.04 climb). |
| WP-D11 | **Scale ladder toward L2** | bigger/wider configs train on real data; loss falls with scale | ✅ **done** — `NatTrainConfig::byte_lm{,_medium,_large}` + example `scale_ladder`. On the corpus: S (20718 p, 3-zone) **4.097** → M (56534 p, 3-zone) **4.054** → L (114956 p, **5-zone incl. SM/CB SSM**) **3.953** bits/byte. The architecture scales on real data; widening to 5 zones (ADR-0008) is the best rung. |

**Toward L2 (next architectural steps, in order):** WP-D7 per-position autoregressive
LM (the efficiency step — predict at every position, not one byte per fixed context) ·
WP-D5 BPE vocab · WP-D8 code-aware NORMALIZE · then bigger still + committed compute.
The scale ladder (WP-D11) is the evidence that justifies it: loss falls monotonically
with size on real data.

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
- [x] BPE + model-based quality (WP-D5) — `nat-data::bpe` (1.99 bytes/tok @ vocab 1024)
      + `quality::NgramModel` perplexity scorer.
- [x] Corpus large + balanced enough that the **conclusive H-01 is decisive** (5/5
      seeds on the 1.12M-token corpus, not the 3/5 synthetic marginal) and **H-02
      holds out-of-sample** (held-out battery, trained 3.10 vs L0 2.63) — WP-D6.
- [x] Every shard's provenance + license auditable from the manifest (fail-closed
      `ALLOWED_LICENSES`; immutable `source`/`license`/`fetch_date`/`raw_hash`).

## Honest posture

The seed result is a *path proof*, not a capability claim — it overfits 4KB. No
real-data H-01/H-02 verdict is claimed until WP-D6. If, at volume, partitioning
still doesn't beat dense per-param, H-01 is refuted — say so and change course.

## REPORT — close-out (2026-06-22)

**Status: COMPLETE.** All eleven WPs (WP-D1..D11) delivered. The sprint's mandate —
take the pipeline from a path-proof to a **decisive** H-01 and a **held-out** H-02 on
real, permissively-licensed text — is met. The two honest gaps NAT-S2 left are closed.

### The two headline results
- **H-01 is decisive (5/5 seeds, real data).** `nat-ablation::run_real_corpus_ablation`:
  real `NatTrainModel` vs an equal-param dense transformer (20718≈20701), both
  mini-batch-trained on the 1.12M-token PD corpus (next-byte LM), capability on a
  **held-out** split. **NAT 2.88–2.91 < dense 2.97–2.99** at equal params, all five
  seeds — partitioning *beats* dense per parameter on real text. The marginal synthetic
  3/5 (NAT-S2 WP-5) is superseded. **H-01 holds; load-bearing bet stands at L1.**
- **H-02 holds out-of-sample.** `nat-eval::h02_heldout` (PR #29): the trained router,
  scored on prompt classes it never saw, still beats the L0 hand-wired baseline
  (**3.10 vs 2.63**) — at L1 it generalizes to held-out prompt classes rather than
  memorizing the training prompts. (Held-out on the prompt-class battery, which is
  H-02's natural domain; the corpus LM split is H-01's. Full-scale labeled batteries
  remain the L2 read.)

### Pipeline delivered
Byte tokenizer (D1) · persist+manifest (D2) · loader (D3) · next-byte LM (D4) · BPE
1.99 bytes/tok @ vocab 1024 + n-gram quality scorer (D5) · decisive H-01/held-out H-02
(D6) · per-position autoregressive LM, 3.42 bits/byte at half the params (D7) ·
code-aware normalize (D8) · from-text + LaTeX-strip, Boole+Tractatus (D9) · mini-batch
SGD over the full corpus (D10) · S→M→L scale ladder, 5-zone L best at 3.953 bits/byte (D11).

### Corpus state at close
Values-spine build (`scripts/fetch-values-spine.sh`): **1,120,711 tokens / 779 shards /
aggregate quality 0.852 / 0 quarantined**. Fail-closed license gate + immutable
provenance enforced. `corpus/` stays gitignored — never committed.

### Gate / hypothesis state at close
- `gates.yaml`: **g3-h01 met:true** (5/5 real), **g3-routing met:true** (held-out).
- `hypotheses.md`: **H-01 supported (real-data L1, 5/5)**, **H-02 supported (held-out L1)**.

### Carried forward (continuous, not blocking close)
- **Corpus growth** is now the research-loop / HERMES-S1 job — DATA-S1 built the
  pipeline and got the decisive read; growing volume per `INTENT.md` continues there.
- **Honest scale caveat:** all of the above is ~20K–115K params, byte/BPE level, on
  ~1M tokens — **not L2**. Whether H-01 survives BPE depth, more params, and orders of
  magnitude more data is the open L2 question. If a larger run refutes it, say so.
- Model-based quality is a *score* today (`run_pipeline_with_scorer`); wiring it as a
  fail-closed **gate** is open backlog item #4.
