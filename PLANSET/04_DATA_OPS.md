# Data Operations & Training Plan

**Document:** RFC-CIT-NAT-0001 / Data Ops
**Status:** Draft v0.1
**Companion to:** `00_MASTER_PLAN.md`, `02_ARCHITECTURE_SPEC.md`
**Owners (proposed):** James Paulk, Dan Heiwig

---

## 1. Decision on file (from the design session)

- One **unified** model (not zone-specialist sub-models composed). Zones are
  internal partitions; the model is trained as one.
- On-prem budget **4–6 TB**, scalable to **~100 TB** via cloud / Citrate
  storage if the data earns it.
- Prefer **open-source datasets** and **from-scratch** training. Pretrained zone
  initialization is an allowed fallback, license-cleared by counsel only.
- Federated nodes train **partitioned data toward the full model** (not
  zone-isolated). The selection criterion is "grandma-proof"... whatever is
  easiest for a node operator to run correctly.

The architecture is unified, but the *data is annotated by zone affinity* so the
router has a signal to learn from. This is the key move: we do not split the
model, we tag the data.

## 2. Honest compute posture (read this before sizing anything)

A single DGX Spark-class node is bandwidth-bound and is a prototyping and
fine-tuning device. Training a 10B model from scratch to a compute-optimal token
budget on one such node in two to three months is not realistic. The Master Plan
scale ladder is the honest path:

- **L0 (~150M):** wire up the pass on the Spark in days.
- **L1 (~1–2B):** train on the Spark in 2–4 weeks; this is where we prove routing
  differentiation and the GGUF round-trip, and build the eval harness.
- **L2 (~10B):** the real run, requiring aggregate compute beyond one Spark
  (federation + cloud burst).
- **L3:** federated cycle as a research milestone.

A rough, honest token-budget sanity check for L2: a compute-optimal 10B model
wants on the order of ~200B tokens (Chinchilla-style ~20 tokens/param), and
useful small models are often trained well past that. At ~2 bytes/token after
tokenization, ~200B tokens is ~400 GB of *token* data, drawn from a much larger
*raw* corpus after cleaning and dedup (raw-to-clean yield is often 10–30%). So
the 4–6 TB on-prem budget is plausibly enough *raw* data for a first L2 run, with
headroom to ~100 TB if we widen modality coverage or push past compute-optimal.
These are order-of-magnitude figures to size pipelines, not promises. They get
refined by the L1 eval harness.

## 3. The shape of the data

We organize the corpus by **zone affinity tag**, because the router learns to
route, and it needs differentiated training signal. Each document gets one or
more zone-affinity tags during ingestion. A document can serve multiple zones.

| Zone | What it needs | Candidate open sources |
|------|---------------|-------------------------|
| Sensorimotor (SM) | multimodal alignment: paired modality + text | open video+caption sets, open audio+transcript sets, open image+text sets |
| Cerebellar (CB) | sequential/timing data | music notation/MIDI, motion/gesture logs, game replays, rhythmic sequences |
| Hippocampal (HP) | salient narrative, dialogue, memoir | open story corpora, open dialogue datasets, long-form narrative |
| Prefrontal (PF) | reasoning chains, language | open reasoning/chain-of-thought sets, math word problems, philosophy/argument text, general web text |
| Codec (CX) | verified executable logic | permissively-licensed code, unit tests, formal-proof corpora, doc+code pairs |
| MCP/MX | tool-use traces, function-call schemas | open tool-use/function-calling datasets, API schemas |

Note: `MX` is non-learned, so its "data" is tool schemas and example traces used
to validate the harness, not to train weights. SM at v1 may stay text-heavy with
a thin multimodal slice if paired data is scarce; we widen it as the corpus
grows.

### 3.1 Concrete open-source dataset families to evaluate
General/web and reasoning text, large permissive code corpora, math and
chain-of-thought collections, dialogue and narrative corpora, multimodal
paired sets, and arXiv/academic dumps. The selection rule is: permissive license,
documented provenance, and a clean extraction path. **Every dataset gets a
license review before it enters the corpus** (counsel-relevant for the
from-scratch + patent posture). Maintain a `datasets.csv` with: name, source URL,
license, size, zone tags, license-review status.

## 4. Pipeline architecture

A staged pipeline, orchestrated with Ray (preferred for Python ML data work) or
Airflow if the team standardizes there. Stages:

```
INGEST → NORMALIZE → DEDUP → QUALITY_SCORE → ZONE_TAG → TOKENIZE → SHARD → MANIFEST
```

1. **Ingest.** Pull from source, record provenance (source, license, fetch date,
   hash). Raw lands in cold storage, never mutated.
2. **Normalize.** To a common document schema: text + modality refs + metadata.
   Strip boilerplate, fix encoding, segment.
3. **Dedup.** Exact + near-dup (MinHash/LSH). Cross-shard. Dedup is the single
   biggest lever on raw-to-clean yield and on model quality.
4. **Quality score.** Heuristic + model-based filters (language ID, perplexity
   gate, toxicity/PII screen, length and structure checks). Each doc gets a
   quality score; low scores are dropped or quarantined, not silently deleted.
5. **Zone tag.** Assign zone-affinity tags (§3). Start rule-based (source-derived:
   a code repo → CX; a story corpus → HP), refine with a light classifier.
6. **Tokenize.** Shared tokenizer across zones (the model is unified). Multimodal
   refs tokenized via modality encoders for SM.
7. **Shard.** Fixed-size shards with a deterministic order and a seed, so
   training is reproducible (Research Strategy §8 reproducibility floor).
8. **Manifest.** Emit a manifest per shard: doc count, token count, zone-tag
   distribution, provenance hashes. The manifest is what a federated node trusts.

### 4.1 Cleaning principles
- **Provenance is immutable.** Raw is never edited; cleaning produces new
  artifacts with lineage back to raw.
- **Quarantine over delete.** Dropped data goes to quarantine with a reason code,
  so filters can be audited and tuned.
- **PII and license screening are gates, not warnings.** A doc that fails either
  does not advance.
- **Determinism.** Same raw + same config hash → same shards. This is required
  for federated trust.

## 5. Federated data strategy (grandma-proof)

The criterion the design session landed on: easiest for a node operator to run
correctly. That points to:

- **Nodes receive sharded, manifested, pre-tokenized data**, not raw. The hard
  ingestion/cleaning work is centralized (or run by trusted operators); the node
  operator just trains on verified shards. This is the grandma-proof default.
- **Manifests + hashes let a node verify** it has the right data before training.
- **Signed zone outputs** (Architecture §5.3, §10.3) are what nodes submit; the
  async gather merges them. A node owning one zone trains the full model but
  contributes primarily through its zone's updates.
- **The deterministic merge path** (Q16.16) is what lets independently-trained
  contributions reconcile.

A node operator's job is: install the harness, pull verified shards, train, submit
signed outputs. That is the bar. Raw scraping and cleaning are not in the node
operator's path.

## 6. Staging by rung

| Rung | Corpus slice | Cleaning depth | Goal |
|------|--------------|----------------|------|
| L0 | tiny curated sample (~GBs) | full pipeline, small scale | prove the pipeline + forward pass |
| L1 | balanced multi-zone subset (~tens of GB tokens) | full | prove routing differentiation, GGUF round-trip, build eval harness |
| L2 | full corpus (~200B+ tokens) | full | the product run |
| L3 | federated shards | full + manifest verification | federated proof |

Build the pipeline at L0 so it is correct, then scale the same pipeline up. Do
not build a throwaway L0 pipeline.

## 7. Evaluation harness (built at L1, used everywhere after)

The eval harness is itself a deliverable. It measures:
- **Routing differentiation (H-02):** zone-mix divergence across prompt classes.
- **Provenance faithfulness (H-03):** replay the logged mix, check output match.
- **Capability per parameter (H-01):** NAT versus a dense baseline of equal
  params on a fixed task suite. *This is the bet-deciding metric.*
- **Efficiency (H-04):** per-zone compute, SSM vs attention at equal sequence
  length.
- **Standard quality:** perplexity and a small task battery for sanity.

The harness writes results straight into the case-study template (Research
Strategy §4).

## 8. What two to three months on one Spark realistically yields

Stated plainly, honest posture:
- **Yes:** L0 and L1 done. A working zone-partitioned forward pass, a trained
  1–2B model that demonstrably routes by prompt, a clean GGUF round-trip into an
  Ollama-class harness, the provenance trace working, and the eval harness built.
  That is a real, demonstrable proof of the architecture and a fundable result.
- **Realistically not:** a finished, compute-optimal 10B model from scratch on
  one node. L2 needs aggregate compute. The two months buy the evidence that
  makes committing L2 compute (and the federation) a justified bet rather than a
  hope.

That framing is the product story too: "someone can validate this architecture
on a single Spark in weeks, and the full 10B run is what the federation is for."

## 9. First actions for the data team

1. Stand up cold storage + the immutable-raw convention.
2. Build the `datasets.csv` registry; start license review with counsel.
3. Implement the pipeline at L0 scale, end to end, with manifests.
4. Implement zone-tagging (rule-based first).
5. Produce the L0 shard set; hand to the architecture team for the forward pass.
6. Build the eval harness skeleton in parallel.
