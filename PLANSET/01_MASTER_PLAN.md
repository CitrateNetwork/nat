# Citrate Neuroarchitectural Transformer — Master Plan

**Document:** RFC-CIT-NAT-0001 / Program Plan
**Status:** Draft v0.1 (pre-review)
**Owner:** Larry Klosowski (@saulbuilds)
**Entity:** Mozi Satori, Inc. / Citrate Network
**Date:** 2026-06-18

---

## 0. What this is

The Citrate Neuroarchitectural Transformer (NAT) is a modular transformer whose
embedding space is partitioned into named functional zones. Each zone mirrors,
as a mimetic analog, a role the brain handles at a different speed and depth.
Zones run in parallel, communicate over a fixed topology with learned
modulation, and merge through attention scoring plus pruning. Every inference
emits a provenance log of which zones fired and why, which is the wedge against
the opacity problem and the basis for on-chain auditability on Citrate.

This is a research bet, not a finished thesis. The brain analogy is a design
heuristic. The product claim is narrower and testable: a transformer that stays
backwards-compatible with GGUF and ONNX, runs in an Ollama-style harness, can be
trained in a federated cycle on Citrate, and produces an auditable trace of its
own reasoning.

## 1. Why now, why us

Current LLMs route every parameter through every computation, so there is no
honest answer to "which part of the model produced this." That is tolerable for
chat and unacceptable for regulated domains... finance, healthcare RCM (Dandi),
defense procurement (the federal/Fortune 500 binder work), and any on-chain
inference where a decision has to be replayable.

Citrate already has the primitives this needs: a live L1 with deterministic ML
(Q16.16), 80+ smart contracts, a TLA+ verification discipline, and a federated
learning target. NAT is the model layer that those primitives were waiting for.

The honest-posture version of the pitch: we do not yet know that zone
partitioning improves capability per parameter. We know it improves
*interpretability* by construction, and we have a credible path to test the
capability question at small scale before committing the full 10B run.

## 2. Scope

### In scope (v1)
- Six-zone architecture: Sensorimotor, Cerebellar, Hippocampal, Prefrontal,
  Codec (code compilation), and the MCP Harness (executive function / tool use).
- Hybrid routing: fixed inter-zone topology, context-aware learned modulation.
- Parallel zone execution with async gather, attention-scored merge, noise
  pruning.
- State Space Model (SSM) cores inside temporal zones; attention cores inside
  reasoning zones.
- Provenance logging as a first-class output of the forward pass.
- Auxiliary metadata format wrapping GGUF/ONNX (zone declarations, training
  recipes, composition rules).
- GGUF compatibility and Ollama-harness trainability.
- Rust reference implementation.

### Out of scope (v1, revisit later)
- Full federated training across untrusted nodes at production scale. We
  validate centralized first, then add distributed training.
- Smell modality. Audio, vision, touch, proprioception, and text only.
- Neuromorphic / spiking hardware. We stay on commodity GPU + Citrate compute.

## 3. Goals and non-goals

**Goals**
1. Prove the zone-partitioned forward pass works and produces a usable
   provenance trace.
2. Show backwards compatibility: a NAT export loads and runs in an Ollama-class
   GGUF harness.
3. Demonstrate context-aware routing... different prompts measurably activate
   different zone mixes.
4. Stand up a federated training cycle on Citrate at proof scale.
5. Produce IP-ready documentation (TLA+ specs, Gherkin acceptance criteria) that
   counsel can convert into patent claims and engineers can build from.

**Non-goals**
- Beating frontier models on general benchmarks. NAT is a small, efficient,
  multi-competent model, judged on efficiency-per-parameter and auditability,
  not on raw leaderboard position.
- One-to-one neurological fidelity. The analogy serves the engineering, not the
  reverse.

## 4. The scale ladder (honest compute posture)

A single DGX Spark-class node (GB10, ~128GB unified memory, bandwidth-bound) is a
prototyping and fine-tuning device. Training a 10B model from scratch to a
compute-optimal token budget on one such node in two to three months is not
realistic, and saying otherwise would violate honest posture. The plan threads
the needle with a scale ladder:

| Rung | Params | Where | Purpose | Honest expectation |
|------|--------|-------|---------|--------------------|
| L0 | ~150M | Spark, days | Wire up the forward pass, prove provenance log emits | Toy quality, architecture validated |
| L1 | ~1–2B | Spark, 2–4 weeks | Prove routing differentiates by prompt; prove GGUF export | Coherent, narrow, useful for eval harness |
| L2 | ~10B | Federation + cloud burst | The real product run | Months of wall-clock; needs aggregate compute beyond one node |
| L3 | ~10B | Citrate federated cycle | Prove grandma-proof distributed training | Research milestone, not a v1 guarantee |

The two-to-three-month window on a single Spark gets us through L0 and L1
cleanly, and gives us the eval harness and the export pipeline. L2 is where the
4–6TB (scalable to ~100TB) corpus and the federation earn their keep. We size
L2 honestly in the Data Operations plan.

## 5. Program phases (gated)

Following the five-gate pattern already used across Citrate work.

**Gate 1 — Theory locked.** Architecture spec, formal-spec scaffold, and this
plan reviewed and signed. Exit: named reviewers approve (Taurien legal/IP,
Lauren IP/BD, plus an ML reviewer). *This document set is the Gate 1 artifact.*

**Gate 2 — Reference forward pass.** L0 model runs end to end in Rust. Zones
execute in parallel, router modulates, merge prunes, provenance log validates
against the Gherkin acceptance criteria. Exit: green on the Gate 2 feature file.

**Gate 3 — Trainable and portable.** L1 model trains on the Spark, exports to
GGUF, loads in an Ollama-class harness, runs inference with intact provenance.
Exit: round-trip export/import test passes; routing-differentiation metric beats
baseline.

**Gate 4 — Federated proof.** Two or more nodes train partitioned data toward
the shared model on Citrate; async gather and signed zone outputs merge
deterministically. Exit: federated run reproduces centralized result within
tolerance; on-chain provenance hashes verify.

**Gate 5 — Productize.** Design interface (individual + federation views) ships
from the visual brief, docs are patent-filed, L2 run is scheduled with committed
compute. Exit: counsel files, design ships, L2 kickoff.

## 6. Workstreams and owners (proposed)

| Workstream | Lead (proposed) | First deliverable |
|-----------|-----------------|-------------------|
| Architecture & Rust core | Larry + eng | L0 forward pass |
| Formal verification (TLA+) | eng + reviewer | Merge + state-machine specs |
| Data operations | James / Dan | Source taxonomy + ingestion pipeline |
| Federated training on Citrate | eng | Async gather contract |
| IP / patent | Taurien, Lauren | Claims draft from formal scaffold |
| Visual design | Claude Design (brief-driven) | Training console v1 |

## 7. Risk register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Zone partitioning hurts capability per parameter | Medium | High | L0/L1 ablation against a dense baseline of equal params before L2 commit |
| SSM↔attention gluing unstable in training | Medium | Medium | Isolate SSM zones first; freeze cross-zone heads early, unfreeze on schedule |
| 10B from scratch infeasible in budget | High (single node) | High | Scale ladder; allow pretrained zone init as fallback (counsel-cleared licenses only) |
| Federated determinism breaks (float nondeterminism) | Medium | High | Q16.16 deterministic path for merge; signed, ordered gather |
| Patent novelty already covered by prior art | Medium | High | Prior-art sweep done (see Journal); counsel review at Gate 1 |
| GGUF cannot express zone metadata cleanly | Low | Medium | Auxiliary sidecar format; GGUF stays the tensor container, sidecar carries zone graph |

## 8. Prior art posture (summary)

A scan of current work (multimodal transformers in federated learning,
FedMoE, distributed mixture-of-experts, brain-inspired LLMs such as BriLLM,
neuroscience-inspired embodied frameworks) shows active research on
*federated MoE* and on *brain-as-inspiration* models, but no clear prior art on
explicit, declared zone partitioning of the embedding space with a baked-in
provenance trace and an auxiliary format wrapping GGUF/ONNX. That gap is the
novelty wedge. This is a working assessment for counsel, not a legal opinion.
Detail in the Journal and the Architecture spec §11.

## 9. Definition of done for this document set

This planset hands off to Claude Code (for refinement and build) and Claude
Design (for the interface). It is done when:
- An ML engineer can read the Architecture spec and start building.
- A verification engineer can read the Formal scaffold and start writing TLA+ and
  `.feature` files.
- A data engineer can read the Data Operations plan and start ingestion.
- A designer can read the Visual brief and start the console.
- Counsel can read the spec + scaffold and start a claims draft.

## 10. Document map

- `00_MASTER_PLAN.md` — this file.
- `01_RESEARCH_AND_DOCUMENTATION_STRATEGY.md` — Agentile method, case-study
  protocol, how every stage gets documented.
- `02_ARCHITECTURE_SPEC.md` — the comprehensive design, RFC-style, patent-oriented.
- `03_FORMAL_SPEC_SCAFFOLD.md` — TLA+ modules, Gherkin features, acceptance
  criteria.
- `04_DATA_OPERATIONS_AND_TRAINING_PLAN.md` — data shape, sources, cleaning,
  pipelines, compute math.
- `05_VISUAL_DESIGN_BRIEF.md` — interface brief for Claude Design.
- `JOURNAL.md` — Claude's first-person record of how this got designed.
