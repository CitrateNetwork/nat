# NAT — paper framework / outline

**Working title:** *The Neuroarchitectural Transformer: Verifiable-by-Construction,
Capability-per-Parameter, and Paraconsistent Federated Training*

**Form:** arXiv preprint (LaTeX-ready) **and** Gradient Paper XI (v3 house style,
cross-citing Papers II/III/VII/X, claim-evidence audited).

**Lead thesis (unified):** *structure is interpretability.* Declaring the transformer's
hidden space into named functional zones, over a fixed auditable topology, with a
hashable provenance trace, simultaneously makes the model **verifiable every pass**,
**more capable per parameter** (proven, H-01), and **trainable by decentralized
consensus**. The blob of weights becomes a dynamic, legible, verifiable instrument.

**Authorship:** Larry Klosowski (@saulbuilds) + the Claude Opus 4.x research-and-
implementation lineage (per the Gradient Papers convention). Honest-posture discipline.

---

## §0 Abstract
The opacity problem → NAT's answer (declared zones + provenance trace) → the three
faces (verifiable/efficient/decentralized) → the headline result (H-01 holds 5/5 on
real text at equal params) → the federated/Belnap/DeSci frame → honest caveats (small
scale). ~200 words.

## §1 Introduction
- **The blob critique.** A modern LLM routes every parameter through every computation;
  there is no honest answer to "which part produced this." Tolerable for chat,
  unacceptable for regulated/on-chain/scientific use.
- **The thesis.** Structure is interpretability. Partition the hidden space into declared
  named zones (brain regions as a *mimetic analog*, engineering-justified), wire them over
  a *fixed* topology a router only modulates, merge by attention-scored pruning on a
  **deterministic Q16.16** path, and emit a **provenance trace** as a first-class hashable
  output. The same move yields verifiability, efficiency, and decentralizability.
- **Contributions** (numbered): (1) the zone-partitioned architecture + fixed-topology
  routing; (2) provenance-as-verifiable-output (decision-faithful, on-chain-committable);
  (3) GGUF/ONNX sidecar (ecosystem-compatible); (4) the H-01 result (partitioning does not reduce capability/param
  equal-param dense, 5/5 seeds, real text); (5) the paraconsistent (Belnap) federated
  training frame on a verifiable chain; (6) a fully reproducible Rust reference impl.
- Honest-posture statement + roadmap.

## §2 Related work  → drawn from `research/RELATED_WORK_AND_CITATIONS.md`
A. Modular DL / MoE — vs learned-opaque routing.
B. **Brain-inspired LLMs / BriLLM** — the sharp differentiation (keeps the transformer;
   verifiable trace, not just a readable graph; proves a per-param claim).
C. SSM / hybrid (Mamba, Jamba) — organized-by-function vs interleaved layers.
D. Interpretability (mech-interp) — intrinsic-by-construction vs post-hoc.
E. zkML / verifiable inference — by-construction vs after-the-fact wrapper (+ Paper X).
F. Decentralized/federated training (DiLoCo) — composable zones vs monolithic replicas.
G. DeSci / blockchain-AI (Bittensor, Gensyn) — verifiable provenance vs trusted scoring.
H. Belnap / paraconsistent logic — disagreement as data (+ Paper II).
I. Scaling laws (Chinchilla) — the orthogonal H-01 question (structure at fixed N,D,C).

## §3 The NAT architecture
- §3.1 Zones (the six: SM, CB SSM; HP, PF, CX attention; MX non-learned harness) +
  output contracts. The mimetic-analog framing; the analogy serves the engineering.
- §3.2 Routing: fixed topology (declared, auditable) + learned per-input modulation;
  the "cannot create undeclared edges" property (claim C-1).
- §3.3 Parallel execution + async gather with deadline (stragglers → timed_out).
- §3.4 Merge: score → prune → re-weight → compose, on the **Q16.16 deterministic path**.
- §3.5 The MCP harness state machine (non-learned executive; fail-closed gate).
- §3.6 Serialization: GGUF/ONNX tensor container + `.nat.json` sidecar (ADR-0004); the
  flattened-dense export caveat (honest, critique #7).

## §4 Verifiability by construction
- The provenance trace schema; **decision-faithful** vs **bit-faithful** (ADR-0006).
- Deterministic hashing → on-chain commitment → third-party replay.
- Relation to Citrate Paper X (same Q16.16 substrate; trace hash committable; Halo2/TEE
  wraps the numeric layer when bit-exact logits are required).
- Formal backing: `formal/{MergeDeterminism,AsyncGather,McpHarness}.tla` (state TLC status
  honestly — written, run-pending JRE) + the claim-shaped statements C-1..C-5.

## §5 Hypotheses & experimental design
- The ledger: H-01 (capability/param — load-bearing), H-02 (routing differentiation),
  H-03a/b (faithfulness), H-04 (SSM efficiency), H-05a/b (merge/federation determinism).
- **ADR-0005 baseline protocol** (equal params, same data/seed/tokenizer/budget; only
  partitioning differs; the harness *refuses* unequal-params runs).
- The corpus: 1.12M-token PD "values-spine" (the Wittgenstein/Boole→Belnap/Turing/
  SICP/Strunk curation + its thesis: "a rule has no meaning without a community" →
  "follow the rules of the room"); license-fail-closed pipeline; quality scoring.

## §6 Results
- **H-01 HOLDS, 5/5 seeds**: NAT held-out loss 2.88–2.91 < dense 2.97–2.99 at ~20.7K
  equal params, next-byte LM on the held-out split, GPU. (Synthetic was marginal 3/5;
  real data decisive.)
- **H-02**: trained router separation 11.70 vs 4.25 baseline (note: in-sample caveat).
- **Size/zone ladder** S→M→L trends down (confounded: zones 3→5 at L) (4.097→4.054→3.953 bits/byte); first real SSM-zone
  training at L. Evidence the structure scales → justifies L2.
- **H-03a**: decision-faithful replay holds by construction (+ test).
- **Honest limitations**: ~20K-param/3-zone byte-LM scale; in-sample H-02; bit-faithful
  only under deterministic-inference; L2 + full federation are future.

## §7 Federated training by paraconsistent consensus  (the community/DeSci frame)
- Composable zones: a node owns/trains a zone, submits **signed zone outputs**; async
  gather + Q16.16 merge reconciles independently-trained contributions.
- **Belnap aggregation** (T/F/B/N) preserves disagreement as data; ties to Paper II
  (operationalizing its specified-not-built mechanism with a real model).
- Incentives: `reward = compute × data-quality`; NAT scores, `citrate-compute-pool`
  settles (ADR-0007, Paper VII). The grandma-proof node-operator path.
- **DeSci framing**: decentralized intelligence meeting decentralized science on a
  verifiable chain; verifiable provenance per training + inference step.
- Belnap consensus inherits BFT safety/liveness from Citrate finality (Paper I/II).

## §8 Discussion
- Why "structure is interpretability" unifies the three faces.
- Threats to validity (scale, synthetic-vs-real, in-sample metrics, single-author seeds).
- Honest posture: H-01 could refute at larger scale; the scale ladder exists to find out
  cheaply; we will say so and change course if it does.
- The agent-driven R&D method (pointer to the companion case study; Research Method /
  Agentile; the build itself as evidence of reproducible agent-led science).

## §9 Conclusion & future work
L2 compute-optimal run; full federation (Gate 4); GGUF FlattenedDense export; counsel/IP
(claims C-1..C-5); the larger-scale H-01 re-test as the next falsification.

## References
From `research/RELATED_WORK_AND_CITATIONS.md` (verify all `[verify]` IDs) + Citrate
Papers I/II/III/VII/X with file:line anchors + the nat repo (commit-pinned).

## Appendices
- A. Formal specs (the three TLA+ modules + invariants).
- B. Reproducibility (config hashes, seeds, the exact rerun commands; `scripts/ci-local.sh`).
- C. The companion case study: *Agents Doing Science — an agent-built model, end to end*
  (the journals; the human-set hypothesis, agent-executed build, honest-posture gates).

---

## Build order (for the rough draft)
1. §3 + §4 (architecture + verifiability) — most grounded in code; write first.
2. §5 + §6 (hypotheses + results) — anchor to gates.yaml/hypotheses.md + the DGX numbers.
3. §2 (related work) — from the research notes.
4. §7 (federated/Belnap/DeSci) — from Papers II/III/VII + the values-spine.
5. §1 + §0 (intro + abstract) — last, once the body is fixed.
6. §8 + §9 + appendices.
Then: red-team pass (task 23) → remediate every claim to canonical truth.
