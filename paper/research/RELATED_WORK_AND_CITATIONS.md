# NAT paper — related work, citations, and the novelty wedge per area

**Status:** research notes for the arXiv paper / Gradient Paper XI. Compiled 2026-06-22
from a structured web sweep. Each section gives the key citations and — crucially —
**how NAT differs**, which is what the related-work section and reviewers will demand.

> Note: arXiv IDs marked `[verify]` are from memory of canonical works and must be
> confirmed against the live listing before submission. IDs from the 2026-06-22
> search are marked `[searched]`.

---

## A. Modular deep learning & mixture-of-experts (MoE)

**Key works.**
- Pfeiffer, Ruder, Vulić, Ponti. *Modular Deep Learning.* arXiv:2302.11529 (2023) `[searched]`.
  Unifying survey: module implementation, **routing**, **aggregation**, training.
  Advantages named: positive transfer, **compositionality**, **parameter efficiency**.
- Shazeer et al. *Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer.*
  arXiv:1701.06538 (2017) `[verify]`. The conditional-computation root.
- Lepikhin et al. *GShard.* arXiv:2006.16668 (2020) `[verify]`; Fedus et al. *Switch
  Transformer.* arXiv:2101.03961 (2022) `[verify]`; Jiang et al. *Mixtral of Experts.*
  arXiv:2401.04088 (2024) `[verify]`.
- *A Survey on Mixture of Experts.* (2024) `[verify arXiv:2407.06204]`; *MoErging:
  Routing Among Specialized Experts.* arXiv:2408.07057 (2024) `[searched]`.

**How NAT differs (the wedge).** MoE routing is **learned from scratch and opaque** —
experts are interchangeable, discovered, unnamed; the router is a black box and the
expert assignment carries no semantics. NAT's zones are **declared, named, and wired
over a *fixed* topology** an auditor and a TLA+ model both read directly; the router
*modulates* that fixed graph and **cannot create undeclared edges**. MoE optimizes for
throughput (sparse activation); NAT optimizes for **auditability + capability-per-param
at equal total params** (H-01), with *every* zone's participation recorded. MoE is
"which expert," discovered; NAT is "which declared function," by construction.

---

## B. Brain-inspired / neuroscience-motivated LLMs  ← closest prior art

**Key work.**
- BriLLM: *Brain-inspired Large Language Model.* arXiv:2503.11299 (2025) `[searched]`.
  "Signal Fully-connected flowing (SiFu)" over a fully-connected graph; tokens → nodes
  (cortical-area analogy); **replaces attention** with dynamic signal propagation.
  Claims full node-level interpretability + context-length-independent scaling. 1–2B
  reaches ~GPT-1 generative quality.

**How NAT differs (must differentiate sharply — same neighborhood).**
1. **BriLLM abandons the transformer; NAT keeps it.** BriLLM swaps attention for a
   bespoke propagation graph — which forfeits the entire GGUF/ONNX/Ollama ecosystem.
   NAT stays a transformer (attention + SSM cores inside zones), **GGUF-compatible via
   a sidecar**, so it runs in the existing stack. Backwards compatibility is a design
   axiom, not an afterthought.
2. **BriLLM's interpretability is node-level token mapping; NAT's is functional zones
   + a hashable provenance trace.** BriLLM tells you which token-node lit; NAT records
   *which functional zone fired, with what confidence, what was pruned, and why* — and
   makes that record **deterministically hashable and on-chain-verifiable** (the Q16.16
   merge). NAT's interpretability is a *verifiable artifact*, not just a readable graph.
3. **NAT proves a capability claim (H-01) against an equal-param dense baseline**;
   BriLLM reports generative quality, not a per-parameter ablation.

This is the paper's single most important "vs prior art" paragraph.

---

## C. State-space models & hybrid SSM–attention (the temporal-zone cores)

**Key works.**
- Gu, Goel, Ré. *Efficiently Modeling Long Sequences with Structured State Spaces (S4).*
  arXiv:2111.00396 (2021) `[verify]`.
- Gu, Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.*
  arXiv:2312.00752 (2023) `[verify]`; Dao, Gu. *Mamba-2 / State-Space Duality.* (2024).
- Lieber et al. *Jamba: A Hybrid Transformer-Mamba Language Model.* arXiv:2403.19887
  (2024) `[searched]`. ~43% Mamba / 7% attention / 50% MLP; balances long-context + ICL.

**How NAT differs.** Jamba interleaves SSM and attention **layers** in one undifferentiated
stack. NAT places SSM cores **inside named temporal zones** (Sensorimotor, Cerebellar)
and attention cores inside reasoning zones (Hippocampal, Prefrontal, Codec) — the
hybrid is *organized by declared function*, and each SSM zone carries a thin attention
head only for cross-zone talk. ADR-0002. The DGX scale-ladder L-rung (115K params,
5-zone) is the first real-data training of these SSM zones (monotone bits/byte gain).

---

## D. Interpretability: the opacity problem; intrinsic vs post-hoc

**Key works.**
- Rai et al. *A Practical Review of Mechanistic Interpretability for Transformer-Based
  LMs.* arXiv:2407.02646 (2024) `[searched]`.
- Bereska, Gavves. *Mechanistic Interpretability for AI Safety — A Review.*
  arXiv:2404.14082 (2024) `[searched]`. Taxonomy: **intrinsic** (before training) vs
  **developmental** vs **post-hoc** (after training).

**How NAT differs (the lead thesis: "structure is interpretability").** Mech-interp is
overwhelmingly **post-hoc**: reverse-engineer circuits out of a trained black box (SAEs,
probing, activation patching). It *reconstructs* what ran; it cannot *prove* it. NAT is
**intrinsic by construction**: the zone partition is *declared*, so "which functional
component produced this contribution" is a property of the architecture, not of a tool —
and the provenance trace makes it a hashable, replayable record. NAT moves
interpretability from "explain after the fact, approximately" to "**record at the source,
verifiably, every pass**."

---

## E. Verifiable / zero-knowledge ML, on-chain inference  ← the verifiability wedge

**Key works.**
- Kang et al. *ZKML: Optimizing System for ML Inference in Zero-Knowledge.* EuroSys 2024
  `[searched]`. Sun et al. *zkLLM.* (2024) `[searched]` — verifiable transformer inference
  (GPT-2 proof ~287s). *zkGPT.* USENIX Security 2025 `[searched]`. *A Survey of ZKP-Based
  Verifiable ML.* arXiv:2502.18535 (2025) `[searched]`. EZKL (ONNX→zk-SNARK). Modulus Labs
  (≤18M-param on-chain verification). TEE attestation (MAA/NRAS) — see Citrate Paper X.

**How NAT differs (and complements Citrate Paper X).** zkML wraps a **black box** in an
*external* proof: prove, after the fact, that *some* opaque computation ran on committed
weights — at heavy cost (hundreds of seconds/proof; bounded model size). It verifies the
*output*, not the *reasoning*. NAT makes the model **verifiable by construction**: the
provenance trace — which zones, which prune, which weights, on the deterministic Q16.16
merge — *is* the verifiable artifact, **decision-faithful on every pass with no per-
inference SNARK**. Relationship to Paper X: NAT's Q16.16 merge path is the same fixed-point
substrate Paper X's precompiles use; a NAT trace hash is directly committable, and a Halo2
proof or TEE attestation can wrap the *numeric* layer when bit-exact logits are required
(H-03b). NAT supplies the *structural* guarantee zkML cannot: **interpretable + verifiable
reasoning, not just verified output.**

---

## F. Decentralized & federated training  ← the community-training wedge

**Key works.**
- McMahan et al. *Communication-Efficient Learning… (FedAvg).* AISTATS 2017 `[verify]`
  (already cited in Citrate Paper II).
- Douillard et al. *DiLoCo: Distributed Low-Communication Training of LMs.*
  arXiv:2311.08105 (2023) `[searched]` — FedAvg-style outer loop; ~500× less comms;
  robust to heterogeneous data + churn. OpenDiLoCo (Prime Intellect, arXiv:2407.07852,
  2024) `[searched]` — real cross-continent training, 90–95% utilization. DiLoCoX
  (107B, 2025) `[searched]`.
- Hu et al. *LoRA.* arXiv:2106.09685 (2021) — adapter-level federation (Citrate Paper II/III).

**How NAT differs / fits.** DiLoCo federates a **monolithic dense** model (replicas of the
whole net, periodic averaging). NAT federates **composable named zones**: a node owns/
trains a zone, submits **signed zone outputs**, and the **async gather + deterministic
Q16.16 merge** reconciles independently-trained contributions — the same gather discipline
proven in `formal/AsyncGather.tla`. Where DiLoCo averages gradients, NAT's federation runs
through **paraconsistent (Belnap) aggregation** (§H) so disagreement is preserved, not
averaged away. NAT is the *model architecture* that makes Citrate Paper II/III's federated
meta-learning concrete.

---

## G. Decentralized science (DeSci) & blockchain-incentivized AI

**Key works.**
- *SoK: Blockchain-Based Decentralized AI (DeAI).* arXiv:2411.17461 (2024) `[searched]`.
- Bittensor (TAO, **Yuma Consensus** scoring miner model outputs); Gensyn (compute
  marketplace). *DeScAI: convergence of DeSci and AI.* Frontiers in Blockchain (2025)
  `[searched]`.

**How NAT differs / fits.** Bittensor/Gensyn incentivize **opaque** model/compute
contributions scored by validators on output quality — you trust the score, not the
computation. NAT contributes the missing **verifiable substrate**: a contributor's work is
a *signed, provenance-traced, Q16.16-deterministic* zone update whose value is
`reward = compute × data-quality` (NAT scores; `citrate-compute-pool` settles, ADR-0007).
This is DeSci with **verifiable provenance on every training and inference step** — the
honest-metering problem Bittensor's Yuma Consensus approximates statistically, NAT makes
checkable by construction. *Decentralized intelligence meeting decentralized science on a
verifiable chain.*

---

## H. Paraconsistent / Belnap four-valued logic  ← the consensus wedge

**Key works.**
- Belnap. *A Useful Four-Valued Logic.* (1977) — values {T, F, **B**oth, **N**either};
  logical order vs information/knowledge order.
- *Four Imprints of Belnap's Useful Four-Valued Logic in Computer Science.*
  arXiv:2503.20679 (2025) `[searched]` — modern CS survey. Belnap–Dunn (FDE / relevance
  logic) for multiple inconsistent/incomplete sources. SEP, *Paraconsistent Logic.*
- Citrate Paper II (Paraconsistent Consensus) — Belnap aggregation over Q16 embeddings;
  "consensus and learning are the same process at different time scales."

**How NAT differs / fits.** This is NAT's federated-aggregation logic. When nodes with
different data train the same zone, per-dimension contributions go to a Belnap state:
**T/F** (agreement), **B** (genuine disagreement — route to multiple zones, don't average),
**N** (no signal — don't fabricate one). NAT operationalizes Paper II's *specified-not-built*
mechanism with a real model: the zone outputs are the contributions, the Q16.16 merge is
the local aggregation, and the on-chain checkpoint commits the Belnap-aggregated zone
weights. Disagreement is **data about the network's epistemic state**, not error.

---

## I. Scaling laws / capability-per-parameter (the H-01 frame)

**Key works.**
- Hoffmann et al. *Training Compute-Optimal LLMs (Chinchilla).* arXiv:2203.15556 (2022)
  `[verify]` — ~20 tokens/param compute-optimal; smaller-but-better-trained beats larger.
- Kaplan et al. *Scaling Laws for Neural LMs.* arXiv:2001.08361 (2020) `[verify]`.

**How NAT differs / fits.** Scaling laws hold **architecture fixed** and vary N, D, C.
H-01 asks the orthogonal question Chinchilla doesn't: **at fixed N, D, seed, and compute,
does *structure* (zone partitioning) change capability per parameter?** The DGX result:
**yes, in NAT's favor** — NAT held-out loss 2.88–2.91 < equal-param dense 2.97–2.99, 5/5
seeds, on a 1.12M-token PD corpus (next-byte LM, held-out split). Small scale (~20K params,
3 zones) — honest caveat — but the scale ladder S→M→L is monotone (4.097→4.054→3.953
bits/byte), evidence the structure *scales*. This is the empirical core.

---

## J. Foundational / ecosystem (cite from canon)

- Vaswani et al. *Attention Is All You Need.* arXiv:1706.03762 (2017) `[verify]` — the base.
- GGUF / `llama.cpp` (Gerganov) — the on-device inference ecosystem NAT stays compatible
  with via the sidecar (ADR-0004). Ollama as the adoption onramp.
- TLA+ (Lamport) — the formal method behind `formal/{MergeDeterminism,AsyncGather,McpHarness}.tla`.

---

## The one-paragraph novelty statement (synthesized)

NAT is, to our knowledge, the first architecture to combine: **(1)** declared, named
zone-partitioning of a transformer's hidden space over a fixed, auditable topology
(vs learned-opaque MoE and vs the attention-abandoning BriLLM); **(2)** a first-class,
deterministically-hashable **provenance trace** that makes the model *verifiable and
interpretable by construction on every pass* (vs post-hoc mech-interp and vs after-the-fact
zkML); **(3)** a GGUF/ONNX-compatible sidecar preserving the existing inference ecosystem;
and **(4)** a federated training story that reconciles independently-trained zones via
**paraconsistent (Belnap) aggregation** on a deterministic Q16.16 substrate, settling
contributor reward as compute × data-quality. The load-bearing empirical claim — that
partitioning **beats** an equal-parameter dense baseline (H-01, 5/5 seeds on real text) —
is what turns the design from a plausible story into a result.

## Open citation tasks before submission
- Verify every `[verify]` arXiv ID against the live listing.
- Pull BriLLM in full (WebFetch arXiv:2503.11299) and write the precise differentiation.
- Confirm the MoE survey arXiv ID (2407.06204 vs the TKDE'25 version).
- Add Citrate canonical citations (Papers I, II, III, VII, X) with file:line anchors.
