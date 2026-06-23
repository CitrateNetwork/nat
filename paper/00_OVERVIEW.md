# The Neuroarchitectural Transformer — Overview (abstract + introduction)

*Verifiable-by-Construction, Capability-per-Parameter, and Paraconsistent Federated Training*

**Draft v0.1 (overview).** arXiv preprint + Gradient Paper XI. Honest posture throughout:
every quantitative claim below is anchored to a file/commit in the `nat` repository or
labeled a hypothesis. Caveats are stated in the same breath as results, not buried.

---

## Abstract

Modern language models route every parameter through every computation, so there is no
honest answer to the question *"which part of the model produced this?"* That opacity is
tolerable for casual chat and disqualifying for regulated, on-chain, or scientific use,
where a decision must be replayable and a contribution must be verifiable. We present the
**Neuroarchitectural Transformer (NAT)**, a transformer whose hidden representation is
partitioned into a fixed set of **declared, named zones** — each with its own core
(attention or state-space), each owning a slice of the representation — that execute in
parallel, communicate over a **fixed topology** whose edge strengths a learned router only
*modulates* (it cannot create undeclared edges), and whose outputs are combined by an
attention-scored, noise-pruned **merge running on a deterministic Q16.16 fixed-point path**.
Every forward pass emits a structured, deterministically-hashable **provenance trace** of
which zones fired, with what confidence, what was pruned, and why. The trace is the product:
it makes the model **interpretable and verifiable by construction, on every pass**, rather
than after the fact. We argue this single architectural move — *declaring structure* —
simultaneously yields three properties usually pursued separately. **Verifiability** is *demonstrated*: the trace is a
committable, replayable artifact whose decision layer is third-party-checkable by construction
(complementing zkML's after-the-fact proofs with a structural guarantee), and the stateful
surfaces are model-checked in TLA+. **Capability per parameter** is *empirically supported at small
scale*: on a 1.12M-token public-domain corpus trained as a next-byte language model with a held-out
split, an equal-parameter ablation finds that zone partitioning **does not reduce** capability per
parameter — and is modestly lower-loss on the mean — across five seeds (NAT held-out loss 2.88–2.91
vs dense 2.97–2.99; a non-inferiority verdict with a 5% slack, N=5, no formal significance test);
a learned router differentiates prompt classes and generalizes to held-out prompts (separation
3.10 vs a 2.63 baseline held-out; 11.70 vs 4.25 in-sample); and a three-rung size/zone ladder
trends downward. **Decentralizability** is *specified and formally modeled, not yet demonstrated*:
we describe (and TLA+-check the local determinism behind) reconciling composable named zones via
paraconsistent Belnap aggregation on a verifiable chain, with the multi-node training cycle left as
future work. All empirical results are at small scale (~20K–115K parameters, byte-level), with no
mixture-of-experts baseline or component ablation yet, and we are explicit that a larger,
better-controlled run could refute them — the scale ladder exists precisely to find out cheaply.
NAT is implemented as a reproducible Rust workspace; a GGUF/ONNX sidecar **design** targets
ecosystem compatibility (the round-trip is not yet shown). We position NAT as the model layer for a
verifiable, federated learning network in which *consensus and learning are the same process*, and
contributors are rewarded as compute × data-quality.

---

## 1. Introduction

### 1.1 The blob of weights

A transformer is, mechanically, an undifferentiated block of parameters through which
every token is pushed in full. This is enormously effective and enormously opaque. When
such a model emits an output, there is no architectural fact of the matter about *which
part of it* was responsible — the representation is a single high-dimensional smear, and
any account of "what the model was doing" must be reconstructed, after training, by
external tooling (probing, sparse autoencoders, activation patching). Mechanistic
interpretability has made real progress at this reconstruction [Rai et al. 2024; Bereska &
Gavves 2024], but it remains *post-hoc*: it approximates, after the fact, a structure the
architecture never committed to. You cannot *prove* what ran; you can only *infer* it.

For chat, that is acceptable. For the settings that matter most to us — regulated decisions
(finance, healthcare, defense), on-chain inference where a decision must be replayable by a
third party, and decentralized scientific training where a contribution must be *verifiable*
to be rewarded — opacity is disqualifying. These settings do not want a better explanation
of a black box. They want the box not to be black.

### 1.2 The thesis: structure is interpretability

NAT's wager is that you can largely dissolve the opacity problem by *declaring structure*
the architecture must respect, rather than discovering it afterward. Concretely:

- **Partition** the hidden representation into a fixed set of **named zones**, each mapped
  (as a *mimetic analog*, not a fidelity claim) to a functional role a brain handles at a
  different speed and depth: sensorimotor binding, cerebellar timing, hippocampal salience,
  prefrontal reasoning, codec compilation, and a non-learned executive harness for tool use.
  Temporal zones use state-space cores [Gu & Dao 2023]; reasoning zones use attention.
- **Wire** the zones over a **fixed, declared topology** that an auditor and a model checker
  can both read directly. A learned router *modulates* the edge strengths and zone
  activations per input — a math prompt drives one zone mix, a narrative prompt another —
  but it **cannot create an edge the topology does not declare.** Adaptivity without opacity.
- **Merge** the zone outputs by attention-scored pruning (drop the noisy majority,
  re-weight the survivors) on a **deterministic Q16.16 fixed-point path**, so the same
  gathered set always composes to the same bits — the property federated reconciliation and
  on-chain verification both require.
- **Emit** a structured **provenance trace** as a first-class output of every forward pass:
  the router's modulation, each zone's activation/confidence/latency/status, the merge's
  scores and prune decisions and survivors, the codec's verification result, and the
  harness's state transitions — serialized deterministically and hashed. On Citrate, that
  hash becomes part of the inference transaction; an auditor replays the recorded decision
  against the committed weights and confirms it.

The trace is not a debug aside; it is the deliverable. "Which functional component produced
this contribution" becomes a property of the architecture, available and verifiable on
every pass, not a property of a post-hoc tool that may or may not be faithful.

### 1.3 One move, three properties

The contribution we most want to land is that *declaring structure* is not a trade — not
"give up capability to gain interpretability." It is a move that pays out along three axes
at once, which the literature usually pursues in isolation:

1. **Verifiable by construction.** Zero-knowledge ML can prove, expensively and after the
   fact, that *some* opaque computation ran on committed weights [Kang et al. 2024; Sun et
   al. 2024] — hundreds of seconds per proof for a small transformer, verifying the *output*
   but not the *reasoning*. NAT's provenance trace is verifiable *by construction*: the
   recorded decision is decision-faithful and replayable with no per-inference SNARK, on the
   same Q16.16 substrate Citrate's verifiable-inference precompiles already use (Paper X).
   The two compose — a SNARK or TEE attestation can wrap the numeric layer when bit-exact
   logits are required — but only NAT supplies the *structural* guarantee.
2. **Capability per parameter that does not pay an interpretability tax.** This is our
   load-bearing, falsifiable claim. Holding parameters, data, seed, and compute fixed and
   varying *only* the partitioning (the ADR-0005 protocol, enforced in code — the harness
   refuses an unequal-parameter comparison), zone partitioning **does not reduce** capability
   per parameter versus an equal-parameter dense baseline on a real next-byte language-modeling
   task — and is modestly lower-loss on the mean — across all five seeds we ran (a non-inferiority
   result with a 5% slack; N=5, no formal significance test). We are deliberately explicit about
   scale (§6): this is ~20K parameters, three zones, byte-level, and there is no mixture-of-experts
   baseline or component ablation yet, so a larger or better-controlled run could overturn it. We
   report it because it is what we measured; the design at minimum does not trade capability for
   the verifiability it buys.
3. **Decentralizable.** Because zones are *composable* — a zone is swappable when its slice
   width and cross-zone contract match — a federation can train and evolve a single zone
   without retraining the whole model. Independently-trained zone contributions reconcile
   through the deterministic merge locally, and across nodes through **paraconsistent
   (Belnap four-valued) aggregation** [Belnap 1977], which preserves genuine disagreement
   (*Both*) and genuine ignorance (*Neither*) as first-class rather than averaging them away.
   This operationalizes the Citrate thesis that *consensus and learning are the same process*
   (Paper II), with contributor reward settled as compute × data-quality (Paper VII).

### 1.4 Contributions

We tag each by status — *[demonstrated]* (implemented and evaluated), *[implemented]* (built and
tested, not an evaluation), *[specified]* (designed, and where noted formally modeled, not yet
demonstrated):

1. *[implemented]* The **zone-partitioned architecture**: declared named zones over a fixed,
   auditable topology with learned-but-bounded routing (the router provably cannot create
   undeclared edges), hybrid SSM/attention cores by function, and a non-learned executive harness.
2. *[demonstrated]* **Provenance-as-verifiable-output** — the paper's primary contribution: a
   first-class, deterministically-hashable trace, decision-faithful and third-party-replayable on
   the Q16.16 path and on-chain-committable, distinct from model cards, logging, and post-hoc
   interpretability (it is an output of the pass, not an external probe), with a precise
   decision-faithful vs bit-faithful distinction.
3. *[specified]* An **ecosystem-compatibility design**: a GGUF/ONNX sidecar intended to preserve
   the existing inference stack; the flattened-dense export and Ollama-class round-trip are not
   yet built (WP-1.4).
4. *[demonstrated, small scale]* The **H-01 result**: an equal-parameter ablation on real text
   finding partitioning does not reduce capability per parameter (non-inferiority, modest mean
   advantage, 5 seeds) — with explicit scale and missing-baseline caveats.
5. *[specified]* A **paraconsistent federated-training frame**: composable zones reconciled by
   Belnap aggregation on a verifiable Q16.16 substrate (the local determinism is TLA+-checked; the
   multi-node cycle is future work), with compute × data-quality incentives.
6. *[implemented]* A **reproducible Rust reference implementation** with **TLC-green** TLA+
   specifications of the stateful surfaces, and a companion case study on agent-led model building.

### 1.5 Honest posture

This is a research bet held to an explicit discipline: every quantitative claim is anchored
to a file or commit, or labeled a hypothesis; the brain analogy is a design heuristic the
architecture is free to abandon wherever the engineering is better served; and the
load-bearing capability claim is stated with its scale caveat in the same breath. If a
larger run refutes H-01, the honest move is to say so and change course — and the scale
ladder exists so that we, and the reader, can find out cheaply. The remainder of the paper
makes the architecture precise (§3), the verifiability claim precise (§4), the hypotheses
and protocol precise (§5), reports what we have measured (§6), develops the federated and
decentralized-science frame (§7), and is candid about what we have *not* yet shown (§8).
