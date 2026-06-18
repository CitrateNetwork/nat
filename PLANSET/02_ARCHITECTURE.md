# Citrate Neuroarchitectural Transformer — Architecture Specification

**Document:** RFC-CIT-NAT-0001 / Architecture
**Status:** Draft v0.1 (pre-review)
**Reviewers (pre-v1.0-rc):** Taurien Buffaloe (legal/IP), Lauren Mendenhall (IP/BD), ML reviewer TBD
**Companion to:** `00_MASTER_PLAN.md`, `03_FORMAL_SPEC_SCAFFOLD.md`

---

## 1. Overview

The Neuroarchitectural Transformer (NAT) partitions a transformer's hidden
representation into a fixed set of named **zones**. Each zone is a specialized
sub-network with its own core (attention or State Space Model), its own input
projection, and its own internal state. Zones execute in parallel over a shared
input, communicate across a **fixed topology** whose edge strengths are
**modulated per input by a learned router**, and have their outputs combined by
an **attention-scored, noise-pruned merge**. A **provenance trace** of the entire
pass is emitted as structured output. Tool use and external actions are mediated
by a non-learned **MCP Harness** that enforces a state machine over validation,
routing, and execution.

The design is backwards-compatible: tensors serialize to GGUF/ONNX, an auxiliary
sidecar carries the zone graph, and the model loads and runs in an Ollama-class
harness. The reference implementation is Rust.

This section is normative for terminology. Later sections refine each component
to the level a formal specification and a patent claim set require.

## 2. Design principles

1. **Structure is interpretability.** Zones are declared, not discovered.
   Knowing which zone produced a contribution is a property of the architecture,
   not of a post-hoc tool.
2. **Backwards compatibility is the onramp.** If it cannot load in Ollama, the
   ecosystem cannot adopt it. GGUF stays the container.
3. **Determinism where it matters.** The merge and the tool-routing path run on a
   deterministic numeric path (Q16.16) so federated results reconcile and
   on-chain provenance verifies.
4. **The analogy serves the engineering.** Brain regions inspire zone roles. The
   architecture is free to diverge wherever the engineering is better served.
5. **Honest posture.** Capability claims are hypotheses until measured (see
   Research Strategy §6).

## 3. Terminology (normative)

- **Zone** — a named sub-network owning a slice of the hidden representation, a
  core, an input projection, and internal state.
- **Core** — the per-zone sequence operator: `ATTENTION` or `SSM`.
- **Topology** — the fixed directed graph of permitted inter-zone edges.
- **Router** — the learned function producing per-input modulation weights over
  topology edges and zone activation.
- **Merge** — the operation that scores, prunes, re-weights, and composes zone
  outputs into the model output.
- **Provenance trace** — the structured, hashable record of zone activations,
  scores, prune decisions, inter-zone flows, and tool routing for one pass.
- **MCP Harness** — the non-learned executive layer that validates and sequences
  tool calls via an explicit state machine.
- **Sidecar** — the auxiliary metadata file declaring zones, topology, recipes,
  and composition rules, paired with a GGUF/ONNX tensor container.

## 4. The six zones

Each zone has: a role, a core type, the modalities it consumes, its training
signal, and its output contract. Hidden width `D` is partitioned into per-zone
slices; slice widths are configuration, not fixed by this spec.

### 4.1 Sensorimotor (`SM`)
- **Role:** ingest and temporally bind multimodal input (vision, audio, touch,
  proprioception, text tokens).
- **Core:** `SSM`. Multimodal signals are sequences with temporal structure; SSM
  recurrence binds them in linear time.
- **Input:** raw or lightly featurized modality streams via modality-specific
  encoders feeding a shared SM slice.
- **Training signal:** multimodal alignment (contrastive + reconstruction).
- **Output contract:** a temporally-bound feature sequence over the SM slice plus
  a cross-zone summary head.

### 4.2 Cerebellar (`CB`)
- **Role:** timing, motor sequencing, learned reflexive patterns.
- **Core:** `SSM`. Inherently sequential; benefits from state evolution over
  attention.
- **Input:** SM summary + positional/timing deltas.
- **Training signal:** repetition + error correction (sequence prediction with a
  timing loss).
- **Output contract:** a timing/sequence correction signal + cross-zone head.

### 4.3 Hippocampal (`HP`)
- **Role:** memory consolidation; tag novelty and salience; bind short-term
  context into retrievable structure.
- **Core:** `ATTENTION`. Retrieval and salience benefit from content-addressable
  attention.
- **Input:** SM summary + current context window + a memory store interface.
- **Training signal:** novelty/salience weighting (surprise-weighted objective).
- **Output contract:** salience-weighted context + retrieval results + cross-zone
  head.

### 4.4 Prefrontal (`PF`)
- **Role:** reasoning, planning, abstraction, language. The deepest, slowest
  zone.
- **Core:** `ATTENTION`. Flexible look-back across reasoning steps.
- **Input:** HP salience output + CB timing + SM summary.
- **Training signal:** reasoning chains, language modeling, feedback.
- **Output contract:** reasoning state + token logits contribution + cross-zone
  head.

### 4.5 Codec (`CX`) — code compilation
- **Role:** turn reasoning outputs into verifiable, executable logic. The
  determinism anchor.
- **Core:** `ATTENTION` over a constrained vocabulary, with an external
  compile/verify hook.
- **Input:** PF reasoning state.
- **Training signal:** verified executable logic (syntax-valid, test-passing,
  formally checkable where possible).
- **Output contract:** a candidate program/spec + a verification result
  (`pass | fail | unverified`). A `fail` is a first-class output, not a
  discarded one... the provenance trace records the failure.

### 4.6 MCP Harness (`MX`) — executive function / tool use
- **Role:** validate, sequence, and route tool calls and external actions.
  Mimetic analog of executive control (error detection, task switching).
- **Core:** **none.** `MX` is not a learned zone. It is a deterministic state
  machine plus a validator. It consumes the merged signal, checks tool
  preconditions, enforces the action gate, and emits tool calls.
- **Input:** merged zone output + tool registry (MCP servers/tools available).
- **Output contract:** a validated tool call (or a refusal with reason) + the
  state-machine transition record.

The split matters for IP and for verification: five learned zones plus one
non-learned executive harness. The harness is where determinism and the state
machine live, and it is the most straightforwardly formalizable component.

## 5. Routing (hybrid)

### 5.1 Topology (fixed)
The permitted inter-zone edges are declared in the sidecar and fixed at build
time. Default topology:

```
SM → CB        SM → HP        SM → PF
CB → PF        HP → PF        PF → CX
(all learned zones) → MX merge boundary
```

`SM` feeds the temporal and memory zones; `CB` and `HP` feed `PF`; `PF` feeds
`CX`; everything converges at the merge boundary that `MX` reads. Topology is a
property an auditor and a TLA+ model can both read directly.

### 5.2 Modulation (learned)
For a given input, the **router** produces:
- a zone-activation vector `a ∈ [0,1]^Z` (how strongly each zone participates),
  and
- edge-modulation weights `m_e ∈ [0,1]` for each topology edge `e` (how much
  signal flows along it).

The router learns to modulate a **fixed** topology. It cannot create edges that
the topology does not declare. This is the property that keeps the system
auditable while adaptive: a math prompt may drive `a` toward `{CB, CX, PF}`; a
narrative prompt toward `{HP, PF}`; a sensory task toward `{SM, ...}`.

### 5.3 Execution model (parallel, staggered, async gather)
Zones execute in parallel. They finish at different times (PF and CX are
typically slowest). The merge boundary uses an **async gather**: each zone
returns its output tagged with a confidence score and a logical timestamp; the
merger waits up to a configured deadline for stragglers, then composes with what
has arrived. Late zones past the deadline are recorded as `timed_out` in the
provenance trace and excluded from that pass.

This is the same gather pattern the federated case needs: on Citrate, different
nodes own different zones, submit signed outputs asynchronously, and the merge
collects them under the same deadline discipline.

## 6. Merge

The merge is the heart of the capability/efficiency tradeoff. It runs in four
ordered steps on the gathered zone outputs:

1. **Score.** Compute an attention score for each gathered zone output against
   the merge query (derived from the input + router state). Combine with the
   zone's self-reported confidence.
2. **Prune.** Drop the bottom 70–80% of contributions by combined score
   (configurable threshold). Pruning is noise rejection; pruned contributions are
   recorded in the trace with their scores so the decision is auditable.
3. **Re-weight.** Normalize the surviving scores into a composition weighting.
4. **Compose.** Combine survivors by weighted sum into the merged representation
   that produces token logits / action signal and is handed to `MX`.

The merge runs on the deterministic Q16.16 path so the same gathered set always
composes to the same result. This is required for federated reconciliation and
on-chain verification.

## 7. Provenance trace

The provenance trace is a structured, ordered record emitted alongside the model
output on every pass. Minimum fields:

```
trace {
  input_hash
  router { zone_activation[Z], edge_modulation[E] }
  zones[ {
    id, core, activated, confidence, latency_ms, status  // ok|timed_out|pruned
  } ]
  inter_zone_flows[ { from, to, strength } ]
  merge { scores[], prune_threshold, survivors[], weights[] }
  codec { verification: pass|fail|unverified, artifact_hash }
  mcp { state_transitions[], tool_calls[ {tool, args_hash, result_status} ] }
  output_hash
}
```

Properties the trace must satisfy (formalized in `03_FORMAL_SPEC_SCAFFOLD.md`):

- **Faithfulness.** Replaying the recorded zone mix and weights reproduces
  `output_hash`. (Hypothesis H-03.)
- **Completeness.** Every activated zone appears; every prune decision is
  recorded; every tool call is recorded.
- **Hashability.** The trace serializes deterministically so it can be hashed and
  committed on-chain.

On Citrate, `trace` (or its hash) becomes part of the inference transaction. An
auditor replays it against the committed weights and verifies the output. That
replayability is the opacity solution... we are not hiding the math, we are
recording it.

## 8. MCP Harness state machine

`MX` enforces an explicit state machine over tool use. No transition is skippable.

```
INPUT_VALIDATION
  → ZONE_ROUTING
    → ZONE_EXECUTION (parallel)
      → OUTPUT_AGGREGATION (merge)
        → TOOL_PRECONDITION_CHECK
          → TOOL_SELECTION
            → ACTION_GATE        // human-in-the-loop / policy gate
              → TOOL_EXECUTION
                → LOG_PROVENANCE
                  → RETURN
```

Guards:
- `TOOL_PRECONDITION_CHECK` fails closed: if preconditions are unmet, transition
  to `RETURN` with a recorded refusal.
- `ACTION_GATE` is where an approval policy (including human-in-the-loop, per the
  Citrate Agent HITL model) is enforced before any external side effect.
- A `CX` verification of `fail` blocks `TOOL_EXECUTION` for any action that
  depends on that artifact.

The state machine is deterministic and side-effect-ordered, which makes it the
cleanest target for TLA+ (safety: no side effect before `ACTION_GATE`; liveness:
every pass reaches `RETURN`).

## 9. State Space Model integration

Temporal zones (`SM`, `CB`) use SSM cores (S4/Mamba-class) for:
- **Linear-time recurrence** versus attention's quadratic cost at sequence
  length.
- **Native temporal dynamics**, which suit motor sequencing and multimodal
  binding.
- **An explicit internal state** that is logged and audited like any other zone
  signal.

Each SSM zone carries a small attention head used only for cross-zone
communication, so an SSM zone can still talk across the topology to attention
zones. The hybrid-inside-the-zone pattern (SSM recurrence + thin attention head)
is what keeps efficiency without losing composability. The gluing between SSM
and attention zones is a known training risk (Master Plan risk register);
mitigation is to stabilize SSM zones in isolation first, then unfreeze cross-zone
heads on a schedule.

## 10. Serialization and compatibility

### 10.1 Tensor container
All learned weights serialize to **GGUF** (primary) and **ONNX** (interchange).
Nothing about zone structure changes the tensor layout that a standard loader
sees... a NAT GGUF is a valid GGUF.

### 10.2 Sidecar format
A sidecar file (proposed extension `.nat.json` or embedded GGUF metadata KV)
declares:

```
sidecar {
  version
  zones[ { id, core, slice_offset, slice_width, modalities[], recipe_ref } ]
  topology { edges[ {from, to} ] }
  router { type, params_ref }
  merge { prune_threshold, deadline_ms }
  mcp { tool_registry_ref, state_machine_ref }
  composition_rules[ ... ]   // how zones may be swapped/recomposed
}
```

A loader that ignores the sidecar runs the model as an opaque transformer (the
Ollama onramp). A NAT-aware runtime reads the sidecar and runs the full
zone-partitioned pass with provenance (the ecosystem offramp). This is ADR-0004:
GGUF is the container, the sidecar carries the zone graph, the onramp stays
intact.

### 10.3 Composability
`composition_rules` declare which zones may be swapped or recomposed (the
"rip-and-replace threads" idea). A zone is swappable if its slice width and
cross-zone head contract match. This is what lets a federation evolve one zone
without retraining the whole model, and it is the structural analog of mixture-of
-experts that the design conversation was reaching for... threaded and
composable rather than a flat pool of experts behind a router.

## 11. Novelty wedge (for counsel)

Working assessment for IP review, not a legal opinion. Counsel (Taurien, Lauren)
to validate against a full prior-art search.

Prior art exists for: federated mixture-of-experts (FedMoE and related),
multimodal transformers in federated learning, brain-inspired LLMs (BriLLM and
related), and neuroscience-inspired embodied agent frameworks. What the scan did
not surface, and what NAT combines:

1. **Declared zone partitioning** of the hidden representation with fixed,
   auditable inter-zone topology (versus learned-from-scratch expert routing).
2. **Provenance trace as a first-class, hashable forward-pass output** designed
   for on-chain commitment and replay verification.
3. **An auxiliary sidecar format** that adds zone declarations, recipes, and
   composition rules on top of GGUF/ONNX while preserving backwards
   compatibility.
4. **A non-learned MCP executive harness** with an explicit, formally verifiable
   state machine gating tool use, integrated into the same provenance record.

The defensible combination is "auditable-by-construction, GGUF-compatible,
zone-partitioned transformer with on-chain-verifiable provenance and a
formally-specified tool-use harness." Each piece may have neighbors in the
literature; the combination and the on-chain replay-verification angle are the
wedge. The Formal scaffold turns these into claim-shaped statements counsel can
work from.

## 12. Open questions carried into build

- Slice widths per zone... fixed, or learned/allocated during training?
- Router architecture... lightweight MLP gate, or a small attention gate?
- Memory store interface for `HP`... in-context only at v1, external KV later?
- Codec verification depth... syntax + tests at v1, formal proofs later?
- Exact prune threshold... 70% or 80%, set by L1 ablation, not by assertion.

These are logged as hypotheses/ADRs (Research Strategy §5–6), resolved by
measurement, not by this document.
