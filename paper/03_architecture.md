# §3 The NAT architecture

This section is normative for the design. We describe the six zones, the fixed-topology
hybrid router, the parallel execution and async gather, the deterministic merge, the
non-learned executive harness, and the serialization that keeps NAT inside the existing
inference ecosystem. Every component named here corresponds to a module in the reference
implementation (`nat-core`, `nat-candle`); we give the crate/file where it lives.

## 3.1 Zones

NAT partitions a transformer's hidden representation of width `D` into a fixed set of
**named zones**, each owning a contiguous slice of width `D_z`, its own sequence core, its
own input projection, and its own internal state. There are six zones; five are learned and
one is a non-learned executive harness.

| Zone | Role | Core |
|------|------|------|
| `SM` Sensorimotor | ingest + temporally bind multimodal input | state-space (SSM) |
| `CB` Cerebellar | timing, sequencing, learned reflex | state-space (SSM) |
| `HP` Hippocampal | memory consolidation, novelty/salience | attention |
| `PF` Prefrontal | reasoning, planning, language (deepest) | attention |
| `CX` Codec | reasoning → verifiable executable logic | attention |
| `MX` MCP harness | validate / sequence / route tool use | **none** (state machine) |

The mapping to brain regions is a **mimetic analog**, not a fidelity claim. It earns its
place by suggesting a useful division of labor — fast temporal binding versus slow deep
reasoning — and the architecture is free to abandon it wherever the engineering is better
served (a stance we hold to throughout; see §8). The load-bearing engineering facts are: the
zones are *declared*, the partition is *fixed at build time*, and the assignment of a
representational slice to a named function is a property of the architecture rather than a
discovery made after training.

All six zones are declared and the forward pass is exercised end-to-end on all six (the L0
reference); but the *capability* evidence in §6 is on the three data-rich zones `{HP, PF, CX}`.
This is deliberate staging (ADR-0008): the multimodal `SM` and timing `CB` zones have the thinnest
data at this scale, so we hold them out of the load-bearing ablation until "the data earns it,"
and the only five-zone exercise to date is the scale-ladder loss probe of §6.3, not a capability
comparison. The architecture is six zones; the measured claims are three.

Core choice by function follows ADR-0002. Temporal zones (`SM`, `CB`) use state-space cores
[Gu & Dao 2023] for linear-time recurrence and an explicit, loggable internal state; reasoning
zones (`HP`, `PF`, `CX`) use attention for content-addressable look-back. In the reference
implementation the SSM recurrence `h_t = a·h_{t-1} + b·x_t, y_t = c·h_t` is computed as a
single lower-triangular matmul `y = M·x` (`nat-candle::cores`), which is vectorized and
device-agnostic — the same op graph runs on CPU and on the GB10's GPU by swapping the Candle
device. Each SSM zone carries a thin attention head used only for cross-zone communication, so
an SSM zone can still talk across the topology to an attention zone.

The core *backend* is pluggable behind a `CoreFactory` trait (`nat-core::cores`): the L0
toy cores validate the architecture; the L1 Candle cores train. Which backend ran is recorded
in every provenance trace (`trace.backend ∈ {toy-l0, candle-cpu, candle-cuda}`), and a
`uses_toy_cores()` guard lets the ablation and any deployment refuse a toy-backed run — so a
measured result can never silently be a toy artifact (§4, §5).

## 3.2 Routing: fixed topology, learned modulation

The permitted inter-zone edges are **declared in the sidecar and fixed at build time**. The
default topology is

```
SM → CB     SM → HP     SM → PF
CB → PF     HP → PF     PF → CX
(all learned zones) → the merge boundary that MX reads
```

`SM` feeds the temporal and memory zones; `CB` and `HP` feed `PF`; `PF` feeds `CX`; everything
converges at the merge boundary. This graph is a property an auditor and a TLA+ model can both
read directly.

For each input the **router** produces a zone-activation vector `a ∈ [0,1]^Z` (how strongly
each zone participates) and an edge-modulation weight `m_e ∈ [0,1]` for each *declared* edge
(how much signal flows along it). The crucial property — the one that keeps the system
auditable while adaptive — is that **the router can only modulate edges the topology declares;
it cannot create an edge that is not there.** In the implementation this is structural: the
router iterates `sidecar.topology.edges` and nothing else, so an undeclared edge has no code
path to receive a weight (`nat-core::router`). A math prompt may drive `a` toward `{CB, CX,
PF}`, a narrative prompt toward `{HP, PF}`, a sensory task toward `{SM, …}` — adaptivity
without the opacity of learned-from-scratch routing. At L0 the router is a fixed deterministic
function of cheap class signals; at L1 it is a trained gate (`LearnedRouter`, WP-3), and its
differentiation is measured in §6 (H-02).

This is the first claim-shaped statement for counsel (C-1): *a transformer whose hidden
representation is partitioned into declared named zones, each owning a core from {attention,
state-space}, communicating over a fixed declared topology whose per-input edge strengths are
produced by a learned router that cannot create undeclared edges.*

## 3.3 Parallel execution and async gather

Zones execute in parallel; they finish at different times (the deep `PF` and `CX` are typically
slowest). The merge boundary uses an **async gather**: each zone returns its output tagged with
a confidence score and a status; the merger waits up to a configured deadline, then composes
with whatever arrived. A zone that misses the deadline is recorded `timed_out` and excluded
from that pass — the merge never blocks on a straggler. This is the same gather discipline the
federated case needs (§7): on a network, different nodes own different zones and submit signed
outputs asynchronously under the same deadline. The gather's safety (no zone both `ok` and
`timed_out`) and liveness (the window always closes) are stated and machine-checkable in
`formal/AsyncGather.tla` (§4).

## 3.4 The merge

The merge is where the capability/efficiency trade-off is realized. It runs four ordered steps
on the gathered zone outputs (`nat-core::merge`):

1. **Score.** Compute a combined score for each gathered output (an attention score against a
   merge query derived from the input and router state, combined with the zone's self-reported
   confidence).
2. **Prune.** Drop the bottom 70–80% of contributions by score (a configurable threshold).
   Pruning is noise rejection; **pruned contributions are recorded in the trace with their
   scores**, so the decision is auditable rather than silent.
3. **Re-weight.** Normalize the surviving scores into a composition weighting summing to one.
4. **Compose.** Combine the survivors by weighted sum into the merged representation that
   produces token logits and is handed to `MX`.

Steps 2–3 are a single canonical function (`nat-provenance::prune_and_reweight`) used both to
*produce* the trace and to *verify* it, so the verification in §4 is not circular. The compose
step runs on the **deterministic Q16.16 fixed-point path** (`nat-types::Q16`): a value `v` is
the integer `round(v · 2^16)` in an `i64`, with `i128` intermediates for multiply. Integer
arithmetic is bit-identical across CPUs and across federated nodes, where IEEE-754 float is not
— this is what lets independently-computed merges reconcile and on-chain provenance verify (the
same motivation as Citrate's verifiable-inference substrate, Paper X). The training-time merge
(`nat-candle`, WP-2) is a differentiable form **reconciled to** this Q16.16 provenance merge: a
battery test pins that its hardened survivor set equals the canonical `prune_and_reweight`
survivors (the soft top-k via temperature-annealed softmax converges to the hard recorded survivors
as `τ → 0`), so the model the gradient sees and the model the verifier replays make the **same
survivor decision** — the composition weights are soft during training and anneal toward the hard
recorded weights.

## 3.5 The MCP harness

Tool use and external actions are mediated by a **non-learned** executive harness (`MX`,
`nat-mcp`) — a deterministic state machine plus a validator, the mimetic analog of executive
control. It consumes the merged signal, checks tool preconditions, enforces an action gate, and
emits a validated tool call or a recorded refusal. Its states are walked in order
(`INPUT_VALIDATION → ZONE_ROUTING → ZONE_EXECUTION → OUTPUT_AGGREGATION →
TOOL_PRECONDITION_CHECK → TOOL_SELECTION → ACTION_GATE → TOOL_EXECUTION → LOG_PROVENANCE →
RETURN`), with two safety guards that fail closed: **no external side effect occurs before the
action gate approves**, and **a failed Codec verification can never reach tool execution**.
Because the harness is non-learned and side-effect-ordered, it is the most straightforwardly
formalizable component; `formal/McpHarness.tla` states exactly these two invariants
(`NoUngatedSideEffect`, `NoExecOnFailedCodec`) plus termination, and they are claim-shaped
(C-4). The split — five learned zones plus one non-learned executive — is where the safety and
determinism story concentrates.

## 3.6 Serialization: the GGUF sidecar

Backwards compatibility is the adoption onramp: if the model cannot load in an Ollama-class
runtime, the ecosystem cannot adopt it. NAT keeps **GGUF/ONNX as the tensor container** and adds
an auxiliary **sidecar** (`.nat.json`, `nat-sidecar`) that declares the zone graph, topology,
router/merge parameters, and composition rules (ADR-0004). A sidecar-unaware runtime runs the
tensors as an opaque transformer (the onramp); a sidecar-aware runtime runs the full
zone-partitioned pass with provenance (the offramp). We are precise about the status here, because it is
not yet built: a literal zone-partitioned graph with parallel heterogeneous SSM+attention zones
does not serialize to a layout `llama.cpp` runs as-is, so the ecosystem onramp depends on a
**flattened-dense** export. That export is **specified, not yet implemented** (Gate-3 item
`g3-gguf`, WP-1.4): the sidecar declares an `export_kind` field whose `FlattenedDense` variant is
reserved for it, but only the `ZonePartitioned` form is produced today, and the GGUF round-trip and
Ollama-class load are not yet demonstrated. The sidecar is the source of truth for the zone graph
in either case; the claim that NAT "runs in the existing inference ecosystem" is therefore a design
target, not a measured result. The composition rules (a zone is swappable when
its slice width and cross-zone contract match) are what let a federation evolve one zone without
retraining the whole model (§7), and are the structural analog of mixture-of-experts the design
was reaching for — *threaded and composable* rather than a flat pool of experts behind a router.
