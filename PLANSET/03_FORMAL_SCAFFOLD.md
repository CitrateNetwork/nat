# Formal Specification Scaffold — TLA+ & Gherkin

**Document:** RFC-CIT-NAT-0001 / Formal Scaffold
**Status:** Draft v0.1
**Companion to:** `02_ARCHITECTURE_SPEC.md`
**Audience:** verification engineers, ML engineers, IP counsel

---

## 0. How to read this

This document is the bridge between the prose architecture and three concrete
work products:

1. **TLA+ modules** that state the safety and liveness properties of the parts of
   NAT that have explicit state... the merge, the async gather, and the MCP
   harness state machine. These are skeletons, sized to be filled in and checked
   with TLC.
2. **Gherkin feature files** with acceptance criteria, organized by gate. These
   are the executable definition of done for each gate.
3. **Claim-shaped statements** for counsel, derived from the invariants, so the
   novelty in `02_ARCHITECTURE_SPEC.md §11` can be drafted into patent claims.

Terminology is normative from the Architecture spec §3. The scaffold reuses it
exactly.

---

## PART A — TLA+ Modules

Three modules cover the stateful surfaces. The learned numeric cores (zone
forward passes) are not modeled in TLA+... they are tested empirically. TLA+
covers orchestration, ordering, and the determinism guarantees that federation
and on-chain verification depend on.

### A.1 `MergeDeterminism.tla`

Goal: prove that the merge composes the same gathered set to the same result,
and that pruning is monotone and complete.

```tla
---------------------------- MODULE MergeDeterminism ----------------------------
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS Zones,            \* set of zone ids
          PruneThreshold    \* fraction in (0,1), e.g. drop bottom 0.8

VARIABLES gathered,         \* set of zone outputs that arrived before deadline
          scored,           \* function: zone -> score (deterministic)
          survivors,        \* set of zones surviving prune
          weights,          \* function: survivor -> normalized weight
          output            \* composed result handle (hash)

TypeOK ==
  /\ gathered \subseteq Zones
  /\ scored \in [gathered -> Nat]
  /\ survivors \subseteq gathered
  /\ weights \in [survivors -> Nat]

\* Pruning keeps exactly the top (1 - PruneThreshold) fraction by score.
PruneCorrect ==
  \A z \in gathered :
    z \in survivors <=> ScoreRank(z, scored) >= CutoffRank(gathered, PruneThreshold)

\* Determinism: the composed output is a pure function of (survivors, weights).
ComposeDeterministic ==
  output = Compose(survivors, weights)

\* Completeness: every gathered zone is either a survivor or recorded as pruned.
TraceComplete ==
  \A z \in gathered : (z \in survivors) \/ Recorded(z, "pruned")

Invariant == TypeOK /\ PruneCorrect /\ ComposeDeterministic /\ TraceComplete

\* SAME-SET-SAME-OUTPUT (the federation-critical property):
\* two passes with identical gathered sets and scores produce identical output.
DeterminismTheorem ==
  \A g1, g2 :
    (g1 = g2) => (Compose(Survivors(g1), Weights(g1)) = Compose(Survivors(g2), Weights(g2)))
==============================================================================
```

`Compose`, `ScoreRank`, `CutoffRank`, `Survivors`, `Weights`, `Recorded` are
operators to define against the Q16.16 deterministic path. The load-bearing
property is `DeterminismTheorem`... it is what lets federated nodes and on-chain
verifiers agree.

### A.2 `AsyncGather.tla`

Goal: prove the gather terminates, respects the deadline, and records stragglers.

```tla
------------------------------- MODULE AsyncGather -------------------------------
EXTENDS Naturals, FiniteSets

CONSTANTS Zones, Deadline

VARIABLES clock,
          arrived,          \* zones whose output has arrived
          status,           \* zone -> {pending, ok, timed_out}
          closed            \* boolean: gather window closed

Init ==
  /\ clock = 0
  /\ arrived = {}
  /\ status = [z \in Zones |-> "pending"]
  /\ closed = FALSE

Arrive(z) ==
  /\ ~closed
  /\ status[z] = "pending"
  /\ arrived' = arrived \cup {z}
  /\ status' = [status EXCEPT ![z] = "ok"]
  /\ UNCHANGED <<clock, closed>>

Tick ==
  /\ ~closed
  /\ clock' = clock + 1
  /\ UNCHANGED <<arrived, status, closed>>

CloseWindow ==
  /\ clock >= Deadline
  /\ ~closed
  /\ closed' = TRUE
  /\ status' = [z \in Zones |->
                 IF status[z] = "ok" THEN "ok" ELSE "timed_out"]
  /\ UNCHANGED <<clock, arrived>>

Next == (\E z \in Zones : Arrive(z)) \/ Tick \/ CloseWindow

\* Safety: no zone is both ok and timed_out.
Consistent == \A z \in Zones : ~(status[z] = "ok" /\ status[z] = "timed_out")

\* Liveness: the window always closes (the merge never blocks forever).
WindowCloses == <>(closed = TRUE)

\* Every non-arrived zone at close is recorded timed_out (completeness for trace).
StragglerRecorded ==
  closed => (\A z \in Zones : z \notin arrived => status[z] = "timed_out")
==============================================================================
```

`WindowCloses` is the liveness guarantee the design conversation asked for...
"I'll wait this long for stragglers, but I'm composing with what I have."

### A.3 `McpHarness.tla`

Goal: prove the tool-use state machine never produces a side effect before the
action gate, and always terminates.

```tla
------------------------------- MODULE McpHarness -------------------------------
EXTENDS Naturals

CONSTANTS Tools

VARIABLES state,            \* current state in the machine
          sideEffected,     \* boolean: has an external side effect occurred
          gatePassed,       \* boolean: ACTION_GATE approved
          codecVerified     \* {pass, fail, unverified}

States == { "INPUT_VALIDATION","ZONE_ROUTING","ZONE_EXECUTION",
            "OUTPUT_AGGREGATION","TOOL_PRECONDITION_CHECK","TOOL_SELECTION",
            "ACTION_GATE","TOOL_EXECUTION","LOG_PROVENANCE","RETURN" }

Init ==
  /\ state = "INPUT_VALIDATION"
  /\ sideEffected = FALSE
  /\ gatePassed = FALSE
  /\ codecVerified \in {"pass","fail","unverified"}

\* Transitions are linear with fail-closed branches to RETURN.
Step ==
  \/ /\ state = "TOOL_PRECONDITION_CHECK"
     /\ \/ /\ PreconditionsMet  /\ state' = "TOOL_SELECTION"
        \/ /\ ~PreconditionsMet /\ state' = "RETURN"   \* fail closed
     /\ UNCHANGED <<sideEffected, gatePassed, codecVerified>>
  \/ /\ state = "ACTION_GATE"
     /\ \/ /\ Approved /\ codecVerified # "fail"
           /\ gatePassed' = TRUE /\ state' = "TOOL_EXECUTION"
        \/ /\ (~Approved \/ codecVerified = "fail")
           /\ state' = "RETURN"                         \* fail closed
     /\ UNCHANGED <<sideEffected, codecVerified>>
  \/ /\ state = "TOOL_EXECUTION"
     /\ gatePassed = TRUE                                \* GUARD
     /\ sideEffected' = TRUE
     /\ state' = "LOG_PROVENANCE"
     /\ UNCHANGED <<gatePassed, codecVerified>>
  \/ ...   \* remaining linear transitions

\* SAFETY (the load-bearing invariant): no side effect without a passed gate.
NoUngatedSideEffect == sideEffected => gatePassed

\* SAFETY: a failed codec verification can never reach tool execution.
NoExecOnFailedCodec == (state = "TOOL_EXECUTION") => (codecVerified # "fail")

\* LIVENESS: every pass reaches RETURN.
AlwaysReturns == <>(state = "RETURN")
==============================================================================
```

`NoUngatedSideEffect` and `NoExecOnFailedCodec` are the two safety properties
counsel and a security reviewer will care about most. They are also directly
claim-shaped (see Part C).

---

## PART B — Gherkin Feature Files

Organized by gate. Each `.feature` is the acceptance definition for that gate.
Scenarios are written so a test engineer can implement steps directly.

### B.1 Gate 2 — Reference forward pass

```gherkin
Feature: Zone-partitioned forward pass with provenance
  As an ML engineer
  I want the six-zone forward pass to run end to end
  So that the architecture is validated and the trace is emitted

  Background:
    Given a NAT model at rung "L0"
    And the sidecar declares zones "SM,CB,HP,PF,CX,MX"
    And the topology declares edges per the architecture default

  Scenario: All zones execute in parallel over one input
    When I run inference on a text prompt
    Then each learned zone "SM,CB,HP,PF,CX" produces an output
    And each output carries a confidence score and a latency
    And the MCP harness "MX" does not run a learned core

  Scenario: The router modulates a fixed topology
    When I run inference
    Then the router emits a zone-activation vector of length 6
    And the router emits edge-modulation weights only for declared edges
    And no signal flows along an edge absent from the topology

  Scenario: The merge scores, prunes, and composes
    Given gathered outputs from the activated zones
    When the merge runs
    Then it assigns each gathered output a combined score
    And it prunes the bottom 70 to 80 percent by score
    And every pruned zone is recorded in the trace with its score
    And the surviving weights normalize to 1

  Scenario: The provenance trace is complete and hashable
    When inference completes
    Then the trace lists every activated zone with status "ok|timed_out|pruned"
    And the trace records merge scores, prune threshold, and survivors
    And serializing the trace twice yields the same hash
```

### B.2 Gate 2 — Async gather

```gherkin
Feature: Async gather with deadline
  So that the merge never blocks on a slow zone

  Scenario: Gather closes at the deadline
    Given zone "PF" will not return before the deadline
    When inference runs with a configured gather deadline
    Then the gather window closes at the deadline
    And "PF" is recorded with status "timed_out"
    And the merge composes from the zones that arrived

  Scenario: Straggler completeness
    When the gather window closes
    Then every zone not arrived is recorded "timed_out"
    And no zone is both "ok" and "timed_out"
```

### B.3 Gate 2 — MCP harness

```gherkin
Feature: MCP harness gates tool use
  So that no external side effect occurs without approval

  Scenario: No side effect before the action gate
    Given a tool call is selected
    When the action gate has not approved
    Then no tool executes
    And the harness transitions toward "RETURN"

  Scenario: Failed codec blocks dependent execution
    Given the Codec zone returns verification "fail"
    And a selected tool depends on that artifact
    When the harness reaches the action gate
    Then the tool does not execute
    And the refusal is recorded in the trace

  Scenario: Every pass reaches RETURN
    When inference runs to completion
    Then the harness state reaches "RETURN"
    And the state transitions are recorded in the trace
```

### B.4 Gate 3 — Trainable and portable

```gherkin
Feature: GGUF round-trip and Ollama-class load
  So that the ecosystem onramp holds

  Scenario: NAT exports to valid GGUF
    Given a trained NAT model at rung "L1"
    When I export to GGUF with the sidecar
    Then a standard GGUF loader loads the tensor container without error

  Scenario: Sidecar-unaware runtime runs the model opaquely
    Given an Ollama-class runtime that ignores the sidecar
    When it loads the NAT GGUF
    Then it runs inference as an opaque transformer
    And it produces coherent output

  Scenario: NAT-aware runtime runs the full zone pass
    Given a NAT-aware runtime
    When it loads the same GGUF plus sidecar
    Then it runs the six-zone pass
    And it emits the provenance trace

  Scenario: Routing differentiates by prompt class
    Given prompt classes "math", "narrative", "sensory"
    When I run inference on each class
    Then the dominant activated zones differ by class
    And the difference exceeds the configured significance threshold
```

### B.5 Gate 4 — Federated proof

```gherkin
Feature: Federated training cycle on Citrate
  So that distributed nodes train toward one shared model

  Scenario: Async signed gather across nodes
    Given two nodes each owning different zones
    When they submit signed zone outputs asynchronously
    Then the merge gathers them under the deadline discipline
    And the signatures verify before composition

  Scenario: Federated result matches centralized within tolerance
    Given a centralized reference result for a fixed seed and config
    When the federated cycle runs on the same data partition
    Then the federated result matches the reference within tolerance

  Scenario: On-chain provenance verifies
    Given a completed federated inference
    When the trace hash is committed on Citrate
    Then an auditor replays the trace against the committed weights
    And the replay reproduces the output hash
```

---

## PART C — Claim-shaped statements (for counsel)

Working language for IP review only. Counsel (Taurien, Lauren) to assess against
a full prior-art search and convert into properly drafted claims. Each maps to an
invariant or feature above.

- **C-1 (zone partitioning + fixed topology).** A transformer in which the hidden
  representation is partitioned into declared named zones, each owning a core
  selected from {attention, state-space}, executing in parallel and communicating
  over a fixed declared topology whose per-input edge strengths are produced by a
  learned router that cannot create undeclared edges. *(Architecture §4–5;
  Gate 2 routing feature.)*

- **C-2 (provenance as verifiable output).** A method whereby a single forward
  pass emits a structured, deterministically-serializable provenance record of
  zone activations, scores, prune decisions, inter-zone flows, and tool routing,
  such that replaying the record against committed weights reproduces the output,
  and the record hash is committable to a distributed ledger for third-party
  verification. *(Architecture §7; MergeDeterminism `DeterminismTheorem`;
  Gate 4 on-chain feature.)*

- **C-3 (backwards-compatible sidecar).** A serialization in which learned
  weights are stored in a standard tensor container (GGUF/ONNX) and an auxiliary
  sidecar declares the zone graph, topology, training recipes, and composition
  rules, such that a sidecar-unaware runtime executes the model as an opaque
  transformer while a sidecar-aware runtime executes the zone-partitioned pass.
  *(Architecture §10; Gate 3 GGUF features.)*

- **C-4 (formally-gated tool-use harness).** A non-learned executive harness
  governing tool use via an explicit state machine with the safety properties
  that no external side effect occurs before an approval gate and that a failed
  deterministic-verification result blocks dependent execution. *(Architecture
  §8; McpHarness `NoUngatedSideEffect`, `NoExecOnFailedCodec`.)*

- **C-5 (composable zones).** A composition mechanism whereby an individual zone
  may be swapped or replaced without retraining the whole model when its slice
  width and cross-zone contract match, enabling federated evolution of single
  zones. *(Architecture §10.3.)*

The defensible combination, per Architecture §11, is C-1 through C-4 together:
auditable-by-construction, GGUF-compatible, zone-partitioned, with
on-chain-verifiable provenance and a formally-specified tool harness.
