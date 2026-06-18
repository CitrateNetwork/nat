# Gate 2 — Reference forward pass (Formal Scaffold §B.1).
# Acceptance is realized by crates/nat-core/tests/gate2_forward_pass.rs, one
# Rust test per scenario. (Cucumber step-binding is deferred to L1; at Gate 2 the
# executable definition of done is the mapped integration test.)
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
