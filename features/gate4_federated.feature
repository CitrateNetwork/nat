# Gate 4 — Federated proof (Formal Scaffold §B.5). NOT YET IMPLEMENTED (L3
# research milestone). The async signed gather across nodes reuses the same
# deadline discipline proven in formal/AsyncGather.tla; settlement of the
# resulting contributions is owned by citrate-compute-pool
# (docs/SETTLEMENT_SEAM.md), not by NAT.
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

  Scenario: Reward weight follows compute and data quality
    Given a node submits a signed contribution with metered compute and a data-quality score
    When citrate-compute-pool settles the contribution
    Then the reward weight equals compute multiplied by data quality
    And a zero data-quality score yields zero reward weight
