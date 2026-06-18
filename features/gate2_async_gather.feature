# Gate 2 — Async gather (Formal Scaffold §B.2, formal/AsyncGather.tla).
# Acceptance: crates/nat-core/tests/gate2_async_gather.rs.
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
