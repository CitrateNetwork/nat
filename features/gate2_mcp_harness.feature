# Gate 2 — MCP harness (Formal Scaffold §B.3, formal/McpHarness.tla).
# Acceptance: crates/nat-core/tests/gate2_mcp_harness.rs.
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
