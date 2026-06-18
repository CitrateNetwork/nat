//! Gate-2 acceptance: the MCP harness gates tool use.
//!
//! Maps to `features/gate2_mcp_harness.feature` (Formal Scaffold §B.3,
//! McpHarness.tla). No external side effect occurs without approval, and a
//! failed codec verification blocks dependent execution.

use nat_core::{NatModel, ToolRequest};
use nat_mcp::{run, McpInput, State};
use nat_types::Verification;

/// Scenario: No side effect before the action gate.
#[test]
fn unapproved_tool_does_not_execute_and_walks_to_return() {
    let model = NatModel::l0();
    let tool = ToolRequest {
        tool: "write_file".into(),
        preconditions_met: true,
        approved: false, // gate has not approved
    };
    let r = model.forward("please write a file", Some(tool));

    assert!(!r.mcp.side_effected, "a tool executed without approval");
    assert!(!r.mcp.transitions.contains(&State::ToolExecution));
    assert_eq!(*r.mcp.transitions.last().unwrap(), State::Return);
    assert!(r.trace.mcp.refusal.is_some());
}

/// Scenario: Failed codec blocks dependent execution.
#[test]
fn failed_codec_blocks_tool_execution() {
    // Drive the harness directly with a codec `fail` (the model does not
    // synthesize a Fail at L0; the harness behavior is what Gate 2 pins).
    let out = run(&McpInput {
        tool: Some("deploy".into()),
        preconditions_met: true,
        approved: true,                     // even with approval...
        codec_verified: Verification::Fail, // ...a failed codec blocks execution.
        args_hash: "abcd".into(),
    });

    assert!(!out.side_effected);
    assert!(!out.transitions.contains(&State::ToolExecution));
    assert!(out.refusal.unwrap().contains("codec"));
}

/// Scenario: Every pass reaches RETURN.
#[test]
fn every_pass_reaches_return_and_records_transitions() {
    let model = NatModel::l0();
    let r = model.forward("just answer, no tools", None);

    assert_eq!(r.trace.mcp.state_transitions.last().unwrap(), "RETURN");
    assert!(!r.trace.mcp.state_transitions.is_empty());
    // A plain answer requests no tool, so nothing executed.
    assert!(!r.mcp.side_effected);
}
