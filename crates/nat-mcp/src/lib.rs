//! The MCP Harness (`MX`) — the non-learned executive layer (Architecture §8).
//!
//! `MX` is not a learned zone. It is a deterministic state machine plus a
//! validator. It is the most straightforwardly formalizable component of NAT and
//! the direct counterpart of `formal/McpHarness.tla`. The two safety invariants
//! it enforces in code are exactly the two the TLA+ proves:
//!
//! - `NoUngatedSideEffect` — no external side effect occurs before the action
//!   gate approves. Encoded by the `gate_passed` guard on `TOOL_EXECUTION`.
//! - `NoExecOnFailedCodec` — a `fail` from the Codec zone can never reach tool
//!   execution. Encoded by the codec check inside the action gate.
//!
//! Both fail *closed*: an unmet precondition or a denied gate transitions
//! straight to `RETURN` with a recorded refusal, never to execution.

use nat_types::Verification;
use serde::{Deserialize, Serialize};

/// The explicit states of the harness (Architecture §8). No transition is
/// skippable; every pass walks this machine to `RETURN`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum State {
    InputValidation,
    ZoneRouting,
    ZoneExecution,
    OutputAggregation,
    ToolPreconditionCheck,
    ToolSelection,
    ActionGate,
    ToolExecution,
    LogProvenance,
    Return,
}

impl State {
    pub fn as_str(self) -> &'static str {
        match self {
            State::InputValidation => "INPUT_VALIDATION",
            State::ZoneRouting => "ZONE_ROUTING",
            State::ZoneExecution => "ZONE_EXECUTION",
            State::OutputAggregation => "OUTPUT_AGGREGATION",
            State::ToolPreconditionCheck => "TOOL_PRECONDITION_CHECK",
            State::ToolSelection => "TOOL_SELECTION",
            State::ActionGate => "ACTION_GATE",
            State::ToolExecution => "TOOL_EXECUTION",
            State::LogProvenance => "LOG_PROVENANCE",
            State::Return => "RETURN",
        }
    }
}

/// What the harness needs to decide a pass. In a real deployment these come from
/// the merged zone output, the tool registry, the Codec verification, and the
/// approval policy; at L0 they are supplied directly so the state machine can be
/// exercised in isolation (which is the point — it is non-learned).
#[derive(Debug, Clone)]
pub struct McpInput {
    /// A tool was selected by the merged signal (None → nothing to gate; the
    /// pass still walks to RETURN).
    pub tool: Option<String>,
    /// Tool preconditions are satisfied (registry has it, args validate).
    pub preconditions_met: bool,
    /// The action gate's approval decision (human-in-the-loop / policy).
    pub approved: bool,
    /// The Codec zone's verification of the artifact this tool depends on.
    pub codec_verified: Verification,
    /// Hash of the tool arguments, for the trace.
    pub args_hash: String,
}

/// The outcome of one harness run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpOutcome {
    /// The ordered states visited — recorded verbatim in the provenance trace.
    pub transitions: Vec<State>,
    /// Did an external side effect actually occur? True only if the gate passed.
    pub side_effected: bool,
    /// If the harness failed closed, why.
    pub refusal: Option<String>,
    /// The executed tool call, if any (tool, args_hash, result_status).
    pub tool_call: Option<(String, String, String)>,
}

impl McpOutcome {
    /// SAFETY (`NoUngatedSideEffect`): a side effect implies the gate passed.
    /// Holds for every possible run; checked in tests and proven in TLA+.
    pub fn no_ungated_side_effect(&self, gate_passed: bool) -> bool {
        !self.side_effected || gate_passed
    }
}

/// Run the harness to completion. Deterministic; always reaches `RETURN`
/// (`AlwaysReturns`). The transition list is the audit record.
pub fn run(input: &McpInput) -> McpOutcome {
    let mut transitions = vec![
        State::InputValidation,
        State::ZoneRouting,
        State::ZoneExecution,
        State::OutputAggregation,
        State::ToolPreconditionCheck,
    ];

    // No tool selected → nothing to gate. Walk straight to RETURN.
    let Some(tool) = input.tool.clone() else {
        transitions.push(State::Return);
        return McpOutcome {
            transitions,
            side_effected: false,
            refusal: None,
            tool_call: None,
        };
    };

    // TOOL_PRECONDITION_CHECK fails closed.
    if !input.preconditions_met {
        transitions.push(State::Return);
        return McpOutcome {
            transitions,
            side_effected: false,
            refusal: Some("tool preconditions not met".into()),
            tool_call: None,
        };
    }

    transitions.push(State::ToolSelection);
    transitions.push(State::ActionGate);

    // ACTION_GATE: approve only if the policy approved AND codec did not fail.
    // This single condition encodes both safety invariants.
    let gate_passed = input.approved && input.codec_verified != Verification::Fail;
    if !gate_passed {
        let why = if input.codec_verified == Verification::Fail {
            "codec verification failed; dependent tool execution blocked"
        } else {
            "action gate not approved"
        };
        transitions.push(State::Return);
        return McpOutcome {
            transitions,
            side_effected: false,
            refusal: Some(why.into()),
            tool_call: None,
        };
    }

    // Gate passed → the only path on which a side effect may occur.
    transitions.push(State::ToolExecution);
    transitions.push(State::LogProvenance);
    transitions.push(State::Return);
    McpOutcome {
        transitions,
        side_effected: true,
        refusal: None,
        tool_call: Some((tool, input.args_hash.clone(), "ok".into())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> McpInput {
        McpInput {
            tool: Some("write_file".into()),
            preconditions_met: true,
            approved: true,
            codec_verified: Verification::Pass,
            args_hash: "deadbeef".into(),
        }
    }

    #[test]
    fn happy_path_executes_and_reaches_return() {
        let out = run(&base());
        assert!(out.side_effected);
        assert_eq!(*out.transitions.last().unwrap(), State::Return);
        assert!(out.transitions.contains(&State::ToolExecution));
        assert!(out.no_ungated_side_effect(true));
    }

    #[test]
    fn no_side_effect_before_gate() {
        // Gate not approved → no execution, fail closed, still reaches RETURN.
        let mut input = base();
        input.approved = false;
        let out = run(&input);
        assert!(!out.side_effected);
        assert!(!out.transitions.contains(&State::ToolExecution));
        assert_eq!(*out.transitions.last().unwrap(), State::Return);
        assert!(out.refusal.is_some());
    }

    #[test]
    fn failed_codec_blocks_execution() {
        // NoExecOnFailedCodec: even with approval, a codec fail blocks the tool.
        let mut input = base();
        input.codec_verified = Verification::Fail;
        let out = run(&input);
        assert!(!out.side_effected);
        assert!(out.refusal.unwrap().contains("codec"));
    }

    #[test]
    fn precondition_failure_fails_closed() {
        let mut input = base();
        input.preconditions_met = false;
        let out = run(&input);
        assert!(!out.side_effected);
        assert!(!out.transitions.contains(&State::ToolSelection));
    }

    #[test]
    fn every_pass_reaches_return() {
        // Across the cartesian product of decisions, RETURN is always reached
        // and a side effect never occurs without a passed gate.
        for tool in [None, Some("t".to_string())] {
            for pre in [false, true] {
                for appr in [false, true] {
                    for cv in [
                        Verification::Pass,
                        Verification::Fail,
                        Verification::Unverified,
                    ] {
                        let input = McpInput {
                            tool: tool.clone(),
                            preconditions_met: pre,
                            approved: appr,
                            codec_verified: cv,
                            args_hash: "x".into(),
                        };
                        let out = run(&input);
                        assert_eq!(*out.transitions.last().unwrap(), State::Return);
                        let gate = tool.is_some() && pre && appr && cv != Verification::Fail;
                        assert!(out.no_ungated_side_effect(gate));
                    }
                }
            }
        }
    }
}
