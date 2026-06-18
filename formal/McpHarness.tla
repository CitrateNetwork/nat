------------------------------- MODULE McpHarness -------------------------------
\* The MCP harness tool-use state machine (Architecture §8, Formal Scaffold §A.3).
\*
\* Proves the two safety properties counsel and a security reviewer care about
\* most, plus termination:
\*   NoUngatedSideEffect  — no external side effect before the action gate passes.
\*   NoExecOnFailedCodec  — a failed codec verification never reaches execution.
\*   AlwaysReturns        — every pass reaches RETURN.
\* The Rust counterpart is crates/nat-mcp/src/lib.rs.
EXTENDS Naturals

CONSTANTS PreconditionsMet,   \* boolean: tool preconditions hold
          Approved,           \* boolean: the action gate approved
          CodecResult         \* "pass" | "fail" | "unverified"

VARIABLES state,          \* current state
          sideEffected,   \* boolean: an external side effect has occurred
          gatePassed      \* boolean: ACTION_GATE approved this pass

vars == <<state, sideEffected, gatePassed>>

States == { "INPUT_VALIDATION","ZONE_ROUTING","ZONE_EXECUTION",
            "OUTPUT_AGGREGATION","TOOL_PRECONDITION_CHECK","TOOL_SELECTION",
            "ACTION_GATE","TOOL_EXECUTION","LOG_PROVENANCE","RETURN" }

Init ==
  /\ state = "INPUT_VALIDATION"
  /\ sideEffected = FALSE
  /\ gatePassed = FALSE

\* The linear prefix: validation → routing → execution → aggregation → precheck.
Linear ==
  \/ /\ state = "INPUT_VALIDATION"   /\ state' = "ZONE_ROUTING"
     /\ UNCHANGED <<sideEffected, gatePassed>>
  \/ /\ state = "ZONE_ROUTING"       /\ state' = "ZONE_EXECUTION"
     /\ UNCHANGED <<sideEffected, gatePassed>>
  \/ /\ state = "ZONE_EXECUTION"     /\ state' = "OUTPUT_AGGREGATION"
     /\ UNCHANGED <<sideEffected, gatePassed>>
  \/ /\ state = "OUTPUT_AGGREGATION" /\ state' = "TOOL_PRECONDITION_CHECK"
     /\ UNCHANGED <<sideEffected, gatePassed>>

\* TOOL_PRECONDITION_CHECK fails closed to RETURN.
Precheck ==
  /\ state = "TOOL_PRECONDITION_CHECK"
  /\ \/ /\ PreconditionsMet  /\ state' = "TOOL_SELECTION"
     \/ /\ ~PreconditionsMet /\ state' = "RETURN"
  /\ UNCHANGED <<sideEffected, gatePassed>>

Select ==
  /\ state = "TOOL_SELECTION"
  /\ state' = "ACTION_GATE"
  /\ UNCHANGED <<sideEffected, gatePassed>>

\* ACTION_GATE: pass only if Approved AND the codec did not fail. Both safety
\* properties are encoded in this single guard.
Gate ==
  /\ state = "ACTION_GATE"
  /\ \/ /\ Approved /\ CodecResult # "fail"
        /\ gatePassed' = TRUE  /\ state' = "TOOL_EXECUTION"
     \/ /\ (~Approved \/ CodecResult = "fail")
        /\ gatePassed' = FALSE /\ state' = "RETURN"   \* fail closed
  /\ UNCHANGED sideEffected

\* TOOL_EXECUTION is guarded: it is reachable only with gatePassed = TRUE.
Execute ==
  /\ state = "TOOL_EXECUTION"
  /\ gatePassed = TRUE                 \* GUARD
  /\ sideEffected' = TRUE
  /\ state' = "LOG_PROVENANCE"
  /\ UNCHANGED gatePassed

Log ==
  /\ state = "LOG_PROVENANCE"
  /\ state' = "RETURN"
  /\ UNCHANGED <<sideEffected, gatePassed>>

\* RETURN is absorbing (stutter for infinite behaviors / liveness).
Return ==
  /\ state = "RETURN"
  /\ UNCHANGED vars

Next == Linear \/ Precheck \/ Select \/ Gate \/ Execute \/ Log \/ Return

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

------------------------------------------------------------------------------
\* Invariants and properties.

TypeOK ==
  /\ state \in States
  /\ sideEffected \in BOOLEAN
  /\ gatePassed \in BOOLEAN

\* SAFETY (load-bearing): a side effect implies the gate passed.
NoUngatedSideEffect == sideEffected => gatePassed

\* SAFETY: a failed codec can never coincide with reaching tool execution.
NoExecOnFailedCodec == (state = "TOOL_EXECUTION") => (CodecResult # "fail")

\* LIVENESS: every pass reaches RETURN.
AlwaysReturns == <>(state = "RETURN")
==============================================================================
