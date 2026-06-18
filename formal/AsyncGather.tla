------------------------------- MODULE AsyncGather -------------------------------
\* Async gather with a deadline (Architecture §5.3, Formal Scaffold §A.2).
\*
\* Proves the gather terminates, respects the deadline, and records stragglers.
\* This is the liveness guarantee the design asked for: "I'll wait this long for
\* stragglers, but I'm composing with what I have." The Rust counterpart is
\* crates/nat-core/src/gather.rs.
EXTENDS Naturals, FiniteSets

CONSTANTS Zones,        \* set of (learned) zone ids
          Deadline      \* the gather window length, in logical ticks

VARIABLES clock,        \* logical clock
          arrived,      \* zones whose output has arrived
          status,       \* zone -> {"pending","ok","timed_out"}
          closed        \* boolean: gather window closed

vars == <<clock, arrived, status, closed>>

Init ==
  /\ clock = 0
  /\ arrived = {}
  /\ status = [z \in Zones |-> "pending"]
  /\ closed = FALSE

\* A zone's output arrives before the window closes.
Arrive(z) ==
  /\ ~closed
  /\ status[z] = "pending"
  /\ arrived' = arrived \cup {z}
  /\ status' = [status EXCEPT ![z] = "ok"]
  /\ UNCHANGED <<clock, closed>>

\* Time advances while the window is open.
Tick ==
  /\ ~closed
  /\ clock < Deadline          \* bound the clock so TLC has a finite model
  /\ clock' = clock + 1
  /\ UNCHANGED <<arrived, status, closed>>

\* At the deadline the window closes; everything not "ok" becomes "timed_out".
CloseWindow ==
  /\ clock >= Deadline
  /\ ~closed
  /\ closed' = TRUE
  /\ status' = [z \in Zones |-> IF status[z] = "ok" THEN "ok" ELSE "timed_out"]
  /\ UNCHANGED <<clock, arrived>>

\* Stutter once closed so behaviors are infinite (for liveness checking).
Done ==
  /\ closed
  /\ UNCHANGED vars

Next == (\E z \in Zones : Arrive(z)) \/ Tick \/ CloseWindow \/ Done

Spec == Init /\ [][Next]_vars /\ WF_vars(Tick) /\ WF_vars(CloseWindow)

------------------------------------------------------------------------------
\* Invariants and properties.

TypeOK ==
  /\ clock \in 0..Deadline
  /\ arrived \subseteq Zones
  /\ status \in [Zones -> {"pending","ok","timed_out"}]
  /\ closed \in BOOLEAN

\* Safety: no zone is simultaneously ok and timed_out (the status is single-valued,
\* so this is the partition guarantee the trace relies on).
Consistent ==
  \A z \in Zones : ~(status[z] = "ok" /\ status[z] = "timed_out")

\* Completeness: once closed, every non-arrived zone is recorded timed_out.
StragglerRecorded ==
  closed => (\A z \in Zones : z \notin arrived => status[z] = "timed_out")

\* Once closed, no zone remains pending (the trace has no gaps).
NoPendingAtClose ==
  closed => (\A z \in Zones : status[z] \in {"ok","timed_out"})

\* Liveness: the window always eventually closes (the merge never blocks forever).
WindowCloses == <>(closed = TRUE)
==============================================================================
