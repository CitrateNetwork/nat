--------------------------- MODULE UnifiedSettlement ---------------------------
\* Unified settlement seam — the WS-0 / UNIFY-S1 conservation proof (WP-7).
\* (ADR-2026-06-27-nat-rmfl-unification binding #4: "one ledger".)
\*
\* The seam takes the gather's accepted contributions and settles each through the
\* unified ledger (NAT Settlement/ChainCommit -> co-op FederatedSettlement). This
\* module proves the two properties the ledger merge must not violate:
\*
\*   - CONSERVATION: the total reward weight settled equals the total the gather
\*     accepted — the seam neither creates nor destroys weight (no retroactive
\*     dividend, no leak). The Rust counterpart is the fixed-seed sweep
\*     seam::tests::sweep_seam_conserves_weight_and_carries_quality_order_independent.
\*   - STATE-ROOT INDEPENDENCE: settling is a learning-layer write; it never touches
\*     consensus state (the on-chain counterpart of SafetyInvariant.tla).
\*
\* Plus the liveness the cycle needs: every accepted node is eventually settled.
\* Model: a fixed set of accepted nodes, each with a Q16 reward weight (abstracted
\* to Nat) and a data-quality tag (carried, not collapsed — binding #4).
EXTENDS Naturals, FiniteSets

CONSTANTS Nodes,      \* the accepted node ids this round
          Weight,     \* [Nodes -> Nat] : each node's reward weight (compute×quality)
          Quality     \* [Nodes -> Nat] : each node's data_quality term (carried)

ASSUME Nodes # {}
ASSUME Weight \in [Nodes -> Nat]
ASSUME Quality \in [Nodes -> Nat]

\* Concrete functions for model checking (cfg overrides Weight/Quality via `<-`).
\* Distinct per-node weights (node id = its weight) so conservation is a real check,
\* not a |settled|-counting tautology — mirrors MergeDeterminism's ScoreFcn idiom.
WeightFcn == [n \in Nodes |-> n]
QualityFcn == [n \in Nodes |-> n]

VARIABLES settled,        \* set of nodes settled so far
          settledWeight,  \* running total reward weight settled
          stateRoot,      \* consensus state root (must never change here)
          phase           \* "settling" | "done"

vars == <<settled, settledWeight, stateRoot, phase>>

\* Deterministic sum of weights over a set (order-free: integer + is assoc+comm).
RECURSIVE SumWeight(_)
SumWeight(S) ==
  IF S = {} THEN 0
  ELSE LET n == CHOOSE x \in S : TRUE
       IN Weight[n] + SumWeight(S \ {n})

TotalWeight == SumWeight(Nodes)

\* An arbitrary fixed consensus root; the learning-layer settlement must never
\* modify it (state-root independence, structurally).
RootConst == 42

Init ==
  /\ settled = {}
  /\ settledWeight = 0
  /\ stateRoot = RootConst
  /\ phase = "settling"

\* Settle one not-yet-settled accepted node: add its weight, leave consensus alone.
\* The node's data_quality (Quality[n]) is in scope at settlement — carried to the
\* ledger, never collapsed into the weight before this point.
Settle(n) ==
  /\ phase = "settling"
  /\ n \in Nodes
  /\ n \notin settled
  /\ settled' = settled \cup {n}
  /\ settledWeight' = settledWeight + Weight[n]
  /\ phase' = IF settled \cup {n} = Nodes THEN "done" ELSE "settling"
  /\ UNCHANGED stateRoot

Done == phase = "done" /\ UNCHANGED vars

Next == (\E n \in Nodes : Settle(n)) \/ Done

\* Weak fairness drains the settlement — every accepted node is eventually paid.
Spec == Init /\ [][Next]_vars /\ WF_vars(\E n \in Nodes : Settle(n))

------------------------------------------------------------------------------
\* Invariants + liveness.

TypeOK ==
  /\ settled \subseteq Nodes
  /\ settledWeight \in Nat
  /\ stateRoot \in Nat
  /\ phase \in {"settling","done"}

\* CONSERVATION (running): the total settled is always exactly the sum of the
\* settled set's weights — no node's weight is lost or double-counted mid-round.
ConservationInProgress == settledWeight = SumWeight(settled)

\* CONSERVATION (at close): when the round is done, every accepted node is settled
\* and the total settled equals the gather's total — the ledger merge conserves.
ConservationAtClose ==
  (phase = "done") => (settled = Nodes /\ settledWeight = TotalWeight)

\* NO PHANTOM PAYOUT: the seam never settles a node that was not accepted (the
\* settled set stays within the accepted domain — also the domain of Quality, so a
\* settled node always has a carried data_quality term).
NoPhantomPayout == settled \subseteq DOMAIN Quality

\* STATE-ROOT INDEPENDENCE: settlement is a learning-layer write; consensus state
\* is invariant across the whole run (cf. SafetyInvariant.tla::StateRootIndependent).
StateRootIndependent == stateRoot = RootConst

\* LIVENESS: every accepted node is eventually settled (the cycle never strands a
\* contributor mid-round).
AllSettled == <>(phase = "done")

THEOREM TypeSafe    == Spec => []TypeOK
THEOREM ConservRun  == Spec => []ConservationInProgress
THEOREM ConservClose == Spec => []ConservationAtClose
THEOREM NoPhantom   == Spec => []NoPhantomPayout
THEOREM RootStable  == Spec => []StateRootIndependent
THEOREM Liveness    == Spec => AllSettled
==============================================================================
