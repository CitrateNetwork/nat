--------------------- MODULE GradientAggregationAdversarial ---------------------
\* Verifiable gradient aggregation — adversarial / liveness half of the WS-1 pair.
\* (FEDERATED_METALEARNING_METAPLAN.md WS-1; AGG-S1 WP-1.)
\*
\* Companion to GradientAggregation.tla (which proves the reduction is a
\* deterministic, bit-reproducible function). This module adds the adversary
\* action space + the Byzantine-robustness guarantee of the bucketed coordinate-
\* wise trimmed mean, and the outer-step barrier's liveness.
\*
\* The robustness theorem (frontier bet #1's security half): if the per-end trim
\* budget β covers the Byzantine count f (β >= f) and the honest set dominates,
\* the aggregate is pinned inside the honest value band — Byzantine workers, even
\* choosing values freely, CANNOT flip the result outside what honest workers
\* would have produced. Intuition (proved by the invariant): a Byzantine value
\* below the honest minimum has fewer than β values beneath it, so it is always
\* trimmed; symmetric at the top. The kept band ⊆ [honest_lo, honest_hi].
\*
\* Single-coordinate abstraction (multi-coord lifts per-coordinate, as in
\* BelnapAdversarial). Rust target: nat/crates/nat-aggregate robustness proptests
\* + fuzz_gradient_aggregate (AGG-S1 WP-2/WP-3); on-chain challenge path
\* generalizes DisputeResolution.sol's bisection game (AGG-S1 WP-4).
EXTENDS Naturals, FiniteSets

CONSTANTS N,            \* workers are 1..N
          V,            \* value domain per coordinate is 0..V
          TrimCount,    \* β: extremes trimmed from EACH end
          MaxByzantine, \* f: max workers the adversary controls
          HonestLo,     \* honest values lie in [HonestLo, HonestHi]
          HonestHi      \* (the true pseudo-gradient ± bounded honest noise)

Workers == 1..N

ASSUME N \in Nat /\ N >= 1
ASSUME V \in Nat
ASSUME TrimCount \in Nat
ASSUME N > 2 * TrimCount
ASSUME MaxByzantine \in Nat /\ MaxByzantine >= 0
ASSUME HonestLo \in Nat /\ HonestHi \in Nat
ASSUME 0 <= HonestLo /\ HonestLo <= HonestHi /\ HonestHi <= V

VARIABLES mode,         \* [Workers -> {"HONEST","BYZANTINE"}]
          submitted,    \* [Workers -> BOOLEAN]
          submission,   \* [Workers -> 0..V]
          agg,          \* the committed aggregate (0 sentinel until aggregated)
          phase         \* "submitting" | "aggregated"

vars == <<mode, submitted, submission, agg, phase>>

HonestSet    == { v \in Workers : mode[v] = "HONEST" }
ByzantineSet == { v \in Workers : mode[v] = "BYZANTINE" }
HonestDominate == Cardinality(HonestSet) > Cardinality(ByzantineSet)

------------------------------------------------------------------------------
\* The same deterministic trimmed-mean operator as GradientAggregation.tla.

Below(val, w, z) ==
  \/ val[w] < val[z]
  \/ (val[w] = val[z] /\ w < z)

RankOf(val, w) == Cardinality({ z \in Workers : z # w /\ Below(val, z, w) })

TrimmedOf(val) ==
  { w \in Workers : RankOf(val, w) >= TrimCount
                 /\ RankOf(val, w) <= (N - 1) - TrimCount }

RECURSIVE SumOver(_, _)
SumOver(val, S) ==
  IF S = {} THEN 0
  ELSE LET w == CHOOSE x \in S : TRUE
       IN val[w] + SumOver(val, S \ {w})

AggOf(val) == SumOver(val, TrimmedOf(val)) \div Cardinality(TrimmedOf(val))

------------------------------------------------------------------------------
\* State machine: workers submit (honest in-band, Byzantine free) → aggregate.

Init ==
  /\ mode \in [Workers -> {"HONEST","BYZANTINE"}]
  /\ Cardinality(ByzantineSet) <= MaxByzantine
  /\ submitted = [w \in Workers |-> FALSE]
  /\ submission = [w \in Workers |-> 0]
  /\ agg = 0
  /\ phase = "submitting"

\* Honest workers submit a value inside the honest band.
HonestSubmit(w) ==
  /\ phase = "submitting"
  /\ w \in HonestSet
  /\ ~submitted[w]
  /\ \E hv \in HonestLo..HonestHi :
        submission' = [submission EXCEPT ![w] = hv]
  /\ submitted' = [submitted EXCEPT ![w] = TRUE]
  /\ UNCHANGED <<mode, agg, phase>>

\* Byzantine workers submit anything in the value domain.
ByzantineSubmit(w) ==
  /\ phase = "submitting"
  /\ w \in ByzantineSet
  /\ ~submitted[w]
  /\ \E bv \in 0..V :
        submission' = [submission EXCEPT ![w] = bv]
  /\ submitted' = [submitted EXCEPT ![w] = TRUE]
  /\ UNCHANGED <<mode, agg, phase>>

\* The outer-step barrier closes once every worker has submitted, and the
\* deterministic reduction is committed. (Straggler/timeout handling is proved
\* separately by AsyncGather.tla; here all participants submit within the window.)
Aggregate ==
  /\ phase = "submitting"
  /\ \A w \in Workers : submitted[w]
  /\ agg' = AggOf(submission)
  /\ phase' = "aggregated"
  /\ UNCHANGED <<mode, submitted, submission>>

Done == phase = "aggregated" /\ UNCHANGED vars

Next ==
  \/ \E w \in Workers : HonestSubmit(w)
  \/ \E w \in Workers : ByzantineSubmit(w)
  \/ Aggregate
  \/ Done

\* Weak fairness drains the submissions and fires the barrier — the outer step
\* never deadlocks, so the aggregation window always eventually closes.
Spec ==
  /\ Init
  /\ [][Next]_vars
  /\ WF_vars(\E w \in Workers : HonestSubmit(w))
  /\ WF_vars(\E w \in Workers : ByzantineSubmit(w))
  /\ WF_vars(Aggregate)

------------------------------------------------------------------------------
\* Invariants + the liveness property.

TypeOK ==
  /\ mode \in [Workers -> {"HONEST","BYZANTINE"}]
  /\ Cardinality(ByzantineSet) <= MaxByzantine
  /\ submitted \in [Workers -> BOOLEAN]
  /\ submission \in [Workers -> 0..V]
  /\ agg \in 0..V
  /\ phase \in {"submitting","aggregated"}

\* THE ROBUSTNESS THEOREM: under honest dominance with the trim budget covering
\* the Byzantine count (β >= f), the committed aggregate is pinned inside the
\* honest value band. Byzantine workers cannot flip it outside [HonestLo,HonestHi].
ByzantineCannotFlipUnderHonestDominance ==
  (phase = "aggregated"
   /\ HonestDominate
   /\ TrimCount >= Cardinality(ByzantineSet))
  => (HonestLo <= agg /\ agg <= HonestHi)

\* HONEST UNANIMITY PRESERVED: if every honest worker submits the same value H,
\* the aggregate is exactly H (Byzantine extremes are all trimmed). The sharp
\* form of the robustness theorem.
HonestUnanimityPreserved ==
  (phase = "aggregated"
   /\ HonestDominate
   /\ TrimCount >= Cardinality(ByzantineSet)
   /\ \E H \in HonestLo..HonestHi : \A h \in HonestSet : submission[h] = H)
  => (\E H \in HonestLo..HonestHi :
        (\A h \in HonestSet : submission[h] = H) /\ agg = H)

\* The aggregate never escapes the value domain regardless of adversary control
\* (no overflow, no alias of an out-of-range slot).
NoSpontaneousEscape == agg \in 0..V

\* LIVENESS (WindowCloses): the barrier always eventually closes — the outer
\* aggregation step is guaranteed to commit, the federation round never blocks.
WindowCloses == <>(phase = "aggregated")

THEOREM TypeSafe   == Spec => []TypeOK
THEOREM Robust     == Spec => []ByzantineCannotFlipUnderHonestDominance
THEOREM Unanimity  == Spec => []HonestUnanimityPreserved
THEOREM NoEscape   == Spec => []NoSpontaneousEscape
THEOREM Liveness   == Spec => WindowCloses
==============================================================================
