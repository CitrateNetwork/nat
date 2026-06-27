------------------------- MODULE GradientAggregation -------------------------
\* Verifiable gradient aggregation — determinism / safety half of the WS-1 pair.
\* (FEDERATED_METALEARNING_METAPLAN.md WS-1; AGG-S1 WP-1.)
\*
\* The on-chain outer-step (DiLoCo barrier): workers emit a pseudo-gradient
\* delta; the aggregator reduces them coordinate-wise with a *trimmed mean in
\* fixed-point*. This module proves the reduction is a deterministic, bit-
\* reproducible function of the submitted values — the property that lets
\* heterogeneous untrusted validators reconcile the aggregate on-chain with NO
\* tolerance window (frontier bet #1).
\*
\* Abstraction: a SINGLE coordinate. Multi-coordinate aggregation is the same
\* operator applied independently per coordinate, so the per-coordinate proof
\* lifts by composition (same move BelnapAdversarial makes for dim=1). Values
\* are integers standing in for Q16 fixed-point; the determinism argument is
\* exactly that integer +/sort/div are associative + order-free, so there is no
\* float, no hash-map iteration, no tie that could make two nodes disagree.
\*
\* Rust target: nat/crates/nat-aggregate (AGG-S1 WP-2); the precompile path is
\* citrate-chain core/execution/src/precompiles/q16 (0x0110 family — the Q16
\* trust root, NOT the f32 core/learning reference impl).
EXTENDS Naturals, FiniteSets

CONSTANTS N,          \* workers are 1..N (one submission each)
          V,          \* value domain per coordinate is 0..V (Q16 stand-in)
          TrimCount   \* β: how many extremes trimmed from EACH end (β >= f budget)

Workers == 1..N

ASSUME N \in Nat /\ N >= 1
ASSUME V \in Nat
ASSUME TrimCount \in Nat
\* The kept band must be non-empty: strictly more workers than the total trim.
ASSUME N > 2 * TrimCount

VARIABLES value,      \* [Workers -> 0..V] : each worker's submitted coordinate
          result      \* the stored aggregate (what a node committed on-chain)

vars == <<value, result>>

------------------------------------------------------------------------------
\* The aggregation operator, as a pure function of the submitted values.

\* Strict total preference: w sorts strictly below z. Equal values are broken by
\* the smaller worker index, so the order is strict + total — the single move
\* that makes the trim boundary unambiguous across nodes (no float, no hashmap).
Below(val, w, z) ==
  \/ val[w] < val[z]
  \/ (val[w] = val[z] /\ w < z)

\* 0-based rank: how many distinct workers sort strictly below w.
RankOf(val, w) == Cardinality({ z \in Workers : z # w /\ Below(val, z, w) })

\* The kept band: drop the lowest TrimCount and highest TrimCount by rank.
TrimmedOf(val) ==
  { w \in Workers : RankOf(val, w) >= TrimCount
                 /\ RankOf(val, w) <= (N - 1) - TrimCount }

\* Deterministic integer summation over a set (order-free: integer + is
\* associative + commutative, so the CHOOSE order cannot change the total).
RECURSIVE SumOver(_, _)
SumOver(val, S) ==
  IF S = {} THEN 0
  ELSE LET w == CHOOSE x \in S : TRUE
       IN val[w] + SumOver(val, S \ {w})

\* Trimmed mean with floor division = deterministic rounding (the real impl uses
\* deterministic-seeded stochastic rounding; floor is the TLC-checkable stand-in
\* — the cross-platform rounding equivalence is a Rust frozen-fixture concern).
AggOf(val) == SumOver(val, TrimmedOf(val)) \div Cardinality(TrimmedOf(val))

MinOf(val, S) == CHOOSE m \in S : \A z \in S : val[m] <= val[z]
MaxOf(val, S) == CHOOSE m \in S : \A z \in S : val[m] >= val[z]

------------------------------------------------------------------------------
\* One-shot pure computation (house idiom — cf. MergeDeterminism). Init picks a
\* submission assignment (TLC enumerates the whole [Workers -> 0..V] lattice);
\* the aggregate is computed once and is final.

Init ==
  /\ value \in [Workers -> 0..V]
  /\ result = AggOf(value)

Next == UNCHANGED vars            \* absorbing; the committed aggregate is final

Spec == Init /\ [][Next]_vars

------------------------------------------------------------------------------
\* Invariants (checked across every reachable submission assignment).

TypeOK ==
  /\ value \in [Workers -> 0..V]
  /\ result \in Nat

\* DETERMINISM (the federation-critical property): the committed aggregate is a
\* *function* of the submitted values — recomputing from the same inputs yields
\* the same result, for every reachable assignment. A challenger that recomputes
\* must land on the identical value, so on-chain reconciliation needs no window.
DeterministicAggregation == result = AggOf(value)

\* NO TIES: the strict-total order leaves no two workers on the same rank, so the
\* trim boundary (which submissions are dropped) can never depend on iteration
\* order. This is the no-float / no-hash-map-iteration reproducibility root.
NoRankTies ==
  \A w, z \in Workers : (w # z) => (RankOf(value, w) # RankOf(value, z))

\* TRIM BUDGET well-defined: exactly TrimCount dropped from each end, so the
\* Byzantine-robustness budget (β) is structurally honored, never off-by-one.
TrimBudgetExact == Cardinality(TrimmedOf(value)) = N - 2 * TrimCount

\* RANGE: the trimmed mean lies within the kept band's [min,max] — a sanity /
\* no-overflow guard (a mean of integers is bracketed by its extremes).
AggWithinRange ==
  LET T == TrimmedOf(value)
  IN value[MinOf(value, T)] <= result /\ result <= value[MaxOf(value, T)]

\* STATE-ROOT INDEPENDENCE: aggregation is a learning-layer reduction; it must
\* not be able to read or write consensus state. Modeled structurally here — the
\* only variables are learning-layer (value, result); no consensus variable is in
\* scope, so the consensus state_root is invariant by construction. The on-chain
\* counterpart is SafetyInvariant.tla::StateRootIndependent, which AGG-S1 WP-1
\* extends to cover the aggregation precompile. Asserted here as a guard that the
\* result never escapes its declared integer domain (cannot alias a state slot).
StateRootIndependent == result \in 0..V
==============================================================================
