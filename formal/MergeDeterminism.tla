--------------------------- MODULE MergeDeterminism ---------------------------
\* Merge determinism (Architecture §6, Formal Scaffold §A.1).
\*
\* The federation-critical property: the merge composes the same gathered set to
\* the same result, and pruning is correct and complete. The Rust counterpart is
\* nat_provenance::prune_and_reweight (the single canonical decision used to both
\* produce and verify the trace).
\*
\* Model: zones are 1..N, each with a fixed Score. The gather nondeterministically
\* yields a nonempty subset (so TLC enumerates the whole subset lattice); the merge
\* then computes survivors *deterministically*. Ties are broken by zone index, so
\* the preference order is strict and total — which is exactly what makes the
\* decision reproducible across nodes (no float, no hash-map iteration).
EXTENDS Naturals, FiniteSets

CONSTANTS N,            \* zones are 1..N
          Score,        \* [1..N -> Nat] : the combined score per zone
          KeepCount     \* how many survive the prune (the discretized 1-PruneThreshold)

Zones == 1..N

\* A concrete score for model checking: zone z scores z (distinct, monotone). The
\* .cfg overrides the Score constant with this (TLC cfg cannot hold a function
\* literal directly).
ScoreFcn == [z \in Zones |-> z]

ASSUME N \in Nat /\ N >= 1
ASSUME KeepCount \in Nat /\ KeepCount >= 1
ASSUME Score \in [Zones -> Nat]

VARIABLES gathered,     \* the nonempty set of zones that arrived before deadline
          survivors,    \* zones surviving the prune
          pruned        \* zones gathered but pruned

vars == <<gathered, survivors, pruned>>

\* Strict total preference: w ranks strictly above z. Ties (equal score) are
\* broken by the smaller zone index, so there is never an ambiguous boundary.
Prefers(w, z) ==
  \/ Score[w] > Score[z]
  \/ (Score[w] = Score[z] /\ w < z)

\* How many gathered zones strictly outrank z.
RankAbove(g, z) == Cardinality({ w \in g : w # z /\ Prefers(w, z) })

\* keep = min(KeepCount, |g|); at least 1 because KeepCount >= 1 and g nonempty.
Keep(g) == IF KeepCount <= Cardinality(g) THEN KeepCount ELSE Cardinality(g)

\* A zone survives iff fewer than `keep` gathered zones outrank it.
SurvivorsOf(g) == { z \in g : RankAbove(g, z) < Keep(g) }

\* The merge is a one-shot deterministic computation over a chosen gathered set.
Init ==
  /\ gathered \in (SUBSET Zones \ {{}})        \* nonempty subset; TLC enumerates all
  /\ survivors = SurvivorsOf(gathered)
  /\ pruned = gathered \ SurvivorsOf(gathered)

Next == UNCHANGED vars                          \* absorbing; the result is final

Spec == Init /\ [][Next]_vars

------------------------------------------------------------------------------
\* Invariants (checked across every reachable gathered set).

TypeOK ==
  /\ gathered \subseteq Zones
  /\ survivors \subseteq gathered
  /\ pruned \subseteq gathered

\* Pruning keeps exactly `Keep(gathered)` zones — the top fraction, no more, no less.
PruneCorrect == Cardinality(survivors) = Keep(gathered)

\* Completeness: every gathered zone is either a survivor or recorded as pruned,
\* and the two sets are disjoint (the trace has no gaps and no double-counts).
TraceComplete ==
  /\ survivors \cup pruned = gathered
  /\ survivors \cap pruned = {}

\* Determinism (the federation-critical property): survivors are a *function* of
\* the gathered set. Recomputing from the same input yields the same result —
\* checked here by confirming the stored survivors equal the recomputed survivors
\* for every reachable state.
DeterministicDecision == survivors = SurvivorsOf(gathered)

\* The prune boundary is unambiguous: no two distinct zones tie on rank, so the
\* survivor/pruned split can never depend on iteration order.
NoRankTies ==
  \A w, z \in gathered : (w # z) => (RankAbove(gathered, w) # RankAbove(gathered, z))
==============================================================================
