# Hypothesis ledger

Every capability or design claim that is not yet proven (Research Method §6).
Format: `H-id | statement | status: open|supported|refuted | evidence`.

- **H-01** | Zone partitioning does not reduce capability per parameter versus a
  dense baseline of equal size. | **open** | *the load-bearing bet.* Tested by
  the L1 ablation (ADR-0005), gated as a Gate-3 blocker. Not measurable at L0.

- **H-02** | Context-aware routing produces measurably different zone mixes for
  different prompt classes. | **open (L0-encouraging)** | At L0 the router is
  hand-wired, not trained, yet `nat_eval::routing_divergence` already shows
  math vs narrative prompts driving different mixes. Real test is L1 over labeled
  batteries against a significance threshold.

- **H-03a** | Provenance is *decision-faithful*: replaying recorded scores
  reproduces the recorded survivors and weights. | **supported** | Proven by
  construction (`nat_provenance::prune_and_reweight` is the single decision used
  to both produce and verify) and checked: `verify_decision_faithful` + the
  Gate-2 forward-pass test. (Split from H-03 per remediation #3.)

- **H-03b** | Provenance is *bit-faithful*: re-running the full pass reproduces
  `output_hash` bit-for-bit. | **open (holds under deterministic-inference)** |
  Holds at L0 because the merge runs on the Q16.16 path and the toy cores are
  deterministic (the Gate-2 test re-runs and compares hashes). At L1 it holds
  only under a deterministic-inference mode; float zone cores break it otherwise.

- **H-04** | SSM temporal zones cut per-zone compute meaningfully versus
  attention at equal sequence length. | **open (well-supported by SSM lit)** |
  Measured by `nat-eval` at L1.

- **H-05a** | The merge composes the same gathered set to the same result
  (federation-critical determinism). | **supported** | `MergeDeterminism.tla`
  invariants + `nat-core::merge` "same gathered set → same hash" test.

- **H-05b** | A federated *training* cycle reproduces the centralized result
  within tolerance. | **open** | A statistical L3 claim, distinct from H-05a; the
  TLA+ does not cover training convergence (remediation #4). Tested at Gate 4.

## Notes

H-01 is the one that decides whether the whole bet pays off. The scale ladder
(L0/L1 on the Spark) exists to test it cheaply before the expensive L2 commit. If
H-01 refutes, honest posture says change course (Master Plan; Journal).
