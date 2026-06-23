# Hypothesis ledger

Every capability or design claim that is not yet proven (Research Method §6).
Format: `H-id | statement | status: open|supported|refuted | evidence`.

- **H-01** | Zone partitioning does not reduce capability per parameter versus a
  dense baseline of equal size. | **supported (real-data L1, 5/5 seeds)** | *the
  load-bearing bet.* DATA-S1 WP-D6: the real `NatTrainModel` vs an equal-param dense
  transformer (20718≈20701), both **mini-batch-trained on the 1.12M-token PD corpus**
  (next-byte LM), capability on a **held-out** split, 5 seeds, GPU. **H-01 HOLDS,
  5/5 seeds** — NAT held-out loss 2.88–2.91 < dense 2.97–2.99: partitioning *beats*
  dense per parameter on real text. (The WP-5 synthetic read was a marginal 3/5;
  real data is decisive.) Caveat: small byte-LM 3-zone scale (~20K params); L2 scale
  is future. If a larger run refutes it, change course.
  **Re-confirmed on a harder corpus (2026-06-22, `corpus-v3`):** the same ablation on
  the grown **1.91M-token** corpus — now including code (Rust Book + anyhow/itertools/
  serde) and SICP, a more diverse and higher-entropy distribution — **still HOLDS 5/5
  seeds** (mean cap/param nat 1.575e-5 > dense 1.537e-5; NAT loss 3.058–3.074 < dense
  3.138–3.148). Losses are higher than the prose-only run (the distribution is harder),
  but the NAT-over-dense **gap persists**, so the hold is not an artifact of an
  easy/prose-only corpus. Still L1 small scale; L2 remains the open question.

- **H-02** | Context-aware routing produces measurably different zone mixes for
  different prompt classes. | **supported (held-out at L1)** | NAT-S2 WP-3 — the
  trained `LearnedRouter` separates the battery (11.70 vs L0 4.25, in-sample); and on
  an extended battery split train/held-out, the trained router still beats the L0
  baseline on prompts it never saw (**3.10 vs 2.63**, `nat-eval` `h02_heldout`). It
  generalizes, not memorizes. Full-scale labeled batteries remain the L2 read.

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
