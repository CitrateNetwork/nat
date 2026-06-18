# ADR-0006 — Decision-faithful vs bit-faithful provenance

**Status:** accepted · **Date:** 2026-06-18 · **Remediation:** #3

## Decision
Split the faithfulness claim into two levels, and never let the strong one imply
the weak one:

- **Decision-faithful** — replaying the recorded scores reproduces the recorded
  survivor set and weights. A pure integer computation; always holds. This is the
  product guarantee and what `nat_provenance::verify_decision_faithful` checks.
- **Bit-faithful** — re-running the full pass reproduces `output_hash`
  bit-for-bit. Holds only under a deterministic-inference path (the Q16.16 merge
  is deterministic; float zone cores are not, except in a deterministic mode).

## Why
"Replaying the logged zone mix reproduces the output" cannot be bit-exact when
SSM/attention cores run in float on a GPU. Claiming otherwise is a defect counsel
and a reviewer would catch. Patent claim C-2 is reworded to the decision-faithful
guarantee, with bit-faithful offered as an optional deterministic-inference mode.

## Effect on hypotheses
H-03 splits into H-03a (decision-faithful, supported) and H-03b (bit-faithful,
open / mode-dependent).
