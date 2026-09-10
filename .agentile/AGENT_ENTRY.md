# Agent entry point — nat

You are working in the Citrate Neuroarchitectural Transformer repo. Read this
first.

## Orient

1. `README.md` — what NAT is, the six zones, the gates, the layout.
2. `PLANSET/00_OVERVIEW.md` — the planset map; read in the order it gives.
3. `.agentile/planset/gates.yaml` + `hypotheses.md` — what must be true to
   advance, and what is still a bet. **H-01 is the load-bearing bet.**
4. `PLANSET/08_CRITIQUE_AND_REMEDIATIONS.md` — the seven open tensions and how
   each is carried. Do not relitigate; extend.

## Rules of the house (Agentile)

- **Red-test-first.** A work package writes its acceptance criteria (Gherkin in
  `features/`, TLA+ in `formal/`) before the code that turns them green.
- **Every claim anchors to evidence.** A capability claim points at a
  measurement; a design claim points at an ADR or a prior-art reference. No
  evidence yet → label it a hypothesis, not a result.
- **Determinism where it matters.** The merge and tool-routing path run on
  `nat_types::Q16` (Q16.16), never `f32`. Do not introduce float into the
  trace-hash or merge path.
- **The trace is the product.** Anything that changes what a forward pass records
  touches `nat-provenance` and is Tier-1 (`AUDIT_TIER.md`).
- **Honest posture.** If H-01 refutes, say so and change course. That is the
  point of the scale ladder.

## Build

```sh
cargo test --workspace      # includes the Gate-2 acceptance suite
cargo clippy --workspace --all-targets -- -D warnings
```

## Current state

Gate 2 is green (L0 forward pass). Gate 1 is partial (planset + formal + features
done; TLC run and counsel sign-off open). Next is Sprint 1 / Gate 3 — and the
H-01 ablation, the bet-deciding work (`PLANSET/07_SPRINTS_AND_WPS.md`).
