# Critique & Remediations

A research bet earns the right to scale only if it survives an honest review.
This document records the seven substantive criticisms raised against the source
planset in the Gate-1 review, and how each is carried forward — as an ADR, a
tightened claim, or a build decision. None is fatal; together they tighten the
thesis from "asserted" to "proven or disproven."

| # | Criticism | Remediation | Where it lives |
|---|-----------|-------------|----------------|
| 1 | The participant-reward economic layer (compute × data-quality) is barely specified in the source docs. | Do **not** reinvent settlement. NAT emits compute receipt + data-quality score + provenance hash; `citrate-compute-pool` (which already ships a marketplace + tokenomics + reward settlement) settles. | `docs/SETTLEMENT_SEAM.md`, `nat_train::StepContribution`, ADR-0007 |
| 2 | H-01's dense baseline is under-specified, and H-01 is the whole bet. | Pin the baseline protocol: identical token budget, data, tokenizer, seed, compute — only partitioning differs. Make it a hard Gate-3 exit blocker. | ADR-0005, `gates.yaml` gate3 exit |
| 3 | "Provenance faithfulness" can't be bit-exact when zone cores are float. | Split the claim: **decision-faithful** (which zones/prunes/weights — always verifiable) vs **bit-faithful logits** (only under a deterministic-inference path). Patent claim C-2 reworded. | ADR-0006-faithfulness, `nat_provenance` docs, `verify_decision_faithful` |
| 4 | Federated determinism quietly conflates merge-determinism (provable) with training-convergence (statistical). | Keep them separate in all docs. The TLA+ covers merge-time gather only; training tolerance is an empirical L3 claim. | `formal/README.md`, hypotheses H-05 split |
| 5 | Six zones is a lot of unproven structure to pay for at once; SM/CB have thin L0 data. | L0 validates the pass on all six (done), but H-01's *capability* test at L1 runs first on the 3-zone {HP,PF,CX} config where data is real, then widens. | ADR-0008-zone-staging, `gates.yaml` |
| 6 | Rust-from-scratch training is a schedule risk the plan soft-pedals. | Decision (owner): Rust reference only. Mitigation: L0 numerics use a lightweight deterministic path behind the `ZoneCore` trait so Burn/Candle slot in at L1 without rework; L0 timeline is honestly "wire the pass," not "train." | ADR-0009-l0-numerics |
| 7 | The GGUF "Ollama onramp" claim can't be literally true for parallel heterogeneous zones. | Be precise: the Ollama-loadable export is a **flattened-dense** equivalent; the sidecar carries the real zone graph for NAT-aware runtimes. Recorded in the sidecar's `export_kind`. | `nat_sidecar::ExportKind`, Gate-3 feature, ADR-0004 note |

## What the review affirmed

The novelty wedge (declared zone partitioning + first-class hashable provenance +
on-chain replay verification) is sound and without clean prior art. The honest
posture (H-01 named load-bearing, the scale ladder, "the analogy serves the
engineering") is the right discipline. The gate structure correctly front-loads
the bet-deciding question. The provenance-as-output design genuinely solves
auditability by construction, and the MCP harness is the most straightforwardly
verifiable component — now proven by `formal/McpHarness.tla` and its Rust test.

The remediations above are the difference between a pitch and a program.
