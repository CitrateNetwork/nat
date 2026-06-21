# nat — Citrate Neuroarchitectural Transformer

**RFC-CIT-NAT-0001** · Draft v0.1 · Owner: Larry Klosowski (@saulbuilds) · Entity: Mozi Satori / Citrate Network

A zone-partitioned transformer that stays GGUF/ONNX-compatible, emits an
on-chain-verifiable **provenance trace** of its own reasoning, and trains in a
federated cycle on Citrate. The hidden representation is partitioned into six
named **zones**; each zone runs its own core (attention or state-space),
communicates over a **fixed topology** modulated per-input by a learned router,
and the outputs are combined by an **attention-scored, noise-pruned merge**.
Every forward pass emits a structured, hashable trace of which zones fired and
why — that trace is the wedge against model opacity and the basis for on-chain
auditability.

This is a research bet, held to honest posture. The brain analogy is a design
heuristic. The load-bearing question is **H-01**: does zone partitioning cost
capability per parameter versus a dense baseline of equal size? The scale ladder
exists to answer it cheaply (L0/L1 on a Spark) before the expensive L2 run.

## The six zones

| Zone | Role | Core |
|------|------|------|
| `SM` Sensorimotor | ingest + temporally bind multimodal input | SSM |
| `CB` Cerebellar | timing, motor sequencing, learned reflex | SSM |
| `HP` Hippocampal | memory consolidation, novelty/salience | attention |
| `PF` Prefrontal | reasoning, planning, language (deepest) | attention |
| `CX` Codec | reasoning → verifiable executable logic | attention |
| `MX` MCP Harness | validate/sequence/route tool use | **non-learned** state machine |

Five learned zones plus one non-learned executive harness. The harness is where
determinism and the safety story live (no side effect before the action gate).

## Repo layout

```
PLANSET/            refined design docs (00_OVERVIEW .. 09), numbered convention
.agentile/planset/  the Agentile method: gates.yaml, hypotheses.md, ADRs, case studies
formal/             TLA+ modules + .cfg (MergeDeterminism, AsyncGather, McpHarness)
features/           Gherkin acceptance criteria, organized by gate
crates/             the Rust reference implementation (workspace)
docs/               settlement seam, design brief pointers
AUDIT_TIER.md       Tier-1 classification + obligations
```

### Crates

- `nat-types` — shared primitives (`ZoneId`, `CoreType`, `Status`, `Q16` fixed-point). No deps.
- `nat-provenance` — the trace: schema, deterministic hash, decision-faithful replay.
- `nat-mcp` — the non-learned executive harness state machine.
- `nat-sidecar` — the `.nat.json` zone-graph that wraps a GGUF/ONNX tensor container.
- `nat-core` — zones, router, async gather, deterministic merge, the forward pass.
- `nat-data` — the data pipeline (INGEST→…→MANIFEST): quality scoring, zone tagging,
  dedup, deterministic shards. Produces the `data_quality` score the reward seam uses.
- `nat-candle` — Candle-backed zone cores (CPU, GPU-ready) behind `ZoneCore`; the
  L1 training stack (ADR-0010). Kept separate so the L0 build stays Candle-free.
- `nat-ablation` — the H-01 ablation harness: zone-partitioned vs equal-param
  dense baseline under the ADR-0005 protocol. The bet-decider, GPU-free now.
- `nat-train` / `nat-eval` — training loop and eval harness (L0 stubs, wired at L1).

## Gates

NAT follows the five-gate pattern. The current target is **Gate 1 + Gate 2**.

1. **Gate 1 — Theory locked.** PLANSET, formal scaffold, hypotheses, ADRs signed.
2. **Gate 2 — Reference forward pass.** L0 runs end-to-end; the provenance trace
   validates against `features/gate2_*.feature`. **← first build target.**
3. **Gate 3 — Trainable and portable.** L1 trains on the Spark; GGUF round-trip.
4. **Gate 4 — Federated proof.** Multi-node signed gather; on-chain provenance verifies.
5. **Gate 5 — Productize.** Console ships; docs patent-filed; L2 scheduled.

Machine-readable exit criteria live in `.agentile/planset/gates.yaml`.

## Economic layer

NAT does **not** reinvent reward settlement. It emits a metered-compute receipt,
a data-quality score, and a provenance hash that `citrate-compute-pool` (which
already ships a compute marketplace, tokenomics simulation, and reward
settlement) turns into participant rewards. The interface is specified in
`docs/SETTLEMENT_SEAM.md`. Participant economic advantage is a function of
**compute contributed × data quantity/quality submitted**, settled by
compute-pool, scored by NAT.

## Build

```sh
cargo build --workspace
cargo test  --workspace      # Gate-2 acceptance tests mirror features/gate2_*.feature
cargo clippy --workspace --all-targets
```

## Status

**Gate 2 green** (L0 forward pass + provenance trace). **Sprint 1 landed**
(GPU-free): Candle training stack (`nat-candle`, ADR-0010), data pipeline +
quality scoring (`nat-data`), eval/routing harness (`nat-eval`), reproducibility
floor (`nat-train`), and the **H-01 ablation harness** (`nat-ablation`). 10
crates, 87 tests, clippy clean. Tier-1 (`AUDIT_TIER.md`).

**Picking this up on the DGX? Start at [`docs/DGX_HANDOFF.md`](docs/DGX_HANDOFF.md)** —
a zero-context onboarding: build, verify, the no-toy-cores guarantee, the GPU
device swap, and running the real H-01 ablation.

CI is verified locally in Docker (`scripts/ci-local.sh`); GitHub Actions is
pending an enterprise Actions-budget propagation (config is correct).
