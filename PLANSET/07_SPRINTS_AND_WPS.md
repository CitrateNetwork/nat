# Sprints & Work Packages

The build follows the five-gate pattern (Master Plan §5) under the Agentile
red-test-first protocol: a work package writes its acceptance criteria (Gherkin
+ TLA+) first, then the code that turns them green.

## Sprint 0 — First full pass (Gate 1 + Gate 2) — **DONE**

The bootstrap sprint: lock the theory and stand up the L0 forward pass that
emits a provenance trace, green against its acceptance suite.

| WP | Deliverable | Status | Evidence |
|----|-------------|--------|----------|
| WP-0.1 | Repo scaffold: workspace, pinned toolchain, deny.toml, CI, Tier-1 `AUDIT_TIER.md` | ✅ | `Cargo.toml`, `.github/workflows/ci.yml`, `AUDIT_TIER.md` |
| WP-0.2 | Shared primitives + Q16.16 deterministic fixed-point | ✅ | `nat-types` (9 tests) |
| WP-0.3 | The provenance trace: schema, deterministic hash, canonical merge decision, decision-faithful replay | ✅ | `nat-provenance` (4 tests) |
| WP-0.4 | The non-learned MCP harness state machine + safety invariants | ✅ | `nat-mcp` (5 tests) |
| WP-0.5 | The sidecar (`.nat.json`): zone graph, topology validation, `export_kind` | ✅ | `nat-sidecar` (4 tests) |
| WP-0.6 | The L0 forward pass: toy SSM/attention cores, router, async gather, deterministic merge, trace assembly | ✅ | `nat-core` (14 tests) |
| WP-0.7 | Settlement-seam accounting type + eval harness skeleton | ✅ | `nat-train`, `nat-eval` (4 tests) |
| WP-0.8 | TLA+ modules (Merge, AsyncGather, McpHarness) + `.cfg` | ✅ (written, TLC pending JRE) | `formal/` |
| WP-0.9 | Gherkin features by gate + Gate-2 acceptance suite | ✅ | `features/`, `crates/nat-core/tests/gate2_*.rs` |
| WP-0.10 | Agentile planset: gates.yaml, hypotheses, ADRs, first case study | ✅ | `.agentile/planset/` |
| WP-0.11 | Settlement seam spec (compute-pool integration) | ✅ | `docs/SETTLEMENT_SEAM.md` |

**Gate-2 exit (met):** the L0 forward pass passes `features/gate2_*.feature`
(forward pass 4/4, async gather 2/2, MCP harness 3/3); `cargo test` is green
(49 tests); clippy clean. **Gate-1 exit (partially met):** planset + formal
modules + features complete; the one open item is running TLC (needs a JRE in
CI) and counsel sign-off on the claim-shaped statements.

## Sprint 1 — Gate 3 (Trainable and portable) — next

L1 work: a ~1–2B model trains on the Spark, exports to GGUF, round-trips through
an Ollama-class harness, and the routing-differentiation metric beats baseline.

- WP-1.1 — Candle-backed cores behind the `ZoneCore` trait (replace L0 toys).
  **DONE** (`nat-candle`, ADR-0010): `CandleSsmCore` + `CandleAttentionCore` (real
  Candle matmul/softmax, deterministic, trait-conformant, GPU-ready by device
  swap) + `train_tiny_zone_head` proving forward + autodiff + AdamW on CPU. L1
  framework locked to Candle (HF ecosystem, GGUF-native). De-risks critique #6.
  **Wired in:** `NatModel` takes a pluggable `CoreFactory` (toy default,
  `nat_candle::CandleCores` injectable via `with_cores`/`candle_model`). The core
  **backend is recorded in every provenance trace** (`toy-l0` | `candle-cpu`) and
  `uses_toy_cores()` is the hard guard — so a real/DGX run cannot silently fall
  back to toys, and an auditor can verify which backend produced any trace.
- WP-1.2 — `nat-train` real loop + reproducibility floor (config hash, seed, hw).
  **Reproducibility floor DONE** (`nat-train::repro`): `RunConfig::config_hash`
  (hardware-independent logical-run anchor), `Hardware::detect`, `ReproRecord`
  with a stable instance hash + the exact rerun command. Open at L1: the real
  Burn/Candle training loop that emits these per run.
- WP-1.3 — **H-01 ablation** vs a dense baseline of equal params (ADR-0005
  protocol). *This is the bet-deciding work.* **Harness DONE** (`nat-ablation`):
  partitioned vs dense Candle arms, **param-matched in code** (refuses the run if
  it can't match — ADR-0005 enforced, not assumed), trained identically, with a
  capability-per-param verdict + repro hash + recorded backend. `guard_not_toy`
  refuses a toy-backed model. Runs on CPU now (`cargo run -p nat-ablation
  --example ablation`); the DGX swaps in the full NatModel + real corpus for the
  conclusive verdict. Toy-scale numbers are illustrative only.
- WP-1.4 — GGUF export (`FlattenedDense`) + sidecar; Ollama load test (Gate-3 feature).
- WP-1.5 — `nat-eval` routing-differentiation over labeled prompt batteries.
  **Harness DONE** (`nat-eval`): a 4-class battery, per-class activation
  centroids, between/within separation ratio + `differentiates(threshold)`, and
  decision-faithfulness over the battery. At L0 the hand-wired router already
  separates classes (ratio ≈ 4.3); the same harness runs against the trained L1
  router for the real H-02 verdict.
- WP-1.6 — data pipeline (INGEST→…→MANIFEST). **Skeleton DONE** (`nat-data`):
  quality scoring (the economic signal), rule-based zone tagging, exact+near-dup,
  license/PII gates, deterministic order-independent sharding + manifest hash, and
  the end-to-end settlement loop (pipeline quality → `StepContribution` →
  `reward_weight`). Open at L1: real corpora ingestion + a real tokenizer +
  model-based quality filters.

## Sprint 2 — Gate 4 (Federated proof)

L3 research milestone: multi-node signed async gather toward the shared model;
on-chain provenance verifies; contributions settle through compute-pool.

- WP-2.1 — wall-clock async signed gather across nodes (the L0 simulated gather
  becomes real, same deadline discipline).
- WP-2.2 — on-chain trace-hash commitment + auditor replay path.
- WP-2.3 — settlement integration: emit `StepContribution` into compute-pool;
  end-to-end "compute × quality → reward weight" (Gate-4 feature).

## Sprint 3 — Gate 5 (Productize)

Console ships from the visual brief (`06_VISUAL_DESIGN_BRIEF.md`); docs
patent-filed; L2 (~10B) run scheduled with committed federation + cloud-burst
compute.
