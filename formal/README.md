# Formal specifications

Three TLA+ modules cover the stateful surfaces of NAT — the parts with explicit
ordering and determinism guarantees that federation and on-chain verification
depend on. The learned numeric cores (zone forward passes) are **not** modeled
here; they are tested empirically. TLA+ covers orchestration, ordering, and
determinism (Formal Scaffold §A).

| Module | Proves | Rust counterpart |
|--------|--------|------------------|
| `MergeDeterminism.tla` | the merge composes the same gathered set to the same result; pruning is correct, complete, and tie-free | `nat-provenance::prune_and_reweight` |
| `AsyncGather.tla` | the gather terminates, respects the deadline, records stragglers | `nat-core::gather` |
| `McpHarness.tla` | no side effect before the action gate; a failed codec never executes; every pass returns | `nat-mcp` |

Each module's safety invariants and the matching Rust acceptance test enforce the
**same** property at two levels: the model checks it over all reachable states;
the test checks it at runtime. For the MCP harness, `gate2_mcp_harness.rs` walks
the same cartesian product of decisions the `.cfg` constants range over.

## Running TLC

```sh
# Requires Java + tla2tools.jar (TLC was NOT run in the bootstrap environment,
# which had no Java runtime — the modules are written to be checkable, not yet
# checked). CI should run these once a JRE is available.
java -cp tla2tools.jar tlc2.TLC -config AsyncGather.cfg      AsyncGather.tla
java -cp tla2tools.jar tlc2.TLC -config McpHarness.cfg       McpHarness.tla
java -cp tla2tools.jar tlc2.TLC -config MergeDeterminism.cfg MergeDeterminism.tla
```

`MergeDeterminism` and `AsyncGather` are finite by construction. `McpHarness`
fixes its constants per run; to cover the decision space, run it across the
combinations of `PreconditionsMet` / `Approved` / `CodecResult` (the Rust test
covers the same space at runtime, so the two are cross-checks).

## Honest status

These modules complete the skeletons in `03_FORMAL_SPEC_SCAFFOLD.md` (the source
planset) into self-contained, checkable form with all operators defined. They
have **not** been run through TLC in this environment (no JRE). Gate 1's formal
exit criterion is "TLC green on all three" — wiring a JRE into CI and recording
the run is the first open item against `gates.yaml`.
