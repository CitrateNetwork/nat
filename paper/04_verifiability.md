# §4 Verifiability by construction

The provenance trace is the deliverable that distinguishes NAT from both post-hoc
interpretability and after-the-fact verifiable inference. This section makes the claim
precise: what the trace records, the exact sense in which it is faithful, how it is committed
and replayed, how it relates to Citrate's verifiable-inference substrate, and what the formal
specifications cover.

## 4.1 The trace

Every forward pass emits a structured provenance trace alongside the model output
(`nat-provenance::Trace`). Its fields:

```
trace {
  input_hash
  backend                                   // toy-l0 | candle-cpu | candle-cuda
  router { zone_activation[Z], edge_modulation[E] }
  zones[ { id, core, activated, confidence, latency_ms, status } ]  // ok|timed_out|pruned
  inter_zone_flows[ { from, to, strength } ]
  merge { scores[], prune_threshold, survivors[], weights[] }
  codec { verification: pass|fail|unverified, artifact_hash }
  mcp { state_transitions[], tool_calls[ {tool, args_hash, result_status} ] }
  output_hash
}
```

The trace must satisfy three properties. **Completeness**: every activated zone appears, every
prune decision is recorded with its score, every tool call is recorded. **Hashability**: the
trace serializes deterministically (stable field order, fixed-point values encoded as raw
integers, no hash-map iteration) so the same trace always hashes to the same digest. And
**faithfulness**, which we are careful to split into two claims.

## 4.2 Decision-faithful vs bit-faithful (ADR-0006)

A naive reading of "replaying the logged zone mix reproduces the output" cannot be bit-exact,
because the learned zone cores run in floating point on a GPU and float is not deterministic
across hardware. We therefore distinguish two levels, and we claim only what holds.

- **Decision-faithful.** Replaying the *recorded scores* through the canonical merge decision
  reproduces the *recorded survivor set and weights*. This is a pure integer computation on the
  recorded data; it always holds, and it is exactly what `verify_decision_faithful` checks. It
  is the product guarantee: an auditor confirms that "which zones fired, what was pruned, with
  what weights" was not fabricated — it is precisely what the deterministic rule produces from
  the recorded scores. Because the *same* function produces and verifies the decision
  (`prune_and_reweight`, §3.4), this check is meaningful rather than circular.
- **Bit-faithful.** Re-running the full forward pass reproduces `output_hash` bit-for-bit. This
  holds only under a fully deterministic inference path — the Q16.16 merge composes
  deterministically, but the float zone cores are deterministic only in a deterministic-inference
  mode. At the L0 (toy, deterministic) scale it holds and is tested; at L1 it is mode-dependent.

This distinction is not a hedge; it is the honest content of the claim, and it matters for the
patent language (the verifiable-output claim, C-2, is the decision-faithful one, with bit-faithful
offered as an optional mode). It also clarifies how NAT composes with cryptographic verification
(§4.4): the *structural* decision is verifiable for free, every pass; the *numeric* layer can be
wrapped in a proof when bit-exact logits are the thing being certified.

## 4.3 On-chain commitment and replay

On Citrate, the trace hash (or the trace itself) becomes part of the inference transaction. An
auditor pulls the committed weights and the trace, replays the recorded decision against the
weights, and confirms it reproduces the output. That replayability *is* the opacity solution:
we are not hiding the computation behind an explanation, we are recording it in a form a third
party can re-check. The merge runs on the Q16.16 path precisely so this replay is exact for the
decision layer regardless of which CPU the auditor uses.

## 4.4 Relation to Citrate's verifiable-inference substrate (Paper X)

NAT does not replace verifiable inference; it supplies the structural layer that zkML and TEE
attestation lack. Zero-knowledge ML [Kang et al. 2024; Sun et al. 2024; zkGPT 2025] proves,
after the fact and at heavy cost — hundreds of seconds per proof for a small transformer — that
*some* opaque computation ran on committed weights, verifying the *output* but saying nothing
about the *reasoning*. NAT's trace verifies the reasoning, by construction, with no per-inference
proof. The two compose cleanly because NAT's merge uses the **same Q16.16 fixed-point substrate**
as Citrate's verifiable-inference precompiles (Paper X §1): a NAT trace hash is directly
committable, and when an application needs bit-exact certification of the numeric layer it can
wrap that layer in a Halo2-KZG proof or a TEE attestation exactly as Paper X describes. NAT thus
turns "the chain has verifiable inference" from a claim about *outputs* into a claim about
*models whose internal computation is itself the verifiable artifact.*

## 4.5 Formal specifications

The stateful surfaces — where ordering and determinism guarantees live — are specified in TLA+
(`formal/`). Three modules:

- `MergeDeterminism.tla` — the federation-critical property that the merge composes the same
  gathered set to the same result, with pruning correct, complete, and tie-free (the ranking is
  a strict total order, so the survivor/pruned boundary never depends on iteration order).
- `AsyncGather.tla` — the gather terminates, respects the deadline, and records every straggler
  (`Consistent`, `StragglerRecorded`, `WindowCloses`).
- `McpHarness.tla` — `NoUngatedSideEffect`, `NoExecOnFailedCodec`, and termination (§3.5).

Each module's safety invariants and the matching Rust acceptance test enforce the same property
at two levels — the model checker over all reachable states, the test at runtime. We are honest
about status: the modules are written to be checked, but TLC was not run in the bootstrap
environment (no JRE), so "TLC green on all three" is an open Gate-1 item rather than a completed
one. The invariants are also claim-shaped for IP review (C-1 through C-5), with the defensible
combination being *auditable-by-construction, GGUF-compatible, zone-partitioned, with
on-chain-verifiable provenance and a formally-specified tool harness.*
