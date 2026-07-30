# ADR-0011 — Three seam changes so the co-op can run NAT

**Status:** accepted · **Date:** 2026-07-30 · **Decides:** the
`citrate-compute-pool` training backend (Gate 3 → Gate 4 path)

## Decision

Three changes to NAT's public surface, all found by building the real backend in
`citrate-compute-pool` rather than by reading the code:

1. **`Signer::sign` becomes fallible** — `-> Result<Vec<u8>, SignError>`, and
   `SignedContribution::create` returns `Result` with it.
2. **`CausalCore: Send + Sync`**, so `AutoregLm` is `Send`.
3. **`named_parameters()`** on `AutoregLm` and `AutoregDenseLm` — the parameters
   as `(name, values)`, name-ordered.

## Rejected

- **Leaving `Signer::sign` infallible and adapting around it.** The adapter's only
  options on a KMS failure are to panic or to return garbage bytes. Rejected
  below on the grounds that garbage is actively dangerous, not merely untidy.
- **`unsafe impl Send for AutoregLm`** in the consumer. That asserts a property of
  a private trait object in another crate — an assertion that could silently
  become false on any NAT change, with no compiler check. Rejected: the bound is
  free and honest.
- **Exposing `varmap` directly.** Leaks a Candle type into the public API and ties
  every consumer to the framework choice ADR-0010 deliberately isolated.
  `named_parameters` returns plain data instead.
- **Leaving the delta to a safetensors round-trip.** It works — the co-op backend
  shipped that way first — but it is a disk write, a parse and a copy on the hot
  path of every training step.

## Why

### 1. Fallible signing — the one that is a correctness bug, not ergonomics

`Signer::sign` returned `Vec<u8>`, sync and infallible, and
`SignedContribution::create` returned `Self`. The production signer is the
gateway operator signer whose AWS-KMS adapter is a **network call**.

So an adapter facing a KMS timeout could only panic — killing the training loop —
or return garbage bytes. Garbage is the worse one, and this is what decided it:
`gather_and_aggregate` classifies an unverifiable contribution as
`RejectReason::BadSignature`, whose own documentation reads *"forged, tampered, or
unknown node"*, and the contribution "contributes nothing to
`total_reward_weight`". **There is no transient-failure variant in that enum.**

A network blip would therefore have been recorded as the node forging signatures,
and cost it the round's pay. That is a mechanism converting an AWS outage into a
fraud verdict against an honest operator. A `Result` costs one line at each call
site and removes it.

Once fallible, the real adapter is also *better* than the stand-in it replaces:
the gateway signer is **recoverable secp256k1**, so `node_id` can be the operator
address and the verifier recovers it from the signature — no roster to
distribute, which `ToyRosterVerifier` needs and which is one more thing to keep
in sync across a federation.

### 2. `Send + Sync` on the core trait

`AutoregLm` holds `Vec<Box<dyn CausalCore>>`. Every field it actually owns —
candle `Tensor`, `Var`, `Linear` — is already `Send`, but the trait object
carried no bound, so **the model type is not `Send`**. `ModelBackend` in
compute-pool requires `Send + Sync`, so the co-op worker had to confine the model
to a dedicated thread and drive it over a command channel purely to compile.

The bound costs nothing, is true of every implementation we have, and deletes
that thread.

### 3. `named_parameters()`

`nat-federated`'s contribution unit is a per-zone weight delta
(`ZoneWeightDelta`). The only way to obtain one was `save` → read the file →
diff, once per step.

The names matter and are now explicitly part of the contract: `zone_HP.wq`,
`zone_SM.log_a`, `score_PF` carry zone identity, which is what lets a consumer
attribute a delta to the zone that produced it. That was already true and
undocumented — a consumer was depending on it whether we said so or not.
**Renaming a parameter is a breaking change for the federated path**, and saying
so is the point of writing it down.

Returned name-ordered because the federated aggregate is coordinate-wise: two
callers enumerating the same model must produce the same vector, or the aggregate
silently mixes unrelated coordinates.

## Evidence

`cargo test --workspace` in `nat`: **215 passing**, clippy clean. All
`SignedContribution::create` call sites were tests using the local
`ToyKeyedSigner`, which cannot fail — they carry an explicit
`.expect("local toy signer cannot fail")` rather than a silent `unwrap`, so the
claim is visible.

The consuming side is `citrate-compute-pool` PR #9, whose `nat_backend.rs`
documents each of these as a cost it was paying before the change.

## Consequences

- `Signer` implementations outside this repo must return `Result`. The only one
  in-tree is `ToyKeyedSigner`.
- `SignedContribution::create` callers must handle a `Result`. A caller that
  cannot sign should **retry or sit the round out** — never submit, because an
  unsigned or garbage-signed contribution is indistinguishable from forgery at
  the gather.
- The co-op backend can drop its model thread and its per-step safetensors
  round-trip. Both are tracked in that repo, not here.
