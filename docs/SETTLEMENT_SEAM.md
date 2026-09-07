# Settlement seam — NAT → citrate-compute-pool

**Decision:** ADR-0007. NAT does not reinvent reward settlement. It emits the
inputs; `citrate-compute-pool` settles them into participant rewards.

## The economic claim, stated plainly

A participant's economic advantage is a function of **compute contributed × data
quantity/quality submitted**. NAT *scores* that contribution; compute-pool
*settles* it. This split keeps one audited reward implementation (compute-pool is
Tier-1 and already ships a compute marketplace, tokenomics simulation, and reward
settlement) instead of two divergent ones.

## What NAT emits

Per training step (or per gathered federated round), NAT produces a contribution
record. The accounting type is `nat_train::StepContribution`:

```rust
pub struct StepContribution {
    pub compute_metered: Q16,   // normalized FLOP-seconds (or equivalent), Q16.16
    pub data_quality:    Q16,   // [0,1] from the pipeline QUALITY_SCORE stage
    pub tokens:          u64,    // tokens consumed this step
    pub provenance_hash: String, // hash of the step's forward-pass trace(s)
}
```

NAT also computes the proposed **reward weight**, the *signal* compute-pool
converts to payout:

```
reward_weight = compute_metered × data_quality        // Q16.16, deterministic
```

A node that contributes compute on garbage data earns weight zero (quality = 0 ⇒
weight = 0). This is the property `StepContribution::reward_weight` enforces and
`nat-train`'s tests pin.

## The boundary (who owns what)

| Concern | Owner |
|---------|-------|
| Metering compute (FLOP-seconds, normalization) | NAT (`nat-train` at L1) |
| Scoring data quality (the QUALITY_SCORE pipeline stage) | NAT (`PLANSET/04_DATA_OPS.md` §4) |
| The provenance hash (what actually ran) | NAT (`nat-provenance`) |
| The reward-weight *formula* (compute × quality) | NAT (this seam) |
| Converting weight → payout (tokenomics, emission, vesting) | **compute-pool** |
| Settlement, dispute, on-chain transfer | **compute-pool** |

NAT decides *the weight*; compute-pool decides *the money*. The seam is the
`StepContribution` record plus its `reward_weight`.

## Why a deterministic weight matters

`reward_weight` runs on the Q16.16 path (`nat_types::Q16`), so two nodes — and an
on-chain verifier — compute the same weight from the same contribution, bit for
bit. A float reward formula would let nodes disagree on payouts by rounding,
which is unacceptable for a settlement input. This is the same determinism
discipline the merge uses (ADR-0006 / `MergeDeterminism.tla`).

## Integration status

- **Now (L0):** the `StepContribution` type and `reward_weight` formula are
  fixed and tested, so compute-pool can design its settlement adapter against a
  stable type before NAT trains anything.
- **L1:** `nat-train` populates `compute_metered` and `data_quality` for real.
- **Gate 4 (federated):** NAT emits `StepContribution` into compute-pool's
  settlement path; the end-to-end "compute × quality → reward" is the
  `features/gate4_federated.feature` "Reward weight follows compute and data
  quality" scenario.

## Open items for compute-pool

1. The adapter that ingests `StepContribution` and maps `reward_weight` →
   emission under its tokenomics.
2. Whether `compute_metered` normalization is owned here or in compute-pool's
   metering (proposal: NAT meters raw, compute-pool normalizes against its market
   — to be settled with the compute-pool operator).
3. Sybil / honest-metering defenses (a node could over-report compute) — this is
   compute-pool's existing peer-scoring + verification surface, not a new NAT
   mechanism.

## Security contract — self-reported metering (CCP-SEAM-1)

This is an **owner-accepted mitigation**, formalized here so the boundary cannot
drift silently. It is the seam an integration must honor.

**Invariant.** `gather_and_aggregate` authenticates a contribution (its signature
verifies against a roster key) and preserves post-signing tamper-integrity. It
does **not** — and by this decision **must not be relied upon to** — verify that
`compute_metered` reflects real FLOPs, that `data_quality` reflects a real corpus,
or that `provenance_hash`/`trace_hash` bind to a real forward pass. A node holding
a valid roster key can therefore sign a `StepContribution` with `compute_metered`
set arbitrarily high and earn `reward_weight = compute × quality` for work it did
not do. On the canonical settlement path `data_quality` is clamped to `[0,1]`, so
only `compute_metered` is unbounded above.

**Ownership.** The mitigation is **compute-pool's** metering-verification /
peer-scoring surface (open item #3 above). `reward_weight` is an *unauthenticated
proposal* — the signal NAT emits, never the payout. compute-pool MUST independently
meter or attest `compute_metered` before it settles money; it MUST NOT treat NAT's
signature as evidence that the metered value is honest.

**Tripwire.** `nat-federated`'s test
`over_reported_compute_is_accepted_ccp_seam_1` pins the current behavior: a
correct-key node reporting an inflated `compute_metered` is accepted and its full
weight enters `total_reward_weight`. It is the complement of
`forged_signature_is_rejected_before_aggregation` (which covers the *wrong-key*
case). If NAT ever grows a per-round metered ceiling or a compute-pool attestation
gate, that test must be inverted to assert the inflated value is capped/rejected —
and this contract updated to say the gate moved into NAT.
