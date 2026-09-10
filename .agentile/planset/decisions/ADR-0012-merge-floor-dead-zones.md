# ADR-0012 — A merge floor, because a pure softmax can kill a zone

**Status:** accepted · **Date:** 2026-07-30 · **Decides:** the per-position merge
(WP-2 / `autoreg`), and the validity of the H-01 numbers measured before it

## Decision

The per-position zone merge mixes a uniform floor into the softmax:

```
w = (1 - merge_floor) * softmax(score / tau) + merge_floor / nz
```

`AutoregConfig::merge_floor` defaults to `DEFAULT_MERGE_FLOOR = 0.01` — 1% total,
so with five zones each is guaranteed 0.2%. `0.0` reproduces the original pure
softmax exactly.

Also adds `AutoregLm::zone_merge_weights()`, which reports each zone's actual
share, so this class of failure is observable instead of silent.

## The evidence, measured before the fix

On the real 64M checkpoint (`checkpoints-64m/nat-seed2`, BF16, d=1183) over real
corpus-v6 documents, on GPU:

| zone | merge share |
|------|-------------|
| SM | 0.060620 |
| CB | 0.299012 |
| HP | 0.448056 |
| **PF** | **0.000000** |
| CX | 0.192311 |

**`PF` — Prefrontal, the deepest reasoning zone, "reasoning, planning,
language" — had exactly zero share.** Its `score_PF.weight` norm is 2.216 against
0.10–0.21 for every other zone: the scar of the optimizer driving its score down
until the softmax squeezed it out entirely.

With the floor at 0.01, PF's share becomes 0.002000 and the other four move by
less than 0.003 — routing is preserved, the zone is revived.

## Why this is a dead zone, not a preference

A zone's gradient scales with its merge share, because its output enters the
composition multiplied by that weight. Share zero ⇒ gradient zero ⇒ its score
head never updates ⇒ share stays zero. **Self-reinforcing.** Once a zone falls
out, nothing in the architecture can bring it back.

And it is invisible without looking for it: the loss still falls, the other zones
still train, and the checkpoint still loads. Nothing reports that a fifth of the
model is inert.

## Rejected

- **Raising `tau`.** Softens the softmax, but only delays collapse — the dynamics
  that drove PF out still run, just more slowly. It also blunts routing
  everywhere to fix a failure at one extreme.
- **An auxiliary load-balancing loss** (the standard MoE remedy). Effective, but
  it adds a second objective whose weight becomes another thing to tune, and it
  makes the merge's behaviour depend on optimizer state rather than on the merge.
  ADR-0001 rejected MoE routing for interpretability; importing MoE's patch for a
  problem we can fix structurally would be borrowing the wrong half.
- **Re-initialising a collapsed score head.** Reactive: it needs a detector, a
  threshold, and a policy, and it discards whatever that head had learned.
- **Leaving it and documenting it.** The failure is silent and permanent. A note
  in a doc does not stop the next run losing a zone.

## Why a floor

- **Structural, not corrective.** A zone cannot reach zero share, so the
  self-reinforcing loop cannot start. Nothing to detect and nothing to tune.
- **Deterministic.** An affine transform of the softmax, so the merge stays
  bit-reproducible — which ADR-0006 and `MergeDeterminism.tla` require, and which
  a stochastic fix (noise, dropout on the router) would have broken.
- **Cheap and bounded.** At 1%, a zone the router genuinely favours keeps ~99% of
  its share; the test `the_floor_preserves_learned_routing` pins that.
- **Opt-out.** `merge_floor = 0.0` reproduces prior behaviour bit-for-bit, so an
  earlier run can still be replayed exactly.

## Consequence for H-01 — the load-bearing one

**The zone arm has been running with one of its five zones inert.**

Every H-01 number to date — the 4M/8M holds, the 0.176 gap, the 0.188 → 0.251
widening — was measured on a NAT arm where PF contributed nothing to the
composition while still being counted in the parameter budget. The comparison was
"dense at N params vs zones at N params, one zone dead."

That cuts in the hypothesis's favour: the zone arm won those comparisons
handicapped. But it means **the numbers are not the numbers.** H-01 needs
re-running with the floor before any gap figure is quoted as current, and the
prior figures should be cited as measured under a dead PF.

I have not re-run the ladder. Stating that plainly rather than quietly leaving
the old numbers to be read as still valid.

## Evidence

`cargo test --workspace`: 219 passing, clippy clean. Four tests pin the floor:
every zone at or above its guaranteed share, routing preserved within 0.02,
shares still summing to 1, and `floor = 0.0` reproducing the pure softmax.

Measured with `cargo run -p nat-candle --features cuda --example zone_shares`
against the real checkpoint and real corpus.
