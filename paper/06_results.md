# §6 Results

We report four findings: the H-01 ablation (the bet), routing differentiation (H-02), the
scale ladder, and decision-faithfulness (H-03a). Each is stated with its caveat in the same
breath. All numbers are from runs on the GB10 (`candle-cuda`) and are reproducible from the
commands given; they are recorded in the case studies `CS-01-h01-the-bet.md` and
`CS-02-real-data-and-scaling.md` and the hypothesis ledger.

## 6.1 H-01: partitioning lowers held-out loss per parameter, and the margin widens with scale

Under the ADR-0005 protocol (§5.2), H-01 was read at three increasingly demanding settings; it holds
at all three, and the central new finding is that as scale grows the partitioned model's
per-parameter advantage **widens** rather than erodes.

**Byte level, 1.12M tokens.** The real `NatTrainModel` and an equal-parameter dense transformer
(20,718 ≈ 20,701 parameters, matched to 0.08%) were each mini-batch-trained on the public-domain
prose spine as a next-byte LM across five seeds sharing all training settings. The partitioned
model's held-out next-byte loss is **2.88–2.91 versus the dense baseline's 2.97–2.99** (per-seed
extremes, non-overlapping), mean capability-per-parameter 1.670 × 10⁻⁵ vs 1.621 × 10⁻⁵. H-01 is
registered as a *non-inferiority* hypothesis with a 5% slack (`nat_cpp ≥ dense_cpp · 0.95`), so "5/5"
formally means "on no seed did partitioning fall more than 5% behind" — but here the stronger fact
holds: partitioning is strictly lower-loss on every seed.

**A harder corpus does not close the gap.** Regrowing the corpus to the 1.91M-token v3 (prose + Rust
Book + SICP) raises both arms' absolute loss — code and Scheme are higher-entropy than prose — but
the margin survives: NAT **3.058–3.074** vs dense **3.138–3.148**, 5/5 seeds, capability-per-parameter
1.575 × 10⁻⁵ vs 1.537 × 10⁻⁵. Partitioning is not winning because the text is easy.

**At scale, the gap widens.** The decisive read moves to the real per-position autoregressive
architecture (five zones, the differentiable merge reconciled to the Q16.16 provenance merge) against
a dense control parameter-matched **at each rung**, on the v3 corpus under BPE-4096, verified to run
genuinely on GPU rather than a silent CPU fallback. Across an ~8× parameter range the partitioned
model is strictly lower-loss at every rung, 5/5 seeds, and the margin **grows**:

| Parameters | NAT bits/byte | Dense bits/byte | Gap | Holds |
|------------|---------------|-----------------|-----|-------|
| 248,235 | 2.086 | 2.110 | 0.024 | 5/5 |
| 1,005,603 | 1.890 | 1.996 | 0.106 | 5/5 |
| 1,992,978 | 1.845 | 1.986 | **0.141** | 5/5 |

**This is the result the bet turned on.** The standing worry was that partitioning is a small-model
trick a larger dense transformer would erase; across this range it does the opposite — the
per-parameter edge widens from 0.024 to 0.141 bits/byte. At equal parameters, data, seed, and
training budget, reaching lower loss means the same capability is attainable with fewer parameters
and less compute: read as efficiency, the partitioned model **learns more per parameter and per unit
of compute** than its dense twin, and increasingly so with scale. We measure capability per parameter,
not wall-clock training time, and state the claim at that grain.

**Caveats, in the same breath.** Every rung is ≤2M parameters on ~788K BPE tokens, and the 2M point
sits near this corpus's honest data ceiling (held-out loss is still falling, so it is not
overfit-bound, but the next real lever is corpus *volume*, not more parameters); the planned next step
grows data before parameters. Three rungs are a direction, not a scaling law, and a run orders of
magnitude larger could flatten or reverse the trend — if it does, that is the result and we follow it.
There is still **no parameter-matched mixture-of-experts baseline** and **no component ablation** (§8)
isolating which feature — router, pruning merge, partition, or SSM/attention heterogeneity — carries
the effect; a consistent, widening, non-overlapping margin across three independently-matched rungs is
strong evidence the effect is real, not yet proof of its cause. A synthetic pre-read (binned-token-sum,
≈3,882 params) held on the mean but on 3 of 5 seeds; we treat the real-data ladder as primary.

## 6.2 H-02: a trained router differentiates by prompt class

The trained `LearnedRouter` (WP-3) drives measurably different zone mixes for different prompt
classes, and — importantly — it **generalizes rather than memorizes**. On a held-out split of the
evaluation battery (math · narrative · code · sensory), the trained router still separates prompt
classes it never saw during training more sharply than the unlearned baseline: **3.10 versus 2.63**
on the `nat_eval` held-out separation metric (`h02_heldout`). On the in-sample battery the gap is
larger — separation **11.70 versus 4.25** for the unlearned L0 baseline on the same
`separation_ratio` metric — which is the upper, optimistic read; the held-out 3.10/2.63 is the one
we treat as the honest evidence of differentiation, since it measures prompts the router did not
train on. The caveats that remain: this is at L1 small scale, and full-scale labeled batteries
across more prompt classes are the conclusive read (future work). What is no longer open is whether
routing differentiation *generalizes* at this scale — it does, modestly, on held-out prompts.

## 6.3 An earlier single-output ladder (uncontrolled, superseded)

Before the parameter-matched NAT-vs-dense ladder of §6.1, an earlier **uncontrolled** read varied
NAT's own size and zone count on the fixed corpus (single-output next-byte LM); held-out bits/byte
fall at each step:

| Rung | Params | Zones | Held-out bits/byte |
|------|--------|-------|--------------------|
| S | 20,718 | 3 | 4.097 |
| M | 56,534 | 3 | 4.054 |
| L | 114,956 | **5** | **3.953** |

Two cautions keep this from carrying any capability argument. First, three points are a trend, not a
law — we cannot fit or extrapolate a scaling curve from S/M/L. Second, the ladder is **confounded**:
the L rung changes *two* variables at once, parameters (20.7K → 115K) **and** zone count (3 → 5), so
the L improvement cannot be attributed to size alone — the five-zone widening (the first real-data
training of the SM/CB state-space zones, ADR-0008) is plausibly part of it. So it is suggestive
evidence the architecture does not *degrade* with size at this range — no more. The load-bearing
capability claim rests on the **controlled** NAT-vs-dense ladder of §6.1, not on this one. The
**per-position autoregressive** objective first noted here (WP-D7; 3.42 bits/byte at 53K byte-level
parameters) is the same objective the controlled ladder later adopts at BPE scale — it predicts at
every position rather than once per window, and supplies the denser supervision the scale ladder
needs.

## 6.4 H-03a: decision-faithful replay holds by construction

Decision-faithfulness (§4.2) is a **design property, not an empirical result**, and we flag it as
such: replaying the recorded scores through the canonical merge decision reproduces the recorded
survivors and weights because the *same* function (`prune_and_reweight`) produces and verifies the
decision (a non-circular sharing we verify in the code, §4.2), so the check confirms the
implementation matches the specification and cannot, by construction, disconfirm the property. We
report it here for completeness — the structural guarantee is the point of the architecture — but
it is not evidence in the sense the H-01 ablation is. Bit-faithfulness (H-03b) *is* empirical and
mode-dependent: it holds at the deterministic L0 scale and only under a deterministic-inference mode
at L1, as §4.2 sets out.

## 6.5 Summary

The load-bearing hypothesis is supported and strengthening: partitioning is strictly lower-loss per
parameter than an equal-parameter dense control on real data across five seeds, the margin survives a
harder prose+code+textbook corpus, and on a parameter-matched, subword-tokenized ladder it **widens**
with scale (0.024 → 0.141 bits/byte across an ~8× parameter range, 5/5 seeds per rung), every rung
≤2M parameters with honest data-ceiling and no-significance-test caveats. Routing differentiation is
learnable and **generalizes** at small scale (held-out 3.10 vs 2.63); the verifiability guarantee that
motivates the whole design holds by construction; and the federated gather is now scaffolded with
signed, verify-before-compose semantics. What we have *not* shown — a parameter-matched
mixture-of-experts baseline and component ablations to identify the operative cause, a task-level
capability metric, results on a standard corpus, the hold surviving orders-of-magnitude more scale and
data, and an end-to-end multi-node federated cycle — is the subject of §8 and §9.
