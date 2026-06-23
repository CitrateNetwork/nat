# §6 Results

We report four findings: the H-01 ablation (the bet), routing differentiation (H-02), the
scale ladder, and decision-faithfulness (H-03a). Each is stated with its caveat in the same
breath. All numbers are from runs on the GB10 (`candle-cuda`) and are reproducible from the
commands given; they are recorded in the case studies `CS-01-h01-the-bet.md` and
`CS-02-real-data-and-scaling.md` and the hypothesis ledger.

## 6.1 H-01: partitioning does not reduce capability per parameter vs an equal-parameter dense baseline

Under the ADR-0005 protocol (§5.2), the real `NatTrainModel` and an equal-parameter dense
transformer (20,718 ≈ 20,701 parameters) were each mini-batch-trained on the 1.12M-token
public-domain corpus as a next-byte language model, with capability measured as the inverse of
**held-out** cross-entropy per parameter, across five seeds sharing all training settings.

**H-01 is supported: partitioning does not reduce capability per parameter, and is modestly
lower-loss on the mean, across all five seeds.** We state this carefully, because H-01 is
registered as a *non-inferiority* hypothesis ("does not reduce"), and the per-seed verdict in code
is a non-inferiority test with a 5% slack (`nat_cpp ≥ dense_cpp · 0.95`) — so "5/5" means "on no
seed did partitioning fall more than 5% behind," not, by itself, "strictly better on every seed."
With that defined: the partitioned model's held-out next-byte loss is **2.88–2.91 versus the
dense baseline's 2.97–2.99** (these ranges are the per-seed extremes and do not overlap), with mean
capability-per-parameter 1.670 × 10⁻⁵ for NAT versus 1.621 × 10⁻⁵ for dense. The ranges are
non-overlapping and consistent with a real effect in NAT's favor, but we ran **five seeds with no
within-arm variance reported and no formal significance test**, so we call the result *suggestive
and supportive*, not a demonstration of superiority.

Two honest qualifications. First, the synthetic pre-read: on the binned-token-sum task (≈3,882
parameters, full batch) H-01 held on the mean (4.37 ≥ 3.88 cap/param) but on **3 of 5 seeds** — and
3/5 versus 5/5 at N=5 is not a statistically distinguishable difference, so we do not read it as
the architecture "sharpening." The honest summary is that both reads are consistent with H-01, the
synthetic and real runs differ in *task, batching, and metric* (classification/full-batch/accuracy
versus next-byte-LM/mini-batch/cross-entropy), and we treat the real-data run as primary because it
is the more realistic setting — not because 5/5 is decisive where 3/5 was not. Second, the **scale
caveat**: this is ~20K parameters, three zones, byte-level, ~1M tokens; it validates the bet enough
to keep building toward L2, not *at* L2, and a parameter-matched mixture-of-experts control and
component ablations (§8) are the experiments that would establish *which* structural feature carries
the effect. A larger or better-controlled run could overturn it, and if it does we will say so.

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

## 6.3 A size/zone ladder trends downward (suggestive, not a scaling law)

Holding the corpus fixed and growing the model along a three-rung ladder (single-output next-byte
LM), held-out bits/byte fall at each step:

| Rung | Params | Zones | Held-out bits/byte |
|------|--------|-------|--------------------|
| S | 20,718 | 3 | 4.097 |
| M | 56,534 | 3 | 4.054 |
| L | 114,956 | **5** | **3.953** |

Two cautions keep this from being a scaling claim. First, three points are a trend, not a law —
we cannot fit or extrapolate a scaling curve from S/M/L. Second, the ladder is **confounded**: the
L rung changes *two* variables at once, parameters (20.7K → 115K) **and** zone count (3 → 5), so we
cannot attribute the L improvement to size alone — the five-zone widening (the first real-data
training of the SM/CB state-space zones, ADR-0008) is plausibly part of it. So the ladder is
suggestive evidence that the architecture does not *degrade* with size at this range, not proof
that it scales. Separately, a **per-position autoregressive** objective (WP-D7) reached 3.42
bits/byte at 53K parameters; we note this only as a denser training objective that the path to L2
will use — it predicts at every position rather than once per window, sees far more supervision
signal, and so is *not* a controlled size or efficiency comparison against the single-output ladder
above, and we do not present it as one. None of this is an L2 result; it is a small, confounded
curve, which is evidence and not a guarantee.

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

The load-bearing hypothesis is supported (non-inferiority, with a modest mean advantage) at L1 on
real data across five seeds, with honest small-scale and no-significance-test caveats; routing
differentiation is learnable and **generalizes** at small scale (held-out 3.10 vs 2.63); the
size/zone ladder trends downward over three confounded rungs; and the verifiability guarantee that
motivates the whole design holds by construction. What we have *not* shown — a parameter-matched
mixture-of-experts baseline and component ablations to identify the operative cause, a task-level
capability metric, results on a standard corpus, the hold surviving orders-of-magnitude more scale,
and federated training in practice — is the subject of §8 and §9.
