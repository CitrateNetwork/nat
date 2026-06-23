# §6 Results

We report four findings: the H-01 ablation (the bet), routing differentiation (H-02), the
scale ladder, and decision-faithfulness (H-03a). Each is stated with its caveat in the same
breath. All numbers are from runs on the GB10 (`candle-cuda`) and are reproducible from the
commands given; they are recorded in the case studies `CS-01-h01-the-bet.md` and
`CS-02-real-data-and-scaling.md` and the hypothesis ledger.

## 6.1 H-01: partitioning beats an equal-parameter dense baseline on real text

Under the ADR-0005 protocol (§5.2), the real `NatTrainModel` and an equal-parameter dense
transformer (20,718 ≈ 20,701 parameters) were each mini-batch-trained on the 1.12M-token
public-domain corpus as a next-byte language model, with capability measured as the inverse of
**held-out** cross-entropy per parameter, across five seeds sharing all training settings.

**H-01 holds, 5 of 5 seeds.** The partitioned model reaches lower held-out next-byte loss than
the equal-parameter dense baseline on every seed: **NAT 2.88–2.91 versus dense 2.97–2.99**, with
mean capability-per-parameter 1.670 × 10⁻⁵ for NAT versus 1.621 × 10⁻⁵ for dense. Partitioning
does not cost capability per parameter here — it *adds* it.

We are deliberate about two things. First, the contrast with the synthetic pre-read: on the
binned-token-sum task (≈3,882 parameters, full batch) H-01 held *on the mean* (4.37 ≥ 3.88
cap/param) but only on **3 of 5 seeds** — a marginal hold we reported as marginal, which is
exactly why we went and got real data rather than declaring victory. Real data sharpened the
verdict from a coin-flip to unanimous; the most plausible reason is that the synthetic task is too
smooth to separate the architectures while real text's structure rewards the partition. Second,
the **scale caveat**, stated plainly: this is ~20K parameters, three zones, byte-level, ~1M
tokens. The result validates the bet enough to keep building toward L2; it does **not** validate
it *at* L2. A larger run could overturn it, and if it does we will say so.

## 6.2 H-02: a trained router differentiates by prompt class

On the evaluation battery (math · narrative · code · sensory), the trained `LearnedRouter` (WP-3)
produces a between-class versus within-class separation ratio of **11.70**, versus **4.25** for
the unlearned L0 routing baseline on the same `nat_eval::separation_ratio` metric — the router
learns to drive different zone mixes for different prompt classes, more sharply than the hand-wired
baseline. The caveat is real and we flag it: this is **in-sample** (the router is trained and
scored on the same battery, as is the baseline), so it demonstrates that routing differentiation
is *learnable* and *measurable*, not that it generalizes. Held-out batteries at scale are the
conclusive routing-differentiation test, and they are future work.

## 6.3 The scale ladder: loss falls monotonically with size

Holding the corpus fixed and growing the model along a size ladder (single-output next-byte LM),
held-out bits/byte fall monotonically:

| Rung | Params | Zones | Held-out bits/byte |
|------|--------|-------|--------------------|
| S | 20,718 | 3 | 4.097 |
| M | 56,534 | 3 | 4.054 |
| L | 114,956 | **5** | **3.953** |

Two things are worth noting. The improvement is monotone in size — at least consistent with the
structure helping rather than hurting as it grows. And the best rung is the **five-zone** L
configuration, which is the first real-data training of the SM/CB state-space zones (ADR-0008
staged them until "the data earns it," and at this corpus size it does). Separately, moving from a
single-output model to a **per-position autoregressive** form (WP-D7) reached **3.42 bits/byte at
53K parameters** — better loss with roughly half the parameters — which is the efficiency unlock
on the path toward an L2-scale run. None of this is an L2 proof; it is a curve, and a curve is
evidence, not a guarantee.

## 6.4 H-03a: decision-faithful replay holds by construction

The decision-faithfulness property (§4.2) holds by construction and is checked: for every sampled
pass, replaying the recorded scores through the canonical merge decision reproduces the recorded
survivors and weights (`verify_decision_faithful`), because the same function produces and verifies
the decision. Decision-faithfulness is therefore not an empirical contingency at this scale but a
structural guarantee; bit-faithfulness (H-03b) holds at the deterministic L0 scale and is
mode-dependent at L1, as §4.2 sets out.

## 6.5 Summary

The load-bearing hypothesis is supported at L1 on real data, unanimously across seeds, with an
honest small-scale caveat; routing differentiation is learnable and measured, with an in-sample
caveat; the architecture scales monotonically over the ladder we ran, with the five-zone rung
best; and the verifiability guarantee that motivates the whole design holds by construction. What
we have *not* shown — generalization of routing, a task-level capability metric, the hold surviving
orders-of-magnitude more scale, and federated training in practice — is the subject of §8 and §9.
