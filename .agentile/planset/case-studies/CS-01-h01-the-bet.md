# CS-01: H-01 — the bet, from synthetic-marginal to real-decisive

**Gate:** Gate 3 · **Rung:** L1 · **Dates:** 2026-06-21 .. 2026-06-22
**Authors:** Larry Klosowski (architect), Claude Code (build)

## Question
H-01, the load-bearing bet: does zone partitioning reduce capability per parameter
versus a dense baseline of equal size? Untestable at L0 (CS-00); this is its first
real test, under the ADR-0005 protocol.

## Setup
The real trainable `NatTrainModel` (zones + learned router + the WP-2 reconciled
differentiable merge) as the partitioned arm, versus an equal-param dense
single-block transformer (`nat-ablation::real::DenseTransformerArm`) as the control.
The dense arm is sized to the NAT arm's parameter budget by searching its FFN width;
the harness *refuses* a run outside ±5% (ADR-0005), and `guard_not_toy` refuses a
toy-backed arm. Two reads:
1. **Synthetic** (WP-5): the binned-token-sum task, full-batch, 5 seeds. Params
   3882≈3882.
2. **Real corpus** (WP-D6): both arms mini-batch-trained on the 1.12M-token
   public-domain corpus (next-byte LM), capability = 1/(held-out cross-entropy) per
   parameter, 5 seeds, on the GB10 (`candle-cuda`). Params 20718≈20701. Both arms
   share the same windows, epochs, batch size, lr, and shuffle seed.
Rerun: `scripts/dgx-gpu.sh run -p nat-ablation --features cuda --example
real_h01_corpus -- <corpus-dir>`.

## What we measured
Seed-averaged capability-per-parameter for each arm, the per-seed verdict
(partitioned ≥ dense within 5% slack), and the holds-fraction.

## Result
- **Synthetic (WP-5):** HOLDS on the mean (nat 4.37 ≥ dense 3.88 cap/param) but only
  **3/5 seeds** — a *marginal* hold. Reported as not decisive.
- **Real corpus (WP-D6):** **HOLDS, 5/5 seeds.** NAT held-out next-byte loss 2.88–2.91
  vs dense 2.97–2.99 — the partitioned model reaches lower held-out loss than the
  equal-param dense baseline on real text, every seed. Mean cap/param nat 1.670e-5 ≥
  dense 1.621e-5.

## What surprised us
That real data *sharpened* the verdict rather than muddying it. The synthetic read
was a coin-flip-ish 3/5; the real-data read was unanimous. The most likely reason: the
synthetic binned-sum task is too easy and too smooth to separate the architectures,
where real text's structure rewards the partitioning. The methodological point stands
on its own — the marginal synthetic hold was honestly reported as marginal, which is
exactly why we went and got real data instead of declaring victory.

## Decision
Record H-01 as **supported at L1 on real data (5/5 seeds)** in `hypotheses.md`, mark
`gates.yaml` g3-h01 **met** — with the explicit caveat that this is small scale
(~20K params, byte-LM, 3-zone, ~1M tokens). The bet is validated enough to keep
building toward L2; it is not yet validated *at* L2.

## Open threads
- Scale: does the hold survive BPE, depth, and orders of magnitude more params/tokens?
  The scale ladder (CS-02) is encouraging, not conclusive at scale. **Update (2026-06-24):**
  pushed to 4M and 8M on corpus-v4 (31M tokens), both **HOLD 5/5** under a stabilized
  (warmup+clip) recipe — the one 8M seed that first diverged came back in line (1.989 b/byte,
  was 3.314). Clean within-corpus gap widens **0.188 → 0.251**. See CS-02 "Continuation".
- H-02 held-out is now done (`nat-eval::h02_heldout`, PR #29): the trained router beats
  the L0 baseline on prompt classes it never saw (3.10 vs 2.63), so it generalizes, not
  memorizes. Open thread: this is held-out on the prompt-class battery; full-scale
  labeled batteries remain the L2 read.
- The capability proxy is 1/(held-out loss); a task-level metric (accuracy, downstream
  eval) is the stronger read once the model is big enough to have one.
