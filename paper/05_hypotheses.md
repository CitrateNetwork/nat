# §5 Hypotheses and experimental design

NAT is a research bet, and we hold the bet to an explicit ledger: every capability or design
claim is a numbered hypothesis with a status and an evidence pointer (`hypotheses.md`), and a
hypothesis is "supported" only against a measurement, never against an assertion. This section
states the ledger, the protocol for the load-bearing test, and the corpus.

## 5.1 The hypothesis ledger

- **H-01 (load-bearing).** Zone partitioning does not reduce capability per parameter versus a
  dense baseline of equal size. *This is the bet.* If it fails, honest posture says change course.
- **H-02.** Context-aware routing produces measurably different zone mixes for different prompt
  classes.
- **H-03a.** Provenance is *decision-faithful*: replaying recorded scores reproduces the recorded
  survivors and weights. **H-03b.** Provenance is *bit-faithful* under a deterministic-inference
  path (§4.2).
- **H-04.** State-space temporal zones cut per-zone compute meaningfully versus attention at equal
  sequence length.
- **H-05a.** The merge composes the same gathered set to the same result (federation-critical
  determinism). **H-05b.** A federated *training* cycle reproduces the centralized result within
  tolerance — a distinct, statistical claim the merge-determinism proof does not cover.

H-01 is the one that decides whether the whole bet pays off, and the scale ladder exists to test
it cheaply before any expensive commitment. The rest of this section and §6 concern H-01, H-02,
H-03a, and the scaling evidence; H-04, H-03b, and H-05b are stated for completeness and left to
larger runs.

## 5.2 The H-01 protocol (ADR-0005)

An ablation at unequal parameters proves nothing, so the comparison is pinned by protocol and the
protocol is **enforced in code** — the harness *refuses* to run (and to report) a comparison
outside tolerance. The partitioned arm is the real trainable `NatTrainModel` (zones + the learned
router + the differentiable merge reconciled to the Q16.16 provenance merge). The control is an
equal-parameter dense single-block transformer (`nat-ablation::real::DenseTransformerArm`), whose
feed-forward width is searched until its parameter count matches the NAT arm within **±5%**; a run
outside that band errors rather than reports. A second guard, `guard_not_toy`, refuses a toy-backed
arm, so a measured result can never be a toy artifact (§3.1, §4). Both arms share the **same training
data, windows, epochs, batch size, learning rate, and shuffle seed** — only the structure
(partitioned vs dense) differs. Capability is proxied as the inverse of held-out
cross-entropy, measured per parameter, averaged over five seeds. We state the proxy plainly: `1/loss`
is an arbitrary monotone transform of cross-entropy, and because the arms are parameter-matched to
within 0.08% here (20,718 vs 20,701) the per-parameter normalization is nearly a no-op — so the
comparison is, in effect, *lower held-out loss at equal parameters*. The per-seed verdict is the
non-inferiority test "partitioned ≥ dense within a 5% slack," and the headline is the holds-fraction
across seeds. The hypothesis is *registered* as non-inferiority — a deliberately conservative bar — but
on real data the measured outcome is stronger: strictly lower held-out loss at every seed, and at
every rung of the scale ladder, with the margin widening as parameters grow (§6.1). The run is
reproducible: `scripts/dgx-gpu.sh run -p nat-ablation --features cuda --example real_h01_corpus -- <corpus-dir>`.

We ran H-01 twice, deliberately. First on a **synthetic** task (a binned-token-sum, full-batch,
~3.9K params) — easy to control, but, as it turned out, too smooth to separate the architectures.
Then on **real text** (the corpus below), which is the read we treat as decisive. Reporting both,
and reporting the synthetic read as marginal rather than rounding it up, is part of the method. The
same protocol is then instantiated at two scales: the single-output byte-level ablation above
(~20.7K params, `real_h01_corpus`) and a **per-position autoregressive** LM whose dense control is
parameter-matched at **each** rung of a subword-tokenized ladder (248K–2.0M params, BPE-4096, match
≤0.02% throughout; `h01_autoreg_bpe`), so the scale trend is a sequence of independently-matched
ablations rather than one match extrapolated.

## 5.3 The corpus

The L1 corpus is a 1.12M-token, license-clean, public-domain "values spine," built through the
`nat-data` pipeline (license-fail-closed, length and dedup gates, a PII screen, and a quality
score; raw is immutable, dropped data is quarantined with a reason). Its composition is
deliberately opinionated, and the thesis behind the curation doubles as a thesis of the paper:
*a rule has no meaning without a community and a form of life* (Wittgenstein) — which is also why
a maker follows the rules of the room they are in, why provenance must answer to a public
standard, and why good code reads like the codebase around it. The corpus draws on four pillars
on that one foundation: logic (Boole → Frege → Russell → **Belnap**), computation (Turing →
Church → Shannon), craft (SICP, the Rust Book, permissive code), and expression (Strunk, Whitman,
Montaigne, Carroll, Austen), with copyrighted ideas entering only through authored CC0 explainers
we own the framing of. The pipeline reports the corpus health: 2,337 documents, 779
shards, **1,120,711 tokens**, aggregate quality 0.852 (the pipeline's own self-score, on its own
rubric — a curation diagnostic, not an external quality measure), and zero documents quarantined by
its own gates (one PII false-positive excepted). For the scale ladder the corpus was grown the same
fail-closed way — adding the Rust Book and permissive crates, then SICP (CC-BY-SA, owner-approved) —
into a **1.91M-token, 5,064-document v3** at the same 0.852 self-score and zero quarantined. Two
tokenizations are used: **byte-level** (vocabulary 256, deterministic; metric bits/byte, uniform
baseline 8.0) for the small ablations, and a corpus-trained **byte-pair encoding** for the scale
ladder, where a parameter-matched vocabulary sweep locates a compression/efficiency knee near 4,096
(§6) — the vocabulary the ladder uses. Every result reports bits/byte, so byte- and subword-level
runs stay on one comparable axis.

This corpus is small by frontier standards and we make no apology for that — it is the substrate
for a *cheap, honest* test of the load-bearing hypothesis, not a bid for a leaderboard. The
question §6 answers is not "is NAT a good language model," it is "does the *structure* help or
hurt at equal cost," and that question is answerable, and answered, at this scale.
