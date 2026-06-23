# §8 Discussion

## 8.1 Why one move yields three properties

The paper's central argument is that *declaring structure* is not a trade against capability but a
move that pays out along three axes at once, and §6 is the evidence that it does not cost what one
might fear. The three properties share a single root: once the partition is declared and the merge
is deterministic, the same fact — *which named function contributed, with what weight* — is
simultaneously (i) a verifiable record (verifiability), (ii) an inductive bias that, empirically,
helps at equal parameters (capability), and (iii) a unit of contribution a federation can own,
sign, and reconcile (decentralizability). The provenance trace is the connective tissue: it is the
verifiable artifact, it is the audit of the pruning that realizes the efficiency, and it is the
signed contribution that the network rewards. We did not design three mechanisms; we declared one
structure and read three properties off it.

## 8.2 Threats to validity

We hold the results to their limits.

- **Scale.** Every quantitative result is at small scale: ~20K–115K parameters, three to five
  zones, byte-level tokenization, ~1M tokens. H-01's non-inferiority hold is across five seeds but
  at a scale where a larger model, BPE tokenization, more depth, and orders of magnitude more data
  could change the verdict. The size/zone ladder trends downward but a confounded three-rung curve
  is not a scaling law.
- **Missing baselines and ablations.** This is the most important gap and we name it plainly. H-01
  compares NAT against exactly one control — an equal-parameter dense single-block transformer.
  There is **no parameter-matched mixture-of-experts baseline**, the most obvious comparison for a
  paper positioned against MoE, and **no component ablation** isolating which feature carries the
  effect (router vs the pruning merge vs the partition itself vs SSM/attention heterogeneity). The
  ~0.08–0.10 loss gap could in principle be a regularization effect of the pruning merge or of
  using several narrower sub-blocks rather than evidence that *declaring named structure* is the
  operative cause; our current experiments cannot rule that out. We also report only on a bespoke
  corpus, not a standard one (next).
- **Non-standard corpus.** Results are on the 1.12M-token "values spine," a hand-curated corpus no
  one else uses, which makes the numbers non-comparable to external results. A standard small-LM
  benchmark (enwik8/text8/WikiText) is needed for comparability.
- **Statistics.** Five seeds, no within-arm variance reported, no formal significance test; the
  synthetic (3/5) and real (5/5) reads are not statistically distinguishable at N=5, so we do not
  read the difference between them as the architecture "sharpening."
- **Metric.** The capability proxy is inverse held-out loss, not a task-level metric (accuracy,
  downstream eval), because the model is not yet large enough to have a meaningful one. H-02
  generalizes on a held-out split at this scale (3.10 vs 2.63), but a full labeled battery at scale
  is the conclusive routing read.
- **The brain analogy is not yet earned.** We have not shown that the neuro-motivated zone
  assignment outperforms a *random* equal-width partition of the same hidden state. Until that
  ablation is run, the neuroscience framing is motivation, not evidence, and we treat it as such.
- **Specified versus demonstrated.** The federated and Belnap-consensus layer (§7) is specified, not
  demonstrated; bit-faithful provenance (H-03b) is mode-dependent; the TLA+ modules are written but
  not yet TLC-checked in our environment. We label each of these in place rather than rounding up.
- **Single-author seeds and self-reported quality.** The runs are from one operator on one machine;
  independent replication is the standard remedy, and the reproducibility floor (config hashes,
  fixed seeds, exact rerun commands, a containerized CI path) exists precisely to make it cheap.

## 8.3 Honest posture as method

The discipline that produced this paper is itself a claim we want to make explicit, because it is
unusual and we think it is load-bearing. Every capability claim points at a measurement or is labeled
a hypothesis; the brain analogy is a heuristic the architecture is licensed to abandon; the
load-bearing bet is stated with its caveat in the same breath as its result; and the marginal
synthetic H-01 read was reported as marginal — which is exactly why the work went and got real data
rather than declaring victory. The hypothesis ledger, the decision records, and the case studies are
part of the artifact, not commentary on it.

## 8.4 An agent-built model

This work was built largely by AI agents executing against human-set hypotheses and gates, with the
process governed and logged under an explicit methodology (proposed, reviewed, gated changes; every
claim anchored to evidence; every stage closing with a case study). The architecture was designed in
a human–AI design conversation; the reference implementation, the data pipeline, the training stack,
and the H-01 ablation were built by agents under the honest-posture discipline above; and the bet's
resolution — from a marginal synthetic hold to a decisive real-data one — came from an agent, given a
network-connected budget, going and getting real data rather than overclaiming the synthetic read. We
document this in a companion case study (*Agents Doing Science*) not as a novelty but as evidence: a
verifiable architecture and a logged, reproducible method are mutually reinforcing — the same
discipline that makes the model's provenance trustworthy makes the research record trustworthy. The
model and the method answer to the same standard.
