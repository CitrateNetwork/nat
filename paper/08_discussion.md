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

- **Scale.** Every quantitative result is at small scale: ~20K parameters, three to five zones,
  byte-level tokenization, ~1M tokens. H-01's hold is unanimous across seeds but at a scale where a
  larger model, BPE tokenization, more depth, and orders of magnitude more data could change the
  verdict. The scale ladder is monotone and encouraging, but a curve over three rungs is not a law.
- **Synthetic versus real.** The synthetic H-01 read was marginal (3/5); the real-data read was
  decisive (5/5). We treat the real-data read as primary and report the synthetic one as marginal,
  but the gap between the two is itself a caution: the verdict is task-sensitive, and a different
  real task could read differently.
- **In-sample metrics.** H-02's routing differentiation is measured in-sample; it shows
  differentiation is learnable and measurable, not that it generalizes. The capability proxy is the
  inverse of held-out loss, not a task-level metric (accuracy, downstream eval), because the model
  is not yet large enough to have a meaningful one.
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
