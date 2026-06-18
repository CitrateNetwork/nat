# Agentile planset

The Agentile method applied to NAT (Research Method, `PLANSET/05_RESEARCH_METHOD.md`):
every change is proposed, reviewed, and gated; every claim is anchored to
evidence; every stage emits a case study.

```
gates.yaml            gate definitions + machine-readable exit criteria (CI reads this)
hypotheses.md         the live hypothesis ledger (H-01 is the load-bearing bet)
decisions/            ADRs — one per architectural decision (chain-of-title for counsel)
case-studies/         one per closed stage (CS-00 = the L0 forward pass)
experiments/          one dir per run: config, seed, metrics, notes (populated from L1)
```

## The two files everyone reads first

`gates.yaml` (what must be true to advance) and `hypotheses.md` (what is still a
bet). `gates.yaml` is machine-readable so CI can block a merge to a higher rung
before that gate's acceptance is green.

## ADR index

| ADR | Decision |
|-----|----------|
| 0001 | Hybrid routing (fixed topology, learned modulation) |
| 0002 | SSM cores in temporal zones |
| 0003 | Provenance as a forward-pass output |
| 0004 | Auxiliary sidecar format (+ flattened-dense export note) |
| 0005 | The H-01 dense-baseline protocol (Gate-3 blocker) |
| 0006 | Decision-faithful vs bit-faithful provenance |
| 0007 | Integrate with compute-pool for settlement |
| 0008 | Stage zones: six for the pass, three for the first capability test |
| 0009 | L0 numerics behind the ZoneCore trait |

ADRs 0001–0004 are the seed decisions from the design conversation; 0005–0009 are
the Gate-1 review remediations (`PLANSET/08_CRITIQUE_AND_REMEDIATIONS.md`).
