# Research & Documentation Strategy (Agentile)

**Document:** RFC-CIT-NAT-0001 / Research Method
**Status:** Draft v0.1
**Companion to:** `00_MASTER_PLAN.md`

---

## 1. Why this document exists

A research bet earns the right to scale only if every stage leaves behind a
record someone else can audit, reproduce, and challenge. This is the same
discipline that makes the architecture's provenance log valuable... we hold the
build process to the standard we hold the model to. Each stage produces a case
study. Each case study is a small, honest artifact: what we tried, what the data
said, what we changed, and why.

This is the Agentile methodology applied to research rather than product. It
drops into the repo as a `.agentile/planset/` directory so Claude Code can carry
it forward.

## 2. Agentile, in one page

Agentile treats agents (human and AI) as first-class contributors whose work is
governed, logged, and gated. Three rules carry most of the weight:

1. **Every change is proposed, reviewed, and gated.** Nothing reaches a higher
   rung of the scale ladder without passing its gate. Gates are defined in the
   Master Plan §5.
2. **Every claim is anchored to evidence.** A capability claim points at a
   measurement. A design claim points at an ablation or a prior-art reference. If
   there is no evidence yet, the claim is labeled a hypothesis, not a result.
3. **Every stage emits a case study.** The case study is the unit of
   institutional memory. It is written when the stage closes, while the context
   is fresh, and it is short.

## 3. The planset directory

```
.agentile/
  planset/
    README.md                  # how to use this planset
    gates.yaml                 # gate definitions + exit criteria (machine-readable)
    hypotheses.md              # live list of open hypotheses + status
    case-studies/
      CS-00-forward-pass.md
      CS-01-routing-differentiation.md
      CS-02-gguf-roundtrip.md
      CS-03-federated-gather.md
      ...
    decisions/
      ADR-0001-hybrid-routing.md
      ADR-0002-ssm-in-temporal-zones.md
      ADR-0003-provenance-as-output.md
      ...
    experiments/
      EXP-<id>/                # one dir per run: config, seed, metrics, notes
```

`gates.yaml` and `hypotheses.md` are the two files everyone reads first.
`gates.yaml` is machine-readable so CI can check whether a gate's acceptance
features are green before a merge to the `main` of a higher rung.

## 4. Case-study protocol

Every stage closes with a case study using this fixed template. Fixed structure
is the point... it makes the studies comparable and keeps them honest.

```markdown
# CS-<id>: <stage name>

**Gate:** <which gate this serves>
**Rung:** <L0 | L1 | L2 | L3>
**Dates:** <start> .. <end>
**Authors:** <human + AI contributors>

## Question
The one thing this stage was trying to find out. One sentence.

## Setup
Model size, data slice, seed, hardware, config hash. Enough to reproduce.

## What we measured
The metric(s), defined precisely. Baseline value. Target value.

## Result
The number. Plotted if it helps. State it before interpreting it.

## What surprised us
Honest section. What did not go as expected.

## Decision
What we changed, kept, or killed as a result. Links to the ADR if one was filed.

## Open threads
Hypotheses spawned or closed. Links to hypotheses.md entries.
```

A case study that has no "what surprised us" content is suspect. Real research
surprises you. If a stage produced no surprise, say so explicitly and note why
the result was that predictable.

## 5. Decision records (ADRs)

Architectural decisions get a one-page ADR so future engineers (and counsel,
for IP chain-of-title) can see the reasoning. The seed decisions from the design
conversation are already worth recording:

- **ADR-0001 — Hybrid routing.** Fixed inter-zone topology, context-aware learned
  modulation. Rejected: pure hard-wiring (too rigid), pure learned MoE routing
  (loses interpretability). Chosen because it keeps the topology auditable while
  letting signal strength adapt per prompt.
- **ADR-0002 — SSM in temporal zones.** Cerebellar and Sensorimotor zones use
  State Space Model cores for linear-time recurrence and native temporal
  dynamics; Prefrontal, Hippocampal, and Codec keep attention. Cross-zone
  communication uses a small attention head per zone.
- **ADR-0003 — Provenance as a forward-pass output.** The zone-activation trace,
  confidence scores, prune decisions, and tool routing are emitted as structured
  output, hashable for on-chain commitment. Rejected: post-hoc interpretability
  tooling, which cannot prove what actually ran.
- **ADR-0004 — Auxiliary sidecar format.** GGUF/ONNX remain the tensor container.
  A sidecar carries the zone graph, routing topology, training recipes, and
  composition rules. Rejected: forking GGUF, which breaks the Ollama onramp.

## 6. Hypothesis ledger

`hypotheses.md` tracks every capability or design claim that is not yet proven.
Each entry:

```
H-<id> | <statement> | status: open|supported|refuted | evidence: <CS link>
```

Seed hypotheses from the design work:

- **H-01** Zone partitioning does not reduce capability per parameter versus a
  dense baseline of equal size. *(open... this is the load-bearing bet)*
- **H-02** Context-aware routing produces measurably different zone mixes for
  different prompt classes. *(open)*
- **H-03** The provenance trace is faithful... replaying the logged zone mix
  reproduces the output. *(open)*
- **H-04** SSM temporal zones cut per-zone compute meaningfully versus attention
  at equal sequence length. *(open, but well-supported by SSM literature)*
- **H-05** A federated cycle reproduces the centralized result within tolerance.
  *(open)*

H-01 is the one that decides whether the whole bet pays off. The L0/L1 ablation
against a dense baseline (Master Plan risk register, row 1) exists to test it
before the expensive L2 commit.

## 7. Documentation cadence per stage

| Stage event | Artifact produced | Where it lands |
|-------------|-------------------|----------------|
| Stage kickoff | Question + hypotheses logged | `hypotheses.md` |
| Mid-stage decision | ADR if architectural | `decisions/` |
| Experiment run | Config, seed, metrics | `experiments/EXP-<id>/` |
| Stage close | Case study | `case-studies/` |
| Gate review | Gate sign-off recorded | `gates.yaml` |

## 8. Reproducibility floor

No stage closes without:
- A config hash that pins model, data slice, and hyperparameters.
- A fixed seed (or a logged seed if randomized).
- Hardware and software versions recorded.
- The exact command to rerun.

This floor is not bureaucracy. It is what lets a federated contributor... the
"grandma-proof" node operator... trust that the model they are training toward
behaves the same way the reference does.

## 9. How AI contribution is logged

Per Agentile, AI work is governed and attributed, not hidden. Each case study and
ADR names its contributors, human and AI. The Journal (`JOURNAL.md`) is the
narrative companion to this... it records how the architecture was actually
designed, including which ideas came from where. For IP chain-of-title, that
attribution record matters; counsel should treat the Journal and the ADRs as
part of the conception record, subject to their review.

## 10. Handoff

This strategy is meant to be lifted into the repo as-is. Claude Code refines the
templates into real files, wires `gates.yaml` into CI, and starts populating
case studies from Gate 2 onward. The first case study to write is
`CS-00-forward-pass.md`, the moment the L0 forward pass runs.
