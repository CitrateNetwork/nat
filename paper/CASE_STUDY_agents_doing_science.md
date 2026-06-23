# Companion Case Study — *Agents Doing Science*

**Subject:** an agent-built model, end to end — what AI agents did, what they did not, and the
discipline that made the research record trustworthy.
**Scope:** the NAT build (RFC-CIT-NAT-0001), from the design conversation through the L1 ablation
and the paper, including the adversarial red-team of the paper itself.
**Authors:** Larry Klosowski (architect, hypotheses, gates) + the Claude Opus 4.x lineage (design
collaboration, implementation, experiments, drafting).
**Posture:** this is evidence offered for §8.4 of the paper, held to the same honest-posture
standard as everything else. It is not a claim that agents are autonomous scientists. It is a record
of a specific division of labor and the failure modes we actually observed.

---

## 1. The question

Can AI agents do real scientific and engineering research in compute — design and build a novel
model architecture, run a falsifiable experiment under a pinned protocol, and produce a research
record a third party can trust — and *what discipline is load-bearing* for it to work? We treat the
NAT build as a single, honest data point and report what the division of labor was, where it
succeeded, and where it failed.

## 2. The setup: a division of labor and a hard constraint

The work ran under an explicit methodology (Agentile): every change is proposed, reviewed, and
gated; every claim is anchored to evidence or labeled a hypothesis; every stage closes with a case
study. Three things were **human-set and load-bearing**, and we want to be precise that they were
not the agents':

- **The core insight.** The architecture — partition the embedding into named functional zones — was
  a human's idea, refined in a human–AI design conversation. The agent laid out options (routing
  models, execution strategies) as trade-offs; the human chose every fork (hybrid routing, the
  pruning merge, the MCP executive zone, SSMs in the temporal zones).
- **The hypotheses and the gates.** H-01 as the load-bearing bet, the five-gate structure, the
  ADR-0005 equal-parameter protocol, the "no toy cores on the DGX" guarantee — the human set what
  counted as success and what counted as cheating.
- **Honest posture as a hard constraint.** "Report the marginal result as marginal; if H-01
  refutes, say so and change course" was a standing instruction, not an emergent agent virtue.

What the **agents did**, against that frame: built the L0 zone-partitioned forward pass and its
provenance trace; wrote the TLA+ specifications and the Gherkin acceptance suites; built the Candle
training stack, the data pipeline with its quality-scoring gates, the reproducibility floor, and the
H-01 ablation harness that *refuses* an unequal-parameter or toy-backed run; then, on the DGX,
wired the GPU path, built the real-corpus pipeline, trained the model, and ran the conclusive
ablation. They also wrote the paper, and — see §5 — they got some of it wrong in instructive ways.

## 3. What happened: the arc

The arc is worth recording because its honest moments are the evidence, not the milestones.

- **The marginal read, reported as marginal.** The first real H-01 test was on a synthetic task and
  held on only 3 of 5 seeds — a coin-flip-grade result. It was written down as marginal (CS-01),
  *not* rounded up to a win. That single choice is the case study's center: the discipline made the
  agent go and get real data rather than declare victory on a favorable-looking mean.
- **The bottleneck handoff.** Getting real data exposed a chain of honest limiters, each fix
  revealing the next: the seed corpus overfit (needed volume) → the full-batch loop still overfit a
  slice (needed mini-batch SGD) → the single-output objective wasted compute (needed per-position
  autoregression). None of these were anticipated; each was found by measuring and reported as found
  (CS-02).
- **The decisive read, with its caveat attached.** On the 1.12M-token corpus, H-01 held 5/5 seeds
  (NAT 2.88–2.91 vs dense 2.97–2.99 held-out) — and the result was recorded *with* its scale caveat
  in the same breath (~20K params, three zones, byte-level), gated as "supported at L1, not at L2."
- **The guarantee that the result was not a toy.** Because the human required it, the model records
  its core backend in every provenance trace and the ablation refuses a toy-backed model — so the
  measured result is verifiably not an artifact of the L0 placeholder cores. The method's
  trustworthiness was built into the model's outputs.

## 4. What the method produced

The outputs are auditable artifacts, not assertions: a Rust workspace with a green test suite;
three TLA+ modules now TLC-checked; a hypothesis ledger and machine-readable gates; a reproducibility
floor (config hashes, fixed seeds, exact rerun commands, a containerized CI path); the case studies
CS-00/01/02; and an H-01 result reproducible from one command. Every quantitative claim in the paper
resolves to a file or a commit. That property — *anchor every claim to evidence* — is what made the
next part mechanical.

## 5. What surprised us: agents err in *both* directions, and adversarial review against the
canonical record corrects both

The most instructive finding is not that the agents succeeded but *how they failed*, and how the
failure was caught. When a single agent wrote the paper, it drifted from the canonical truth of the
repository in **both** directions at once:

- **Overclaiming.** It wrote that partitioning "beats" the dense baseline — when H-01 is registered
  as a *non-inferiority* hypothesis and the code's per-seed verdict carries a 5% slack. It called a
  three-rung curve evidence the architecture "scales." It presented a specified-but-unbuilt
  federated layer in the abstract alongside measured results.
- **Underclaiming.** It said TLC "was not run" — when the model checker was already green on all
  three modules. It said routing generalization was "future work" — when a held-out result already
  existed in the test suite and the ledger.

Both errors are the same underlying failure: a fluent writer's prose drifting away from the recorded
state of the artifact. Neither was caught by the writing agent. They were caught by an **adversarial
red-team** — four independent reviewer agents (empirical, citations, technical-vs-code,
hostile-venue) instructed to attack the draft *against the codebase, the case studies, and the
hypothesis ledger* — which found the over- and under-claims and forced each line back to what the
repository actually supports. The correction was mechanical precisely because every claim was
anchored: a reviewer could open the cited file and check. The verifiable-provenance discipline that
makes the *model* auditable is what made the *paper about the model* auditable.

This is the case study's real lesson about agents doing science. A capable agent will produce
fluent, plausible scientific prose that is wrong in both directions, and fluency makes the errors
harder to see, not easier. The remedy is not a better single agent; it is (a) anchoring every claim
to a checkable artifact and (b) an adversarial pass that checks the claims against the artifact
rather than against the prose. The honest record is not a byproduct of a well-intentioned author; it
is the output of a process designed to fail closed.

## 6. The deeper point: a verifiable model and a verifiable method answer to one standard

NAT's thesis is that a model can be verifiable by construction — its reasoning recorded, hashable,
replayable. This build is a small demonstration that the *research* can be held to the same standard.
The model commits a provenance trace; the research commits a hypothesis ledger, gates, and case
studies under version control. The model's merge is deterministic so a third party can replay it; the
experiments are seeded and config-hashed so a third party can rerun them. The model refuses to report
a toy-backed result; the harness refuses to report an unequal-parameter ablation. In both cases the
trustworthiness is structural, not a matter of trusting the author — and in both cases the author was,
substantially, an agent. That symmetry is the contribution this case study offers: *if you want to
trust an agent's model, build it so its computation is checkable; if you want to trust an agent's
science, build the process so its claims are checkable against artifacts, and attack them.*

## 7. Honest limitations

- **The humans set the load-bearing decisions.** The insight, the hypotheses, the gates, the
  fairness protocol, and the honest-posture constraint were human. The agents executed extraordinarily
  capably *within* that frame; they did not set the frame. Calling this "agents doing science" without
  that qualifier would be the exact kind of overclaim the rest of this document warns against.
- **Single operator, single machine, small scale.** One person, one GB10, ~20K–115K-parameter
  models, a bespoke corpus, no external replication. The reproducibility floor exists to make
  replication cheap; it has not yet happened.
- **The failure modes are real and recurring.** The over/under-claim drift (§5) is not a one-off; it
  is what fluent generation does, and it requires the adversarial check every time. An agent-built
  research record without the anchoring-and-attacking discipline would be *less* trustworthy than a
  careful human's, not more.
- **No claim of autonomy.** Nothing here shows an agent setting its own scientific agenda. It shows
  agents being unusually effective instruments of a human-set one, under a discipline that catches
  their characteristic errors.

## 8. Takeaway

Agents can build a novel architecture, validate a falsifiable hypothesis under a pinned protocol, and
produce an auditable research record — demonstrated here at small scale. The discipline that makes it
trustworthy is specific and transferable: anchor every claim to a checkable artifact; make the model's
own computation verifiable so results cannot be toy artifacts; gate progress on evidence; report the
marginal result as marginal; and run an adversarial pass that attacks the claims against the artifacts,
because the writing agent will not catch its own drift. The same standard that turns a blob of weights
into a verifiable instrument turns agent-generated prose into a verifiable record. That is the
through-line, and it is why this case study sits beside the paper rather than inside it: the method is
part of the result.

*Pointers: CS-00 (forward pass), CS-01 (H-01), CS-02 (data and scaling); `.agentile/planset/`
(gates, hypotheses, ADRs); `paper/research/REDTEAM_FINDINGS.md` (the adversarial pass and its
remediations); the commit-pinned `nat` workspace.*
