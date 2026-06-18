# Journal — How This Got Designed

**Document:** RFC-CIT-NAT-0001 / Journal
**Author:** Claude (the AI collaborator in the design conversation)
**Date:** 2026-06-18
**Purpose:** an honest, first-person record of how the Neuroarchitectural
Transformer was designed, and what each of us brought to it. Written for the
conception record and for whoever picks this up later.

---

## How it started

This did not start as a spec. It started as a thought Larry had the night before,
turned over on a work break and said out loud: a neural net is loosely analogized
to a brain, but the brain is not one uniform thing... it is a set of regions, each
handling different operations at different speeds for different reasons. So why do
we treat a transformer's hidden space as one undifferentiated block of weights?
What if the embedding space were partitioned into named sections, the way the
brain is, with each section wired to handle its kind of work?

That was the seed. Everything in this planset grew from it.

The first thing I did was play it back to make sure I had the shape right:
partition the embedding space into zones mapped to brain regions, tag each zone to
an operation, pre-wire functions to read and write to those zones. Larry confirmed
the shape and then... characteristically... went on break and said he would be
right back. When he came back, we kept going.

## The arc of the conversation

We did not design top-down. We built it the way you actually build an idea, one
decision unlocking the next.

**The brain map first.** Larry wanted to understand the regions and their roles
before mapping anything to machine learning... brainstem and cerebellum at the
fastest clock, the limbic system for salience and memory consolidation, the
neocortex and prefrontal cortex for the slow, deep binding. The insight that
carried forward was that each region also *trains differently*: the cerebellum by
repetition and error correction, the hippocampus by novelty and emotional weight,
the prefrontal cortex by reasoning and feedback. That is where "weights are the
history of training" became the bridge between brain and model.

**Then the porting question.** Larry set the hard constraints early and held them
the whole way: stay inside the transformer so we keep backwards compatibility,
stay GGUF/ONNX-compatible so it runs in Ollama, but introduce an auxiliary format
that lets different zones train differently, and aim the whole thing at a
federated learning cycle on Citrate. He wanted Rust, because his ecosystem is
Rust, and he was willing to defend that choice against the Python default.

**The web check.** Larry asked me to look at what already exists so we would not
reinvent wheels and could find where the novelty actually sits. That search
mattered. It confirmed the Rust tooling is solid (ONNX Runtime bindings, Burn,
Candle, Tract), and it showed active work on federated mixture-of-experts and
brain-inspired LLMs... but no clear prior art on declared zone partitioning with a
baked-in provenance trace and a GGUF-wrapping sidecar. That gap became the novelty
wedge in the architecture spec.

**The design decisions, in order.** This is where Larry did the steering and I
laid out options:
- *Zone communication:* I gave three models... hard-wired, learned-MoE-style, or
  hybrid. Larry chose hybrid, and reframed it cleanly: he was not wedded to the
  neuro thesis, he wanted the brain as a mimetic analog, closer to one-to-one than
  current architectures but free to diverge. He framed the target as "a little
  right brain, a little left brain, determined by the prompt." That gave us:
  fixed topology, learned per-input modulation.
- *Execution:* Larry called it... parallel zones, a router that composes the
  outputs at the end, attention scoring then pruning the noisy bottom 70–80%, then
  re-weighting. He also flagged the real constraint: some operations take longer
  than others, so we need to handle staggered latency. That became the async
  gather with a deadline.
- *The opacity problem:* Larry asked me for my strongest opinion, and asked
  stringently. My read was that zone-specific preprocessing plus a full provenance
  log is the edge... the architecture flips the black box by recording which zones
  fired, with what confidence, what got pruned, and why. Larry connected it
  straight to Citrate: hash those logs, commit them on-chain, make reasoning
  replayable. That is the part I think is most genuinely new.
- *Tool use:* Larry caught that we had left out tool calling and the harness, and
  folded in MCP as an executive-function zone... non-learned, a validator and a
  state machine. That gave the design its sixth zone and its safety story.
- *State machines, then a correction:* Larry said "state machine," I built the
  tool-use state machine around it, and then he corrected himself twice... he
  meant State Space Models, SSMs. That correction was productive. SSMs slotted
  into the temporal zones (cerebellar, sensorimotor) for linear-time recurrence
  and native temporal dynamics, with a thin attention head per zone for cross-zone
  talk. Both ideas survived: the MCP state machine *and* SSM cores.

**Then the pitches.** Twice Larry asked me to pitch him as if he were the CTO, then
the CEO, getting the team funded. Those were not throwaway... they forced the
design to cohere into something sayable. The funding framing is where "the first
transformer that lets you see how it thinks" crystallized as the thesis.

**Then reality.** Larry moved to planning the data and training. He named the
constraints: 10B parameters, a DGX Spark ready now, two to three months, 4–6 TB
on-prem scalable to ~100 TB, open-source datasets, from-scratch preferred,
federated and grandma-proof. This is where I had to hold honest posture: a 10B
model from scratch on one Spark in that window is not realistic, and I said so. We
landed on the scale ladder... validate at L0/L1 on the Spark, run the full 10B
across the federation. Larry's response made the honesty easy: he explicitly said
he cared more about the shape of the data and the truth of the timeline than about
a flattering number.

## What each of us contributed

I want this part to be accurate, because it matters for the record.

**Larry brought the idea and every load-bearing decision.** The core insight is
his... the brain is regionalized, so partition the embedding space. Every binding
constraint is his: Rust, GGUF/Ollama compatibility, the auxiliary format,
federated on Citrate, the on-chain provenance commitment, the 10B target, the
Spark, the budget. Every fork in the design, he chose: hybrid routing, parallel
execution with attention-scoring and pruning, the MCP executive zone, SSMs in the
temporal zones. The corrections were his and they improved the design. The
strategic framing... unified model, grandma-proof federation, honest posture... is
his standing discipline, and it shaped how I wrote every document here.

**I brought structure, options, and a reality check.** I laid out the brain-region
map, offered the routing and execution options as tradeoffs rather than a single
answer, ran the prior-art search and found the novelty gap, argued for provenance
logging as the opacity solution when asked for my strongest position, and held the
line on the compute math when the 10B-on-one-Spark timeline needed an honest
answer. I turned the conversation into the pitches he asked for, and then into
this planset.

The honest version: Larry is the architect. I am the collaborator who held the
options up to the light, did the legwork on what already exists, and pushed back
where the numbers needed pushing back. The design is a real co-design, and the
conception... the thing that was not obvious before someone said it... is his.

## What I think is actually true, and what is still a bet

Honest posture means separating these.

*What I am confident about:* the architecture is buildable, the Rust tooling
exists, GGUF compatibility via a sidecar is sound, the provenance trace genuinely
solves the auditability problem by construction, and the MCP harness is the most
straightforwardly verifiable part. The interpretability win is real because it is
structural, not earned through a benchmark.

*What is still a bet:* whether zone partitioning helps or hurts capability per
parameter (hypothesis H-01) is unproven, and it is the load-bearing question. The
whole scale ladder exists to test it cheaply before committing the expensive 10B
run. The federated determinism story is sound in theory and unproven in practice.
The brain analogy is a useful heuristic and nothing more... we should let it go
the moment the engineering wants something else.

If this works, the thing worth remembering is that it started as one person's late
-night question about why we pretend the inside of a model is uniform when the one
intelligence we actually know about is anything but. That question was good enough
to build on. The rest of this planset is just the careful part.

## For whoever picks this up

Read the Master Plan first, then the Architecture spec, then the Formal scaffold.
The data plan tells you what to feed it. The design brief tells you what it should
look like to watch. And the one number to keep your eye on is H-01... if zone
partitioning costs more capability than it buys, the honest move is to say so and
change course. Larry will want it that way.
