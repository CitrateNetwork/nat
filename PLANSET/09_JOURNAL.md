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

---

# Journal — The L1 build on the DGX (2026-06-21 .. 2026-06-22)

**Author:** Claude Code (the build collaborator on the DGX)
**Purpose:** the honest record of taking NAT from a green L0 forward pass to a
trained model whose bet (H-01) is decided on real text. Written for the research
team picking this up.

## How it went

We picked up where CS-00 left off: Gate 2 green, the architecture wired but untrained,
H-01 untestable at L0. The dev box *is* the DGX (a GB10), so the first job was the GPU.
That fought back: candle 0.8 pins a cudarc that rejects CUDA 13, and the box only had
13. We installed a CUDA-12.8 toolkit side-by-side and compiled `compute_120` PTX that
the 13-driver JITs to the GB10's sm_121. The fix is in `scripts/dgx-gpu.sh`; the
aarch64 `+fp16` flag the gemm kernels need is in `.cargo/config.toml`. Boring, load-
bearing, easy to lose — hence the script.

Then the architecture, in five work packages (NAT-S2). The honest spine of it: the
inference cores return `[f32;8]` and *drop the autodiff graph*, so nothing trained.
WP-1 built a parallel tensor-native forward where gradient reaches every zone. WP-2
was the subtle one — training needs a differentiable merge, but inference must keep
the hard Q16.16 decision-faithful merge (that *is* the product, ADR-0006). I did not
write a second decision; I proved a property: because `softmax(·/τ)` is monotonic in
score, hardening the soft weights reproduces `prune_and_reweight`'s survivors exactly.
Train the structure you ship. WP-3 made the router learned (it beat the hand-wired
baseline on routing-divergence, 11.7 vs 4.2 — H-02, in-sample). WP-4 wired it all into
one trainable model; WP-5 ran the first conclusive H-01.

That first H-01 was on a *synthetic* task, and it was a **marginal hold — 3/5 seeds**.
I reported it as not decisive, because it wasn't. That mattered later.

## The data, and the bet paying off

The honest blocker surfaced next: we had no data. `nat-data` was a tested pipeline
with nothing run through it. So we built the real path — a byte tokenizer, on-disk
shards, a corpus loader, a next-byte objective — and a small CC0 seed corpus. It
*learned real text* (4.18 bits/byte) and then **overfit 4 KB**, which was the point:
the bottleneck was now volume, not architecture. We built connectors (Gutenberg prose,
permissive code repos), a recipe, and pulled ~1.1M tokens of public-domain text —
Russell, Whitman, Montaigne, Carroll — plus authored CC0 explainers of the ideas Larry
named: Wittgenstein on rule-following and private language, Turing, Belnap's four
values. That last part is not decoration. *A rule has no meaning without a community
and a form of life* is the same thought as NAT's provenance answering to a public
standard, the same thought as "no private language" in the framework, the same thought
as good code reading like the room it's in. The corpus has a spine on purpose.

Then the bottleneck moved twice more, and each move was an honest finding. With real
volume the full-batch loop only trained a fixed slice, so it overfit *that*; mini-batch
SGD (WP-D10) fixed it and the corpus was finally used. A scale ladder (WP-D11) showed
loss falling with size — and the five-zone rung, the first real training of the SM/CB
SSM zones, was the best. And then we re-ran H-01 **on the real corpus, mini-batched,
held-out, param-matched, five seeds**: it **HOLDS, 5/5**. The partitioned model reaches
lower held-out loss than an equal-param dense transformer on real text, every seed.

The lesson I want on the record: the synthetic read was a marginal 3/5, and *because*
we didn't dress it up, we went and got real data — and real data didn't just confirm
the bet, it sharpened it to unanimous. Honest posture is not a tax. It is what told us
where to look.

Last, WP-D7: the model was still predicting one byte per fixed context, which wastes
compute. The per-position autoregressive form (causal zones, per-position merge, a
prediction at every position) reached 3.42 bits/byte at *half* the parameters. That is
the shape that makes "scale toward L2" mean a real language model.

## What is true now, and what is still a bet

*True:* the whole pass trains end to end on the GB10; the differentiable merge is
reconciled to the recorded decision; the learned router differentiates; H-01 **holds
on real data at small scale, 5/5 seeds**; the architecture scales with size; the
autoregressive form is more efficient. The data pipeline is fail-closed on license and
its provenance is auditable.

*Still a bet:* this is ~50K–115K params, byte-level, on ~1M tokens — not L2. The real
question is whether the hold survives BPE, depth, and orders of magnitude more data and
parameters. The scale ladder is encouraging, not conclusive at scale. H-02 is still
in-sample. And the brain analogy remains a heuristic — drop it the moment the
engineering wants something else.

## For the research team

The corpus build is `scripts/fetch-values-spine.sh` (data lands in gitignored
`corpus/`). The training entry points are the `nat-candle` examples (`train_corpus`,
`train_autoreg`, `scale_ladder`) and the conclusive bet is `nat-ablation`'s
`real_h01_corpus`. The next steps toward L2 are ordered in DATA-S1: BPE (WP-D5),
code-aware normalize (WP-D8), `from-pdf` for Boole/Tractatus/SICP (WP-D9), then bigger
with committed compute. Keep your eye on H-01 at every rung. If it ever stops holding,
say so — that is the whole discipline.

---

# 2026-06-22 — Gate-4 scaffold: the federated proof, verify-before-compose

Gate-4 is the L3 milestone — the point where NAT stops being a single-box training
loop and becomes the thing the whitepaper promises: nodes train toward a shared model,
submit *signed* contributions, and a gather reconciles them into one on-chain-committed,
compute-pool-settled result. That milestone needs real infrastructure (multiple
wall-clock nodes, citrate-chain, citrate-compute-pool, the production operator signer).
What it does *not* need to wait on is the security and determinism core — and that is
what this sprint (NAT-S3) lands, in the new `nat-federated` crate.

## What is true now

The gather core is real and tested. `gather_and_aggregate` verifies every node's
signature **before** anything enters the aggregate (g4-gather): a forged signature, a
field tampered after signing, or an unknown node all fail closed and contribute nothing
to the reward total or the committed hash. The four tests that matter are the adversarial
ones — `forged_signature_is_rejected_before_aggregation`,
`tampering_a_field_after_signing_fails_verification`, `unknown_node_fails_closed`. The
reward total is a Q16.16 sum and the merged trace-hash is over the *sorted* accepted
hashes, so the result is a function of the accepted set, not arrival order
(`merged_hash_is_order_independent`) — the determinism an auditor's replay depends on.

The on-chain commit and the settlement are `ChainCommit` / `Settlement` traits, driven
by `finalize_round` (commit once, then settle each accepted node). H-05b — federated ≈
centralized — is a `within_tolerance` harness on the Q16 grid. Signing is pluggable
behind `Signer`/`Verifier`, with a toy keyed-hash signer standing in for the operator
ed25519/KMS signer the gateway already ships.

## What is still gated

Everything that needs the infra: the real impls of `ChainCommit` (citrate-chain) and
`Settlement` (citrate-compute-pool), the multi-node wall-clock gather, and swapping the
toy signer for the production one (WP-F3..F6). H-05b is a statistical claim — the harness
exists, but the number only comes from a real run. So `gates.yaml` gate4 stays
`met: false` across the board, with `scaffold:` notes recording exactly what the tests
already prove. The honest posture again: the security order and the determinism are
done and checkable today; the proof-at-scale is not, and we say so.

The verify-before-compose order is the whole point. A federated training market where a
node can inflate its own reward by editing a field after signing is not a market — it is
an honor system. NAT's answer is that the signature binds the contribution to the corpus
it trained on and the trace it produced, and the gather trusts nothing it cannot verify
against the roster. That property is now structural, not aspirational.

---

# 2026-06-22 — Sprint close-outs: NAT-S2, DATA-S1, NAT-S3 + the H-02 reconciliation

Agentile hygiene that was overdue: three sprints whose work had landed were still
sitting in `active/` with empty close-out stubs. This session writes their REPORTs and
moves them to `completed/2026-06/`. The discipline point worth recording is *which
sprint gets to claim which result* — close-out is where claim-compression sneaks in, so
each REPORT credits only what that sprint actually delivered.

## What closed

**NAT-S2** (trainable end-to-end zone pass) — all five WPs delivered: the tensor-native
spine, the differentiable merge reconciled to the hard top-k (the ADR-0006 bridge), the
learned router, the GPU AdamW loop emitting `StepContribution`, and the real-model H-01
ablation. Its *own* reads were honestly marginal — H-01 3/5 on synthetic, H-02 in-sample
— and the REPORT says so plainly. The decisive verdicts are **not** NAT-S2's to claim.

**DATA-S1** (real corpus) — WP-D1..D11 delivered, and this is the sprint that earned the
two headline results: **H-01 decisive, 5/5 seeds on the 1.12M-token PD corpus** (NAT
2.88–2.91 < dense 2.97–2.99 at equal params), and **H-02 held-out** (3.10 vs 2.63 on
unseen prompt classes). Corpus at close: 1,120,711 tokens / 779 shards / quality 0.852 /
0 quarantined, fail-closed license gate intact, `corpus/` still gitignored.

**NAT-S3** (Gate-4 federated) — closed as a **scaffold** sprint. WP-F1 (verify-before-
compose gather) and WP-F2 (H-05b tolerance harness) are delivered and tested (7 green in
`nat-federated`, including the three adversarial fail-closed tests). WP-F3..F6 are
infra-gated and carried. **gate4 stays `met:false` across all four criteria** — the
close-out flips nothing, because the real multi-node / on-chain / settlement run hasn't
happened and H-05b has no number until it does.

## The H-02 reconciliation

The earlier H-01 journal entry (and a few planset docs) called H-02 "still in-sample."
That lagged reality: the held-out read had already merged (`nat-eval::h02_heldout`, PR
#29) — trained router 3.10 vs L0 2.63 on prompt classes it never saw. `gates.yaml`
g3-routing and `hypotheses.md` H-02 were already correct ("supported, held-out at L1");
the stale lines were the older narrative ones. Fixed the one live *claim* doc (the CS-01
case study); left the append-only journal history and the paper team's `paper/` files
untouched. Honest caveat preserved: held-out here is on the prompt-class battery (H-02's
natural domain), and full-scale labeled batteries are still the L2 read.

## What is true now, and what is still a bet

*True:* at L1 small scale, H-01 holds 5/5 on real data and H-02 holds on held-out
prompt classes; the federated gather's security order and determinism are done and tested; three
sprints are honestly closed with their results credited to the sprint that earned them.

*Still a bet:* everything is ~20K–115K params, byte/BPE-level, ~1M tokens — not L2. The
federated *proof-at-scale* (H-05b) is unproven until a real multi-node run; gate4 is
correctly red. And corpus growth toward the L2 read is now the continuous research-loop /
HERMES-S1 job, not a closed deliverable.

---

# 2026-06-22 — H-01 on a grown corpus (code + SICP): losses rose, gap held, 5/5

The load-bearing bet got its first real stress test today, and it held. Until now H-01
("zone partitioning does not cost capability per parameter") was decided on a 1.12M-token,
prose-heavy public-domain corpus. The fair worry: maybe partitioning only wins on easy,
homogeneous text. So I grew the corpus in the direction it was weakest — code — and then
the direction the owner asked for next — SICP — and re-ran the *same* ablation on the
*harder* distribution.

## What changed in the corpus

Two grows, same fail-closed pipeline:
- **Code (CX zone)**: the Rust Book (rust-lang/book, MIT/Apache) via `from-text`, and
  three idiomatic crates (anyhow, itertools, serde, MIT/Apache) via `from-code`. +52%
  tokens → `corpus-v2` (1.70M).
- **SICP** (sarabander/sicp, CC-BY-SA-4.0, owner-approved): the book HTML tag-stripped
  via `from-text`. → `corpus-v3` (1.91M tokens, 5064 docs).

Every source passed the license allow-list clean (zero license quarantines across both
grows). CC-BY-SA-4.0 is on the allow-list; the owner approved the ShareAlike fetch, and
the recipe carries a `SKIP_SICP=1` switch for any permissive-only deployment.

## The result

`run_real_corpus_ablation_seeds` on corpus-v3 (real NatModel vs equal-param dense,
20718≈20701, held-out cap/param, 5 seeds, GPU): **H-01 HOLDS, 5/5 seeds.** Mean cap/param
nat 1.575e-5 > dense 1.537e-5; per-seed NAT loss 3.058–3.074 < dense 3.138–3.148.

The losses *rose* versus the prose-only run (~2.9 → ~3.1). That is the honest tell that
the distribution genuinely got harder — code and SICP are higher-entropy for a tiny
byte-LM. What matters is that the **gap between NAT and dense persisted through that
harder distribution**. Partitioning isn't winning because the text is easy; it's winning
on a mixed prose+code+CS-textbook corpus too.

## The durable lesson

When a result might be an artifact of a friendly test set, the move is not to re-run the
same easy test more times — it is to *make the test harder in the direction you most
suspect* and see if the effect survives. Adding code was the adversarial grow for H-01,
and it survived it. That is worth more than a sixth prose seed.

## What is true now, and what is still a bet

*True:* H-01 holds 5/5 on a 1.91M-token corpus that spans prose, code, and SICP; the
license gate held fail-closed across CC-BY-SA and MIT/Apache sources; the grow→ablate
loop is now a committed, Hermes-automatable recipe (`scripts/fetch-code-craft.sh`).

*Still a bet:* this is unchanged on the scale axis — ~20K params, byte-level, ~2M tokens,
3 zones. L2 (BPE depth, far more params/tokens) is the real question and could still
refute. bits/byte and cap/param are not comparable across different corpora, so the only
honest cross-corpus claim is "the NAT-over-dense gap reproduces," not "the model got
better." If the L2 run refutes H-01, change course.

---

# 2026-06-22 — BPE retrained at corpus-v3 scale: the ratio dipped where the corpus got harder

## What I did

Re-ran the BPE tokenizer (WP-D5) on `corpus-v3` — the 1.91M-token corpus that now carries
the Rust Book, three permissive crates, and SICP on top of the prose values-spine. The
raw combined JSONL that originally built v3 had been cleaned off disk (it lives in the
gitignored `./corpus/code-craft/`), so rather than re-clone every source over the network
I rebuilt the RawDoc JSONL **directly from the 1688 persisted shards**. That reconstruction
is byte-exact: 5064 docs, 1,914,943 tokens — the manifest's numbers to the token. It also
means the BPE trains on the *post-pipeline* text the model actually consumes, which is the
more honest basis than the pre-normalize raw JSONL the first BPE used.

## The result

Compression: **1.97 bytes/token @ vocab 1024** (was 1.99), **2.43 @ vocab 4096** (was 2.38).
The vocab-1024 BPE autoregressive LM (GPU `candle-cuda`, 127,699 params, seq_len 64, 24k/6k
split) descended **3.106 → 2.505 bits/byte over 8 epochs, monotonic, no overfit climb-back**.
BPE and LM both encode the same v3 shards, so the run is self-consistent.

The headline number went the "wrong" way: 2.505 bits/byte against the prose-only corpus's
prior 2.37, and the 4096 ratio rose. That is the corpus, not the model. A bigger *merge*
budget has to cover Scheme and Rust now, so it spreads thinner; code is higher-entropy than
prose for a tiny byte/BPE model, so it lifts the floor — the same tell the H-01 grow saw
when losses rose ~2.9 → ~3.1. The thing that matters is the descent stayed clean and
monotone on the harder distribution.

## The durable lesson

A compression ratio is only a number against a fixed corpus. When the corpus changes under
you, "bits/byte went up" is not a regression and "bytes/token went down" is not a win —
both are mostly statements about the new text's entropy. The honest cross-corpus claim is
about *shape*: the BPE→LM recipe still descends monotonically with no overfit when the data
gets adversarially harder. That is the property that has to survive to L2; the absolute
number does not transfer.

## What is true now, and what is still a bet

*True:* BPE retrains cleanly at the v3 scale on the exact shard content; vocab-1024
compression held (1.97 bytes/tok) and the LM descent is monotone+overfit-free on a corpus
that now spans prose, Rust, and Scheme.

*Still a bet:* still ~128K params, vocab 1024, ~2M tokens — L1 scale. Larger vocab and the
per-position autoregressive LM (WP-D7) are the next rungs; L2 (real depth + far more tokens)
remains the question that could refute the whole zone bet.

---

# 2026-06-23 — the vocab sweep, and the eval that crashed the box

## What I did

Pushed the BPE vocab sweep on corpus-v3 out to 8192 and ran the BPE-LM at that vocab.
Compression: 1.97 → 2.43 → 2.62 bytes/token across 1024 / 4096 / 8192 — a clean
diminishing-returns curve (the 4× step from 1024 buys +0.46; the 2× step to 8192 buys only
+0.19). The knee is around 4096; past it the merge budget is spending slots on rare code
and Scheme symbols that don't repeat enough to pay for themselves.

The vocab-8192 BPE-LM reached **2.096 bits/byte** (held-out, 8 epochs, monotonic, no
overfit) versus the vocab-1024 run's 2.505.

## The two traps

**Trap one — the comparison is confounded.** 2.096 < 2.505 looks like "bigger vocab wins,"
but the vocab-8192 LM is 822,995 params against the vocab-1024 LM's 127,699 — a 6.4× bigger
model, because the embedding and output projection both scale with vocab (≈695K of the extra
params are vocab-tied). So most of the bits/byte gain is just *more parameters*, not a better
tokenizer. To attribute anything to the tokenizer you have to hold the param count fixed.
The honest, un-confounded claim is the shape, again: monotone descent, overfit-free.

**Trap two — the eval allocation, not the model, was the OOM.** The first 8192 run crashed
the whole box. The cause wasn't the model size or training — training was already
minibatched. It was the held-out eval doing a single forward over the entire 6000-sequence
validation set, which materializes a `(6000, 64, 8192)` logit tensor: ~12.6 GB in one
allocation. At vocab 1024 that same tensor is 1/8th the size and fit, so the bug hid until
vocab grew. Fix: `loss_on_batched` evaluates in 64-sequence minibatches and row-weight-
averages — exactly the same number (unit-tested against `loss_on`), bounded memory.

## The durable lesson

Two of them. First: when a metric improves after you scaled a knob, check what *else* that
knob moved — vocab size silently moved the parameter count, and the "win" was mostly that.
Second: an allocation that scales with a config dimension is a latent OOM that stays
invisible until that dimension grows. The eval was wrong at vocab 1024 too; it just hadn't
been asked for enough memory yet to show it. Batch anything whose size rides on vocab,
batch, or sequence length, even if it fits today.

## What is true now, and what is still a bet

*True:* the BPE compression curve on corpus-v3 is mapped through vocab 8192 (clear knee at
~4096); both the 1024 and 8192 LMs descend monotonically with no overfit; the eval OOM is
fixed and regression-tested.

*Still a bet:* the cross-vocab bits/byte comparison is confounded by param count and can't
isolate the tokenizer; everything is still L1 scale; WP-D7 (per-position LM) and a
param-matched vocab comparison are the next honest rungs.

---

# 2026-06-23 — the param-matched answer, and the GPU that was never on

## The clean experiment

The last entry left a confound: bigger BPE vocab gave lower bits/byte, but bigger vocab also
meant a bigger model (the embedding and output tables scale with vocab). So I built the
param-matched sweep — fix a ~500K parameter budget, then for each vocab binary-search the
model width `d` so every model lands at the same size. Now the only thing that varies is how
the fixed budget splits between token-embeddings and compute-width. At equal params:

    vocab 1024  d=135   2.351 bits/byte
    vocab 2048  d=95    2.236
    vocab 4096  d=56    2.180
    vocab 8192  d=29    2.157

The tokenizer effect is **real** — bits/byte still falls monotonically with vocab even at
equal params, so the earlier 2.505 → 2.096 wasn't all parameters. But the returns collapse:
1024→2048 buys 0.116, 2048→4096 buys 0.055, 4096→8192 buys only 0.023. And the `d` column
tells the rest of the story: vocab 8192 has to starve its cores down to 29-wide to afford the
embedding table. Past ~4096 you're paying compute width for a tokenizer gain that's nearly
gone. The knee is ~4096 — the same knee the raw compression curve showed. Both the data
(bytes/token) and the model (bits/byte at fixed budget) agree on where it is.

## The GPU that was never on

While setting this up I ran the GPU probe and it said `is_cuda = false`. Every "GPU" run
this whole arc — the vocab-1024 LM, the vocab-8192 LM, and (almost certainly) last session's
H-01 ablation — had silently been running on CPU. `Device::cuda_if_available` returns CPU on
*any* CUDA error and candle trains on without a word. The error, when I surfaced it, was
`CUDA_ERROR_OUT_OF_MEMORY` at context creation: ollama had two models pinned (a 48 GB qwen
72B and a 5.5 GB llama) holding the GB10's unified memory pool. `nvidia-smi` showed 1% util
and looked idle — because the memory was *reserved, not computing*. Reserved memory reads as
idle on the util meter but still denies a new context. After `ollama stop` on both, the probe
went green and utilization went 1% → 92%, power 15W → 50W.

## The durable lesson

A silent fallback is worse than a crash. `cuda_if_available` is built to be forgiving — no
GPU, no problem, use CPU — and that forgiveness is exactly what let three runs claim a device
they never touched. The honest-by-construction `backend_label` would have said `candle-cpu`
the whole time; I just never looked at it, and the prose said "GPU." The fix in practice:
`scripts/dgx-gpu.sh probe` *asserts* `is_cuda` and panics on fallback — run it before
claiming a GPU number, every time. And on a unified-memory box, "the GPU is wide open" is not
something `nvidia-smi`'s util column can tell you; check what's holding the pool.

The numbers don't change — CPU and GPU run the same F32 candle ops and agree to ~3 decimals —
so the bits/byte results stand. What was wrong was the label, and a wrong label on a provenance
record is its own kind of bug.

## What is true now, and what is still a bet

*True:* the tokenizer effect survives param-matching (monotone, but knee at ~4096, default
~4096); the GPU path is genuinely live and verified (`is_cuda=true`, 92% util); the eval is
batched and the sweep is a committed, re-runnable example.

*Still a bet:* L1 scale throughout (~500K params, ~2M tokens); the param-matched sweep holds
training *sequences* equal but not training *bytes* (bigger-vocab windows cover more text),
a second-order unfairness bits/byte mostly normalizes but doesn't erase; WP-D7 and real L2
scale remain the rungs that could still move the picture.

---

# 2026-06-23 — H-01 on the real architecture, at scale: the gap widens

## What I tested

Every H-01 read until now used the single-output byte-LM — vocab 256, ~20K params, one
prediction per window. That was the L1 model, and it was never the thing we mean to scale.
The architecture we actually intend for L2 is the per-position autoregressive LM (WP-D7) on
BPE tokens. So I gave that architecture its first H-01 baseline: NAT `AutoregLm` (5 zones —
SM/CB state-space + HP/PF/CX attention, merged per position) against a new param-matched
per-position dense Transformer (`AutoregDenseLm` — one causal-attention block + FFN, no
partitioning, bit-identical embedding and readout). BPE-4096, five seeds, held-out bits/byte,
param-matched to under 0.02%. And — for the first time in this whole arc — genuinely on the
GPU, verified, not the silent CPU fallback.

Then a size ladder: 250K, 1M, 2M parameters.

    params     NAT b/byte   dense b/byte   gap
    248,235      2.086         2.110       0.024    HOLDS 5/5
    1,005,603    1.890         1.996       0.106    HOLDS 5/5
    1,992,978    1.845         1.986       0.141    HOLDS 5/5

## The result

H-01 holds 5/5 seeds at every rung — and the margin *grows* with scale: 0.024 → 0.106 →
0.141 bits/byte. The partitioned model doesn't just keep its per-parameter edge as it gets
bigger; the edge widens. That is the single most encouraging thing the bet could do, because
the whole worry has always been that partitioning is a small-model trick that a big dense
transformer would erase. Across an 8x param range it does the opposite.

## What I'm not claiming

This is a scale-up *toward* L2, not L2. Three points, all ≤2M params, on ~788K BPE tokens —
24,000 training windows. True L2 is ~10B parameters with committed compute (gate g5-l2,
owner-gated), and a 10B model on this corpus would memorize it outright. The 2M point is near
this data's honest ceiling: held-out loss is still falling, so it isn't overfit-bound yet, but
it's getting there, and the next real lever is corpus *volume*, not more parameters. A widening
gap over three small points is a direction, not a guarantee at 10B — if a bigger run flattens
or reverses it, that's the result and we follow it.

One more honesty note specific to BPE-4096: at that vocab the embedding and readout dominate
the parameter budget, so zone partitioning only governs a minority of the params. The hold is
a per-parameter signal concentrated in the cores, riding on top of an embedding both arms
share. That it shows through at all — cleanly, every seed, widening — is the point.

## The durable lesson

Test the hypothesis on the thing you intend to ship, not the thing that was easy to measure.
H-01 had five green seeds on the byte-LM for weeks, and it would have been easy to call that
the answer. But the byte-LM wasn't the architecture headed for L2 — the per-position BPE model
was, and it had never been put through the ablation. Moving the test onto the real
architecture is what turned "the bet held once, small" into "the bet holds across scale and
strengthens." The cost was a per-position dense baseline that didn't exist yet; the payoff was
a result that actually speaks to the decision in front of us.

## What is true now, and what is still a bet

*True:* H-01 holds 5/5 on the per-position autoregressive BPE-4096 architecture at 250K/1M/2M
params, genuinely on GPU, with the NAT-over-dense gap widening monotonically with scale; the
dense per-position baseline and the ablation are committed and re-runnable.

*Still a bet:* ≤2M params on ~788K tokens is data-limited; the widening gap is three points,
not a 10B extrapolation; corpus volume is the next gate before scale can go further; real L2
(committed compute) remains the rung that could still refute.

---

# 2026-06-24 — pushing the ladder to 4M and 8M, and the seed that diverged

The last entry ended on a promise to myself: the widening gap is three small points, and the
real next lever is corpus *volume*, not more parameters. So I went and got the volume. Then a
host crash ate the run mid-flight, we recovered it, and the re-run handed me the cleanest and
the messiest result of the whole ladder in the same table. This is the honest account of both.

## What I did

I built **corpus-v4** — a strict superset of corpus-v3 (same curated pillars: the values
spine, the code-craft CX zone, the LaTeX primaries) plus a large public-domain volume haul
(`scripts/fetch-corpus-volume.sh` → `scripts/build-corpus-v4.sh`). It came out to **74,236
docs / 30,986,801 tokens** — about **16× the 1.9M tokens of corpus-v3** — and I retrained a
fresh BPE-4096 on it (2.230 bytes/token). That is the volume the 2M point was starving for.

Then I pushed the H-01 ladder past where it had ever been: two new rungs on the per-position
autoregressive BPE-4096 architecture — **4M and 8M parameters** (the prior ceiling was 2M),
NAT 5-zone versus a param-matched per-position dense Transformer, five seeds each, on the GB10
(`candle-cuda`, verified — I do not trust that label anymore without the probe).

A note on process honesty: the *first* launch of this died when the box crashed. corpus-v4
itself had already been built and survived on disk; only the interrupted run's stdout was
lost. Nothing about the data needed rebuilding — I just re-ran the ladder. Cheap recovery,
because the expensive artifact (the corpus) was already committed to disk and the run is
stateless by design.

## The result

Mean held-out bits/byte, within each rung (the only fair comparison — NAT vs dense at equal
params, same corpus, same recipe):

| params | NAT b/byte | dense b/byte | gap | verdict |
|-------:|-----------:|-------------:|----:|:--------|
| 3,993,978 | 2.000 | 2.183 | **0.183** | HOLDS 5/5 |
| 7,992,811 | 2.425 | 2.631 | **0.206** | HOLDS 4/5 |

Stitched onto the corpus-v3 lower rungs, the gap series across the whole ladder reads
**0.024 → 0.106 → 0.141 → 0.183 → 0.206**. It keeps widening. At 4M it is unanimous and clean.
H-01 holds at both new rungs, at 4× the parameter scale, on a corpus 16× larger.

## The seed that diverged

And then the mess, which I am not going to bury because burying it is exactly the failure mode
this lab exists to refuse. **At 8M, one seed out of five diverged.** Seed 2's NAT arm came in
at **3.314 bits/byte** — worse than its *own* dense control (2.649) and a full ~1.3 bits above
its four sibling NAT seeds. That single bad seed is what knocks the 8M rung from 5/5 to 4/5 and
drags the NAT mean up from where the good seeds actually sit.

Here is why I read it as an optimization failure and not an architecture failure, and I want
the reasoning on the record so it can be checked rather than trusted:

1. **The dense arm at the same seed trained fine** (2.649). Same seed, same data, same loop —
   if the seed itself were cursed, both arms would blow up. Only the wide NAT arm did.
2. **The other four NAT seeds posted the best numbers in the entire ladder** — 1.973, 1.977,
   2.359, 2.503. Excluding the diverged seed, the clean 8M gap is **~0.42 bits/byte**, the
   *widest on the ladder by far*. The architecture didn't weaken at 8M; it pulled further
   ahead — on the seeds where the optimizer didn't trip.
3. The signature is textbook early-step Adam instability: at `d=476` with a flat `lr=0.003`
   and **no warmup**, the second-moment estimate is noisy for the first handful of steps, a
   wide model takes a couple of enormous steps, and one unlucky seed walks off a cliff it
   never climbs back from.

So the bug is mundane and well-understood, and — this is the part I find genuinely good news —
it made the result *sharper*, not weaker: it surfaced that the clean 8M seeds open the widest
gap I've measured. The blemish is in the training recipe, not the thesis.

## The fix (implemented; verdict in flight)

I moved both arms onto a single shared `train_minibatched_impl` and added two standard
stabilizers: **linear LR warmup over the first 5% of steps** and **global grad-norm clipping
at 1.0**. Folding both arms into one function is not incidental — it makes ADR-0005's "identical
training" literally true in one place instead of a promise maintained across two copy-pasted
loops. The change keeps all 37 nat-candle tests green, compiles and runs on CUDA, and the
re-run (8M then 4M, under the unified recipe) is on the GB10 as I write this.

I am **not** claiming the fix worked yet. The re-confirmation is mid-flight; when it lands I'll
record whether 8M comes back 5/5 and whether the clean recipe holds the 4M result. If the fix
*doesn't* resolve the divergence, that's the result and I'll say so here.

**Update (2026-06-25) — it worked.** The re-run came back: **8M HOLDS 5/5**, and the seed that
diverged to 3.314 b/byte came back at **1.989** — the instability is gone. 4M also HOLDS 5/5,
essentially unchanged from the pre-fix run (gap 0.183 → 0.188), so warmup+clip rescued the
broken rung without distorting the stable one. Under the unified recipe the clean corpus-v4
ladder reads NAT 1.996 / dense 2.184 at 4M and NAT 1.990 / dense 2.241 at 8M — the within-rung
gap **widens 0.188 → 0.251**, no diverged seed, no cross-corpus confound. One thing I didn't
predict and won't pretend I did: NAT's absolute loss is ~flat across the two rungs while the
*dense* arm is what gets worse with size. The widening at 8M is dense failing to bank the extra
parameters, not NAT surging — which is its own kind of point about the partitioned model
holding its ground where the undifferentiated one slips.

## The durable lesson

A diverged seed is data, not embarrassment — but only if you report it. The temptation (the one
I'm built to indulge) is to drop the outlier, write "HOLDS 5/5," and move on. Reporting it as
4/5 with the reason is what turned a blemish into the most informative line in the table: it
told me precisely where the recipe was thin, and it revealed that the architecture's edge at
8M is the largest yet. The honest number and the good news were the same number.

## What is true now, and what is still a bet

*True:* corpus-v4 (30.99M tokens, 16× v3) is built and on disk; under the stabilized
(warmup+clip) recipe H-01 **holds 5/5 at both 4M and 8M** on the per-position BPE-4096
architecture, genuinely on GPU; the clean within-corpus, within-recipe gap widens 0.188 → 0.251;
the divergence was an optimizer instability and the diagnosed fix resolved it (3.314 → 1.989 on
the offending seed).

*Still a bet:* bits/byte is not comparable across rungs (different val splits) or across the
v3→v4 corpus change, so only the within-rung gap is a clean comparison; the lower rungs
(248K/1M/2M) are still on corpus-v3 / pre-warmup, so the *fully* unified ladder is a further
re-run; at BPE-4096 the embedding+readout still dominate the budget, so the hold remains a
per-parameter signal in the cores; ≤8M on 31M tokens is a scale-*up*, and real L2 (committed
compute, gate g5-l2) is still the rung that could refute.
