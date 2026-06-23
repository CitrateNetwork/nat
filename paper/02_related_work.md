# §2 Related work

NAT sits at the intersection of several active lines. We position it against each, and in each
case the distinction is the same shape: where the prior line discovers structure, wraps a black
box, or averages disagreement, NAT *declares* structure and *records* it. (Citations are detailed
in `research/RELATED_WORK_AND_CITATIONS.md`; arXiv identifiers are verified before submission.)

**Modular deep learning and mixture-of-experts.** Modular architectures route inputs through a
subset of parameter-efficient modules and aggregate the result, with compositionality and
parameter efficiency as the stated advantages [Pfeiffer et al. 2023]. Sparse mixture-of-experts
realizes this for transformers — Shazeer et al. (2017), GShard [Lepikhin et al. 2020], Switch
[Fedus et al. 2022], Mixtral [Jiang et al. 2024] — replacing dense feed-forward blocks with gated
experts to scale capacity at near-constant inference cost. NAT shares the conditional-computation
instinct but inverts the routing's epistemics: MoE experts are *discovered, interchangeable, and
unnamed*, and the gate is itself a black box whose assignment carries no semantics. NAT's zones
are *declared and named*, wired over a **fixed** topology a learned router only modulates and
cannot extend. MoE optimizes throughput via sparse activation; NAT optimizes auditability and
capability-per-parameter at *equal total parameters* (the H-01 question, §6), recording every
zone's participation rather than only the chosen expert's.

**Brain-inspired language models.** The closest prior art is BriLLM [2025], which also takes
neuroscience as its starting point and also claims full interpretability, via a "signal
fully-connected flowing" mechanism over a graph that *replaces attention entirely*. The
differences are decisive. BriLLM abandons the transformer and with it the GGUF/ONNX/Ollama
ecosystem; NAT keeps the transformer (attention and state-space cores inside zones) and stays
ecosystem-compatible through a sidecar (§3.6). BriLLM's interpretability is node-level token
mapping — which token-node lit; NAT's is functional-zone provenance — which declared function
fired, with what confidence, what was pruned, and why — recorded as a **deterministically hashable,
on-chain-verifiable artifact**, not merely a readable graph (§4). And NAT advances a falsifiable
per-parameter capability claim against an equal-parameter dense baseline, which BriLLM does not.

**State-space and hybrid sequence models.** State-space models — S4 [Gu et al. 2021], Mamba [Gu &
Dao 2023], Mamba-2 [Dao & Gu 2024] — offer linear-time recurrence and an explicit state, and
hybrids such as Jamba [Lieber et al. 2024] interleave SSM and attention *layers* to balance
long-context efficiency against in-context learning. NAT uses both, but organizes the hybrid *by
declared function* rather than by interleaving: SSM cores live in the temporal zones (SM, CB),
attention cores in the reasoning zones (HP, PF, CX), each SSM zone carrying a thin attention head
for cross-zone talk (§3.1). The five-zone scale-ladder rung (§6.3) is the first real-data training
of these SSM zones.

**Interpretability.** Mechanistic interpretability has made genuine progress at reverse-engineering
trained transformers [Rai et al. 2024; Bereska & Gavves 2024], and its taxonomy distinguishes
intrinsic (before training), developmental, and post-hoc methods. The field is overwhelmingly
*post-hoc*: it reconstructs, after training, a structure the architecture never committed to, and
so it can approximate but not prove what ran. NAT's contribution is to make interpretability
*intrinsic by construction* — the zones are declared, so "which functional component produced this"
is an architectural fact, and the provenance trace makes it a recorded, replayable one. This is the
lead thesis: structure is interpretability.

**Verifiable and zero-knowledge ML.** Zero-knowledge ML proves, after the fact, that an opaque
computation ran on committed weights — ZKML [Kang et al. 2024], zkLLM [Sun et al. 2024], zkGPT
[2025], surveyed in [2025], with tools like EZKL and on-chain systems verifying up to ~18M
parameters. These verify the *output* at heavy cost (hundreds of seconds per proof for a small
transformer) and say nothing about the *reasoning*. NAT is verifiable *by construction*: the
decision-faithful trace is replayable with no per-inference proof, on the same Q16.16 substrate
Citrate's verifiable-inference precompiles use (Paper X), with which it composes when bit-exact
certification of the numeric layer is required (§4.4).

**Decentralized and federated training.** Federated averaging [McMahan et al. 2017] and
low-communication distributed training — DiLoCo [Douillard et al. 2023], OpenDiLoCo [2024],
DiLoCoX [2025] — train *monolithic* models as replicas synchronized periodically, robustly across
poorly connected, heterogeneous workers. NAT federates *composable named zones* instead of whole
replicas: a node owns and trains a zone, submits signed zone outputs, and the deterministic merge
reconciles independently-trained contributions (§7). Where DiLoCo averages, NAT aggregates with
paraconsistent semantics (next).

**Decentralized science and blockchain AI.** Networks such as Bittensor (Yuma Consensus scoring
miner outputs) and Gensyn (a compute marketplace), and the broader DeSci/DeAI movement [SoK 2024;
DeScAI 2025], incentivize *opaque* model and compute contributions scored by validators — you trust
the score, not the computation. NAT supplies the verifiable substrate that line lacks: a
contributor's work is a signed, provenance-traced, Q16.16-deterministic update whose value is
compute × data-quality, settled by `citrate-compute-pool` (§7).

**Paraconsistent logic.** Belnap's four-valued logic [Belnap 1977], with values true, false, both,
and neither — surveyed for computer science in [2025] — is designed for reasoning from multiple
inconsistent or incomplete sources. NAT uses it as the federated-aggregation logic (§7): when nodes
with different data train the same zone, per-dimension contributions resolve to a Belnap state, and
genuine disagreement (Both) and genuine ignorance (Neither) are preserved as first-class rather
than averaged away, operationalizing the consensus mechanism specified in Citrate Paper II.

**Scaling laws.** Compute-optimal scaling [Hoffmann et al. 2022; Kaplan et al. 2020] holds
architecture fixed and trades off parameters, data, and compute. H-01 asks the orthogonal
question those laws hold constant: at fixed parameters, data, seed, and compute, does *structure*
change capability per parameter? Our answer, at small scale, is yes (§6).
