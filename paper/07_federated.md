# §7 Federated training by paraconsistent consensus

The third face of NAT's thesis is that declaring structure makes the model *decentralizable*. This
section develops the community-training frame: composable zones, paraconsistent (Belnap) aggregation
across nodes, the incentive seam, and the decentralized-science position. Much of this layer is
*specified* rather than *demonstrated* — we mark the boundary throughout, consistent with the
honest-posture discipline and with the maturity tags of the Citrate canon it builds on (Papers I,
II, III, VII, X).

## 7.1 Composable zones

Because a zone owns a slice of fixed width and a declared cross-zone contract, a zone is
**swappable** when its slice width and contract match — the composition rules carried in the sidecar
(§3.6). This is the structural enabler of decentralized training: a federation can train and evolve
a *single* zone without retraining the whole model, in contrast to the monolithic-replica model of
DiLoCo-style federated training (§2). A node owns one or more zones, trains them, and submits its
zone outputs; the async gather (§3.3) would collect them under the same deadline discipline used inside a
single forward pass, and the deterministic Q16.16 merge (§3.4) **would** reconcile
independently-computed contributions into a result that every node — and an on-chain verifier —
computes identically. We use the conditional deliberately: the determinism *property* that makes
this sound (same gathered set → same bits) is demonstrated locally and TLC-checked
(`MergeDeterminism.tla`, §4.5), and the gather is implemented today as a single-process deterministic
*simulation* of the deadline; the actual multi-node, cross-network signed gather is the Gate-4
milestone (§7.5), not a result in hand. The swapped zone is also the **routing and adapter target**:
RM-FL's routing meta-model (precompile `0x0111`) emits a destination redefined to address a NAT zone,
and a LoRA registered through `LoRAFactory.verifyAdapterAt` (KZG via `0x0108`) targets that zone's
weights — so a federation specializes one zone without retraining the model (the unification's
binding #1; `nat-federated::seam::RoutingTarget`).

## 7.2 Paraconsistent aggregation: disagreement as data

Federated nodes have *different data distributions*, and averaging their contributions discards the
very signal that distinguishes a healthy decentralized network from a collapsed one. Classical
aggregation (mean, weighted average) projects two genuinely different answers onto a single point;
for personalized or domain-specialist zones that point is right for neither node. NAT adopts the
mechanism specified in Citrate Paper II: aggregate per embedding dimension in **Belnap's four-valued
logic** [Belnap 1977]. For each dimension, the contributions across nodes resolve to a state — **T**
(all agree positive), **F** (all agree negative), **B**oth (sources disagree: information is
contradictory), or **N**either (no source has spoken: we don't know) — and only the T/F states
collapse to a scalar mean over consenting sources; a B-state dimension is routed to *multiple*
downstream zones for cross-validation rather than averaged, and an N-state dimension is left
undefined rather than fabricated. A network that knows it does not know a dimension is more honest,
and more debuggable, than one that pretends to a fictional consensus. This is the precise sense in
which, for NAT on Citrate, *consensus and learning are the same process*: the same checkpoint that
finalizes blocks finalizes the Belnap-aggregated zone weights, and the same accountability that
slashes a Byzantine block-signer slashes a Byzantine weight-signer. The Belnap aggregation runs over
Q16 embeddings — the same fixed-point substrate as the merge and the verifiable-inference precompiles
— so it is bit-deterministic and on-chain-checkable. The substrate is the **Belnap aggregation
precompile `0x0110`** (deterministic Q16, saturating) — *not* the f32 `core/learning::belnap`
reference implementation, which is research-grade and not bit-reproducible across heterogeneous
validators; the unification (UNIFY-S1) points `0x0110` at NAT zone-weight deltas
(`nat-federated::seam::ZoneWeightDelta`, binding #3), and commits/aggregates via the existing
`0x0107/08/09` Poseidon/KZG/Merkle precompiles. *Status:* the aggregation logic and its on-chain
commitment are specified in Paper II and implemented in the `0x0110` precompile and the
`LearningOrchestrator` learning-cycle contracts; NAT supplies the model whose zone outputs are the
contributions, and the end-to-end multi-node federated training cycle (Gate 4) is future work, not a
demonstrated result.

## 7.3 The incentive seam

A decentralized network must reward contribution, and it must reward it *verifiably*. NAT does not
implement settlement; it emits the inputs and lets the audited economic layer settle them (ADR-0007).
Per training step (or gathered federated round) NAT produces a signed contribution — metered compute,
a data-quality score from the pipeline's quality stage, a token count, and the provenance-trace hash —
and a deterministic **reward weight = compute × data-quality**, computed on the Q16.16 path so two
nodes and an on-chain verifier agree on it bit for bit. `citrate-compute-pool` converts that weight to
payout under its tokenomics, and `ContributionAccounting` (Paper VII) records it. The unification
routes settlement through the co-op `FederatedSettlement` seam into the `PatronageLedger`, carrying
the `data_quality` term **separately** (not pre-collapsed into the reward weight) so the patronage
dividend can apply it as the honesty factor (`nat-federated::seam::SettlementRow`, binding #4;
conservation TLC-checked in `UnifiedSettlement.tla`). A node that
contributes compute on garbage data earns weight zero; the data-quality score is the economic signal,
which is why the data pipeline's quality stage (§5.3) is load-bearing rather than hygiene. The node
operator's path is deliberately narrow — pull verified, manifested, pre-tokenized shards; train; submit
signed outputs — the "grandma-proof" criterion: the hard ingestion and cleaning are centralized or run
by trusted operators, and the manifest hashes let a node verify it has the right data before training.

## 7.4 Decentralized intelligence meets decentralized science

The synthesis is the paper's widest claim. Decentralized-AI networks today incentivize *opaque*
contributions scored by trusted validators (§2); decentralized science seeks reproducible,
verifiable, community-owned research. NAT's provenance trace makes *every training and inference step
verifiable by construction*, on a substrate (Q16.16, on-chain commitment) the chain already runs.
That closes the gap: a contributor's work is not a black box a validator must trust, it is a signed,
replayable, provenance-traced artifact anyone can check. The corpus itself embodies the stance — a
license-fail-closed, public-domain "values spine" curated around the thesis that a rule has no meaning
without a community and a form of life (§5.3), assembled and documented in the open. This is what we
mean by decentralized intelligence meeting decentralized science on a verifiable chain: not a model
served *to* a community, but a model *trained by* one, whose every step answers to a public standard.

## 7.5 What is demonstrated versus specified

To be exact about the boundary: the local primitives this layer rests on are *demonstrated* — the
deterministic merge, the async gather, decision-faithful provenance, the trainable zones, the H-01
result, and the contribution/reward types are implemented and tested (§3–§6). The *network* layer —
multi-node signed gather across the federation, Belnap aggregation at checkpoint cadence, and the
end-to-end incentive settlement — is *specified* here and in Papers II/III/VII, and is the Gate-4
research milestone. We present §7 as the architecture's decentralization story and its falsifiable
next step, not as a result already in hand.
