# §9 Conclusion and future work

We presented the Neuroarchitectural Transformer, a transformer whose hidden representation is
partitioned into declared, named zones over a fixed, auditable topology, merged on a deterministic
fixed-point path, and emitting a hashable provenance trace as a first-class output. The wager — that
*declaring structure* dissolves much of the opacity problem and pays out in verifiability,
capability per parameter, and decentralizability at once — is supported, at small scale, by what we
were able to measure: on a real, license-clean, public-domain corpus, zone partitioning did not
reduce capability per parameter versus an equal-parameter dense baseline and was modestly
lower-loss on the mean across all five seeds (held-out loss 2.88–2.91 versus 2.97–2.99, a
non-inferiority result); a learned router differentiated prompt classes and generalized to held-out
prompts (3.10 versus 2.63); held-out loss trended downward over a three-rung size/zone ladder; and
decision-faithful provenance holds by construction with the stateful surfaces TLC-checked. These are
small-scale results without a mixture-of-experts baseline or component ablation, stated with their
caveats, and the value of the scale ladder is precisely that
it let us test the load-bearing bet cheaply before committing to anything larger.

The work that remains is clearly ordered. The **most important next experiments** are the ones that
would turn the H-01 result from suggestive to causal: a **parameter-matched mixture-of-experts
baseline**, **component ablations** (no-router, no-prune, single-core-type, and — to earn the
neuroscience framing — a *random* equal-width partition versus the named one), and **results on a
standard corpus** (enwik8/text8/WikiText) with a per-seed table, variance, and a paired significance
test, ideally with one independent replication. **Scale** is the next test after that: BPE
tokenization, depth, and orders of magnitude more parameters and tokens, to see whether the hold
survives — and to say so honestly if it does not. The **federated proof** (Gate 4) turns §7 from
specified to demonstrated: multi-node signed gather, Belnap aggregation at checkpoint cadence, and
end-to-end incentive settlement through `citrate-compute-pool`. The **GGUF flattened-dense export**
(WP-1.4) builds the ecosystem onramp the sidecar currently only specifies. And a **task-level
capability metric**, available once the model is large enough to have one, replaces the inverse-loss
proxy. (The three TLA+ modules are already TLC-green; the remaining open Gate-1 item is counsel
sign-off on the claim-shaped statements.) Each of these is a falsification opportunity as much as a
milestone; the honest-posture discipline that carried the work this far is what makes them worth
running.

If the broader claim holds — that a model can be a dynamic, legible, verifiable instrument rather
than an undifferentiated blob of weights, with a broader capability range for less pretraining and a
verifiable record on every pass — then the contribution is not only an architecture but a stance:
that interpretability, efficiency, and decentralization are not three problems but three views of one
decision, *declare the structure*, and that the right place to settle distributed intelligence is a
network where consensus and learning are the same process. We have shown enough to think the stance
is worth taking seriously, and we have been explicit about everything we have not yet shown.

---

## Appendix pointers

- **A. Formal specifications.** `formal/{MergeDeterminism,AsyncGather,McpHarness}.tla` — the merge
  determinism, async-gather, and harness-safety invariants; claim-shaped statements C-1–C-5.
- **B. Reproducibility.** Config hashes, fixed seeds, and exact rerun commands per result; the
  containerized CI path (`scripts/ci-local.sh`); the corpus build (`scripts/fetch-values-spine.sh`)
  and the H-01 run (`scripts/dgx-gpu.sh … real_h01_corpus`). The reference implementation is a
  commit-pinned Rust workspace.
- **C. Companion case study.** *Agents Doing Science* — the human-set hypotheses, the agent-executed
  build, the honest-posture gates, and the bet's resolution from marginal-synthetic to
  decisive-real; the case studies CS-00 (forward pass), CS-01 (H-01), CS-02 (data and scaling).
