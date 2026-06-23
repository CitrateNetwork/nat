# NAT paper — adversarial red-team findings + remediation log

Four independent adversarial reviewers (empirical, citations/novelty, technical-vs-code,
hostile-venue) attacked the rough draft 2026-06-22. Consolidated below, with the remediation
applied to each. Severity: BLOCKER / MAJOR / MINOR. Status: ✅ remediated · �ðŸ”œ future-work
(needs an experiment we cannot run in prose).

## Findings that the paper UNDERSTATED (fix = strengthen to truth)

- **B-1 (BLOCKER, §4.5).** Paper says "TLC was not run … open Gate-1 item." **Stale:** TLC is
  green on all three modules (gates.yaml g1-formal `met: true`; commit d05ec61; `scripts/run-tlc.sh`;
  MergeDeterminism 31 states, AsyncGather 40, McpHarness 10, no violations). The open Gate-1 item
  is **counsel sign-off**, not TLC. ✅ Rewrite §4.5 + formal/README to TLC-green; remaining open
  item = counsel.
- **F2 (BLOCKER, §6.2).** Paper says H-02 generalization is "future work." **Contradicts canon:**
  hypotheses.md + gates.yaml + `nat-eval/tests/h02_heldout.rs` record a **held-out** result
  (trained router 3.10 vs L0 baseline 2.63 on unseen prompts — "it generalizes, not memorizes").
  ✅ Lead with the held-out 3.10 vs 2.63; keep 11.70 vs 4.25 as the in-sample upper read; stop
  calling generalization unshown.

## Findings that the paper OVERSTATED (fix = soften to truth)

- **F1/F7 (BLOCKER→MAJOR, §6.1 + abstract).** "partitioning **beats** dense" overstates the
  registered H-01, which is a **non-inferiority** hypothesis ("does not reduce"); and the per-seed
  verdict in code is `nat_cpp ≥ dense_cpp·0.95` (a 5% slack), not strict superiority. The loss
  ranges (2.88–2.91 vs 2.97–2.99) are non-overlapping and suggestive but N=5 with no variance/test.
  ✅ "does not reduce capability per parameter (and is modestly lower-loss on the mean)"; disclose
  the 5% non-inferiority slack; state N=5, no formal significance test; add per-seed framing.
- **F9 (MAJOR, abstract + §1.4).** Abstract presents federation/Belnap and GGUF/ONNX
  "runs in the existing ecosystem" as delivered; both are undemonstrated (gate4 pending; g3-gguf
  `met: false`). ✅ Mark the Belnap/federation layer **specified/formally-modeled, not demonstrated**;
  soften GGUF to a **design** (round-trip not yet shown); annotate the contribution list
  [demonstrated]/[specified]/[implemented-not-evaluated].
- **M-1 (MAJOR, §3.6).** "flattened-dense export" implied to exist; `ExportKind::FlattenedDense` is
  never constructed, no GGUF writer exists, g3-gguf `met: false`. ✅ Mark specified-not-implemented
  (WP-1.4); only `ZonePartitioned` is currently produced.
- **F3 (MAJOR, §6.3 + abstract).** "the architecture scales / monotone in size" from 3 rungs, and
  the L rung changes **two** variables (params 20.7K→115K **and** zones 3→5). ✅ "a 3-rung
  size/zone ladder trends downward (suggestive, not a scaling law)"; flag the 3→5 confound.
- **F4 (MAJOR, §6.3).** "3.42 bits/byte at 53K … better loss with half the params" compares a
  **different objective** (per-position autoregressive) to the single-output ladder. ✅ State it is
  a denser objective, not a controlled size comparison; drop the efficiency-win phrasing.

## Citation fixes (all 25 cites are REAL; these are precision/attribution)

- **C-1 (MAJOR).** `arXiv:2503.20679` is **not** a "modern CS survey" of multi-source paraconsistent
  reasoning — it is Jakl, *Four Imprints of Belnap's … Logic in CS* (d-frames, linear logic, blame
  calculus, LVars). ✅ Recharacterize accurately; cite SEP *Paraconsistent Logic* + Arieli & Avron
  (bilattice reasoning) for the multi-source claim; cite Belnap 1977 directly (Dunn & Epstein eds.,
  *Modern Uses of Multiple-Valued Logic*, Reidel 1977).
- **C-3 (MINOR).** The ~287s figure is **zkGPT's** (GPT-2, <25s, 279×/185× speedup), not zkLLM's
  (≈15 min, Llama-2-13B). ✅ Attribute precisely.
- **C-2 (MINOR).** ZKML/EuroSys 2024 lead authors are **Chen, Waiwitlikhit** (Kang is PI). ✅
  "Chen et al. 2024"; full title *ZKML: An Optimizing System for ML Inference in Zero-Knowledge Proofs*.
- **BriLLM:** differentiation is fair/accurate (verified via the paper) — ✅ add authors (Hai Zhao
  et al.) and cite the SiFu section, not the abstract.
- **C-4/C-5 (MINOR):** consistent arXiv-vs-publication-year convention; pin Belnap 1977 venue.

## Novelty reframe (the strategic fix)

- **N-1/N-2/N-3 (MAJOR).** "first to combine four things" is the weak form; the zone-partition
  conjunct overlaps modular DL [Pfeiffer 2023], fixed-routing MoE (Hash Layers, Roller et al.), and
  named-module cognitive architectures (ACT-R/Leabra). ✅ Demote "first to combine"; lead novelty on
  the **defensible primitive** — the *decision-faithful, deterministically-replayable, on-chain-
  committed provenance trace*, distinguished explicitly from model cards [Mitchell et al. 2019],
  logging, and post-hoc interp tooling; and from the *machine-checked* auditability of the fixed
  topology (the TLA+ specs). Label the Belnap-federation conjunct as a design contribution.

## Hostile-venue critiques → framing remediation (experiments are 🔜 future work)

- **Missing MoE baseline + component ablations** (router/prune/zones/random-vs-named partition) and
  a **standard corpus** (enwik8/text8) — the experiments that would make the causal claim land.
  ✅ Add as explicit, named limitations + the priority future experiments (§8/§9); 🔜 the runs
  themselves are DGX work.
- **Brain framing decorative** while in the title — ✅ keep "mimetic analog" discipline, add that
  demonstrating a neuro-partition beats a random equal-width partition is the test that would earn
  the analogy (the missing random-partition ablation); do not overclaim the neuro motivation.
- **§7 reads as crypto-marketing to an ML audience** — ✅ tighten present-tense over-reaches to
  Gate-4 future tense (m-6), keep the specified-vs-demonstrated split sharp; the section stays
  (this is also Gradient Paper XI) but every network-layer claim is explicitly undemonstrated.
- **Scope vs evidence** — ✅ the abstract no longer invokes frontier/regulated deployment as
  delivered; it motivates them and rests the empirical weight on the small-scale result honestly.

## What to PROTECT (do not gut during remediation)
The provenance schema + decision/bit-faithful split (§4.1–4.2); the fixed-topology bounded-router
invariant (§3.2, code-verified true); the ADR-0005 equal-param enforcement; the Q16.16 deterministic
merge; the honest-posture discipline; the GGUF-sidecar candor. The single-shared-`prune_and_reweight`
non-circularity and Q16.16 no-float-in-hash claims are **code-verified true** — keep as written.

## Minors remediated
F6 (state the 1/loss proxy plainly; at 0.08% param match it ≈ held-out loss); F8 (3/5 vs 5/5 not
statistically distinguishable at N=5 — drop "coin-flip→unanimous"); F10/§6.4 (label decision-faithful
as a design property, not an empirical result); F11 (corpus quality self-reported); m-1 (ADR-0008
3-zone staging caveat); m-2 ("same survivor decision; soft weights anneal τ→0").
