---
created: 2026-07-30
branch: feat/zone-merge-diagnostics
author: Claude (Opus 5), directed by @SaulBuilds
status: scope — not started; owner decision needed on rungs and on what to claim
  in the interim
relates:
  - .agentile/planset/decisions/ADR-0012-merge-floor-dead-zones.md
  - .agentile/planset/decisions/ADR-0005-baseline-protocol.md
  - .agentile/planset/hypotheses.md (H-01)
---

# H-01 re-run — scope

ADR-0012 found `PF` at **exactly zero merge share** in the shipped 64M
checkpoint. A zone with zero share gets zero gradient, so it had stopped
training. Every H-01 number was measured against a NAT arm in that condition,
with PF's parameters still counted in the budget.

This scopes what has to be re-run. It is a scope, not a result: **nothing has
been re-run.**

---

## 1. What is actually known, and what is not

**Known.** The final 64M checkpoint has PF at share `0.000000`, measured on GPU
over real corpus-v6 documents. `score_PF.weight` norm is 2.216 against 0.10–0.21
for the other four. With the floor at 0.01, PF returns to 0.002 and trains
(‖Δ‖₂ 1.93, in range with HP 1.78 and CX 1.56).

**NOT known, and this is the load-bearing gap: WHEN PF died.**

Only the final checkpoint is on disk (`meta.json`: `epochs_completed: 4`), so
there is no trajectory. If PF collapsed late in epoch 4, the H-01 numbers are
barely affected. If it collapsed in epoch 1, three quarters of the training ran
on four zones. **The two cases have very different implications and the evidence
cannot distinguish them.** That is the single strongest argument for re-running
rather than applying a correction.

**Also not known: which rungs were affected.** I probed for this rather than
assuming:

| probe | result |
|---|---|
| d=64, f32, floor=0, 30 steps | severe imbalance by step 5 (CX → 0.001), then **recovers** |
| d=64, f32, floor=0, 400 steps | **no collapse** — stabilises at SM .29 / CB .36 / HP .13 / PF .11 / CX .11 |
| d=64, bf16, floor=0, 400 steps | **no collapse** — same story |

So collapse is **not** universal, and **not** explained by dtype alone. At small
scale and short duration the merge self-balances after an early transient. The
64M run differs in width (1183 vs 64), vocab (16384 vs 256), tokens (306.5M vs
~512 windows) and duration (4 epochs vs 400 steps) — and I cannot say which of
those crosses the threshold.

**Consequence for the scope:** every rung must be *instrumented*, not predicted.
The cheap hope — "only the top rung was affected, patch the table" — is not
available.

---

## 2. What has to be re-run

Under ADR-0005 the arms must be param-matched and share a bit-identical embedding
and readout, so a re-run is a full ablation, not a NAT-side-only rerun.

| rung | why |
|---|---|
| every rung whose NAT arm was trained | share history is unknown at all of them |
| both arms | the dense arm is unaffected by the floor, but ADR-0005 requires the same protocol on both; reusing old dense numbers against new NAT numbers mixes protocols |

The dense arm is genuinely unchanged by ADR-0012 (it has no zones and no merge).
Reusing its numbers is *defensible* and would roughly halve the compute. It is
also exactly the kind of shortcut that makes a comparison quietly
non-comparable — different day, different corpus snapshot, different toolchain.
**Owner call. My recommendation: re-run both**, and if compute forces a choice,
re-run both at the top rung and reuse dense only at the lower ones, saying so in
the table.

---

## 3. The one thing that must change in the protocol

Every run records **per-zone merge shares at every checkpoint**, via
`AutoregLm::zone_merge_weights()`. Not as a diagnostic to consult when something
looks wrong — as a **first-class recorded output alongside the loss**.

This is the actual lesson of ADR-0012. The collapse was not subtle or rare; it
was *unobservable*. It sat in a shipped checkpoint through an entire ablation
ladder because nothing in the protocol asked which zones were training. A re-run
that produces a better number without closing that gap has fixed the symptom.

Concretely, the ablation harness should **fail the run** if any zone's share
falls below its floor for a sustained window — a dead zone invalidates the
comparison, so continuing to train is burning GPU on a number that cannot be
quoted.

---

## 4. Compute

I have not measured per-rung wall-clock and will not guess it. What is known:

- The 64M config from `h01-64m-corpus-v6-2026-07-07.log`: 1,000,000 train /
  250,000 val sequences, 8 epochs, batch 64, lr 0.003, seq_len 64, 2–3 seeds.
  That is ~15,625 steps/epoch, ~125,000 steps for a full 8-epoch arm, per seed.
- The shipped checkpoint reached `epochs_completed: 4`, i.e. the 64M arm was
  **half** the configured schedule.
- `bench_throughput` exists and should be run per rung, per dtype, to turn this
  into hours before any commitment is made.

**Action before scheduling: run `bench_throughput` at each intended rung and put
real hours in this document.** Scoping a GPU campaign on an unmeasured throughput
is how a two-day run becomes a two-week one.

### Hardware note that will bite

The GB10 is `sm_121`. CUDA **12.8**'s nvcc cannot target it; candle rejects CUDA
**13.0** outright (*"Unsupported cuda toolkit version"*). The working
combination is **12.8 with `CUDA_COMPUTE_CAP=120`**, relying on PTX
forward-compat. Any runner script or CI for this campaign needs that pinned, or
it will not build.

---

## 5. What to claim in the meantime

Until the re-run lands, H-01's status line in `hypotheses.md` should carry the
caveat explicitly. The current wording — "supported (real-data L1, 5/5 seeds)" —
reads as clean support and is not currently defensible at the upper rungs.

Suggested: keep the finding, add *"measured before ADR-0012; the NAT arm's PF
zone was inert in the 64M checkpoint and the collapse point is unknown. Lower
rungs unverified either way. Re-run pending."*

The result may well survive — a handicapped arm winning is a stronger result, not
a weaker one. But it has to survive by being re-measured, not by being restated.

---

## 6. Open decisions for the owner

1. **Which rungs.** All of them, or top-down until the gap stabilises?
2. **Re-run the dense arm, or reuse it?** (my recommendation: re-run; see §2)
3. **Seeds per rung.** Prior runs used 2–5. The floor changes early training
   dynamics, so seed variance may differ from before.
4. **Does a dead zone fail the run** (my recommendation: yes) or just get
   recorded?
5. **Interim wording for H-01** — §5 is a proposal, not a change I have made.
