---
created: 2026-07-30
branch: feat/zone-merge-diagnostics
author: Claude (Opus 5), directed by @SaulBuilds
status: scope — not started; compute measured (§4); owner decision needed on
  rungs, token budget, and what to claim in the interim
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

## 4. Compute — measured

Benched on the GB10 with `bench_throughput`, batch 64, seq 64, vocab 16384
(corpus-v6's BPE), 20 steps after warm-up.

The H-01 schedule from `h01-64m-corpus-v6-2026-07-07.log` is **1,000,000 train
sequences x 8 epochs x seq_len 64 = 512,000,000 tokens per arm per seed.**

| rung | params | d | bf16 tok/s | f32 tok/s | **bf16 h** | f32 h | bf16 speedup |
|------|--------|---|-----------|----------|-----------|-------|--------------|
| 248K | 279,871 | 8 | 23,859 | 21,025 | **6.0** | 6.8 | 1.13x |
| 1M | 1,016,321 | 30 | 22,945 | 20,652 | **6.2** | 6.9 | 1.11x |
| 2M | 2,013,718 | 59 | 22,149 | 18,337 | **6.4** | 7.8 | 1.21x |
| 4M | 4,025,406 | 115 | 20,160 | 14,447 | **7.1** | 9.8 | 1.40x |
| 8M | 8,020,261 | 218 | 16,147 | 8,626 | **8.8** | 16.5 | 1.87x |
| 32M | 32,022,343 | 704 | 5,323 | 2,976 | **26.7** | 47.8 | 1.79x |
| 64M | 64,074,343 | 1,184 | 2,963 | 1,683 | **48.0** | 84.5 | 1.76x |
| | | | | **sum** | **109.2** | 180.0 | |

Hours are **per arm, per seed.**

**The dense arm was measured, not assumed** — `bench_throughput` gained a
`NAT_ARM=dense` mode so that doubling for two arms rests on a measurement. Two
independent runs at each rung:

| rung | run | NAT tok/s | dense tok/s |
|------|-----|-----------|-------------|
| 8M | 1 | 16,147 | 15,449 |
| 8M | 2 | 15,341 | 15,546 |
| 64M | 1 | 2,963 | 2,929 |
| 64M | 2 | 2,948 | 2,908 |

The arms are **indistinguishable**: the spread between them (≤4%) is smaller
than the spread between two runs of the *same* arm (8M NAT: 16,147 vs 15,341,
5%). Run 1 has dense slower at 8M, run 2 has it faster — that is noise, not a
difference. Param counts match the equal-param protocol (63,779,401 vs
64,074,343). So the ×2 for two arms is sound, and every figure in this section
should be read with a **±5% run-to-run band**, which does not change any of the
scheduling conclusions below.

### Campaign totals

| plan | bf16 | f32 |
|------|------|-----|
| full ladder, 2 arms, 2 seeds | **437 h — 18.2 days** | 720 h — 30.0 days |
| full ladder, 2 arms, 3 seeds | **655 h — 27.3 days** | 1,080 h — 45.0 days |
| 64M only, 2 arms, 3 seeds | **288 h — 12.0 days** | — |
| 32M only, 2 arms, 3 seeds | **160 h — 6.7 days** | — |
| 248K–8M, 2 arms, 3 seeds | **207 h — 8.6 days** | — |

**Use bf16.** The speedup is 1.1x at the bottom and ~1.8x from 8M up, which is
the difference between an 18-day campaign and a 30-day one. This also answers the
question `bench_throughput`'s own docs left open ("quantify the bf16 speedup once
WP-S2 lands").

### The finding that should change the plan

**The lower rungs are not cheap.** 248K–8M costs 8.6 days at 3 seeds — most of
what 64M alone costs (12 days) — because the schedule is a **fixed 512M tokens
regardless of model size.** Throughput is flat below ~4M params (22–24k tok/s):
those runs are overhead-bound, not compute-bound, so a smaller model does not buy
a shorter run.

Put in tokens-per-parameter, the fixed schedule means:

| rung | tok/param |
|------|-----------|
| 248K | ~1,830 |
| 64M | ~8 |

Compute-optimal is roughly 20. So the small rungs are ~90x **over**-trained and
the top rung is ~2.5x **under**-trained, and the ladder is not comparing runs of
equivalent training adequacy.

That may well be deliberate — holding the data budget constant isolates the
parameter variable, which is what ADR-0005 is about. But it is worth an explicit
decision rather than an inherited default, because scaling the token budget with
model size would cut the lower rungs to under an hour each and let the saved days
go into seeds at 32M/64M, where the answer actually lives.

**Recommendation:** bf16; 64M and 32M first (18.7 days at 3 seeds) since that is
where collapse was observed and where the H-01 claim is weakest; then decide on
the lower rungs once their share traces are in hand from the instrumented run.

### One caveat on the hours above

They are for the **full configured 8-epoch schedule**. The shipped 64M checkpoint
reached `epochs_completed: 4` — half of it. So the prior 64M arm cost ~24 h, not
48 h, and a re-run that actually finishes the configured schedule costs **twice
what the original did**, before counting the dense arm or extra seeds. If the
intent is to reproduce what was measured rather than to run the protocol as
written, halve the 32M and 64M rows — but then say so, because "8 epochs" in the
protocol and "4 epochs" on disk are already out of agreement.

### Hardware note that will bite

The GB10 is `sm_121`. CUDA **12.8**'s nvcc cannot target it; candle rejects CUDA
**13.0** outright (*"Unsupported cuda toolkit version"*). The working combination
is **12.8 with `CUDA_COMPUTE_CAP=120`**, relying on PTX forward-compat. Any
runner script or CI for this campaign needs that pinned, or it will not build.
Every number in this section was measured under exactly that configuration.

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
6. **Fixed 512M tokens at every rung, or scale the budget with model size?**
   New — this only became visible once the hours were measured (§4). The fixed
   schedule makes the small rungs cost nearly as much as the large ones while
   over-training them ~90x.
7. **8 epochs or 4?** The protocol says 8; the shipped checkpoint has 4. Pick
   one and make the log and the protocol agree.
