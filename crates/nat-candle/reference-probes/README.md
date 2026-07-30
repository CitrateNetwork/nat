# Reference probes

Real `divergence_probe` output, committed so anyone measuring a new backend has
something to compare against without needing access to the machine that produced
it. These are measurements, not fixtures — nothing here was hand-written.

| file | machine | backend | dtype | tok/s | self-repeat |
|---|---|---|---|---|---|
| `gb10-cpu-f32.json` | GB10 (DGX Spark, aarch64) | `candle-cpu` | f32 | 7,786 | identical |
| `gb10-cuda-f32.json` | GB10, CUDA 12.8 / `CUDA_COMPUTE_CAP=120` | `candle-cuda` | f32 | 71,098 | identical |
| `m2max-cpu-f32.json` | Apple M2 Max 32 GB, macOS 15.6.1 | `candle-cpu` | f32 | 7,137 | identical |
| `m2max-metal-f32.json` | Apple M2 Max 32 GB, macOS 15.6.1 | `candle-metal` | f32 | 13,472 | identical |
| `m2max-metal-bf16.json` | Apple M2 Max 32 GB, macOS 15.6.1 | `candle-metal` | bf16 | 13,649 | **DIFFERS** |

The GB10 pair was produced 2026-07-30 on one physical machine, and the M2 Max
trio the same day on another. Two same-machine pairs plus a genuine cross-vendor
pair is what makes the set useful: the within-machine pairs isolate the
*backend*, and the across-machine pair says whether vendor matters on top of it.

**`m2max-metal-bf16.json` is the odd one out and is committed on purpose.** It
is the evidence that bf16 on Metal is *not* run-to-run deterministic — the probe
trains one job twice in-process and the two Q16 commitments disagree (reproduced
3/3 runs, while f32 on the same device is identical 5/5). Do not use it in a
divergence comparison; it is a record of a device that cannot reproduce itself,
which is a different and more serious property than diverging from someone else.
See "What bf16 on Metal costs" below.

## Comparing a new machine against them

```sh
cargo run --release -p nat-candle --features metal --example divergence_probe > probe-metal-f32.json

cargo run --release -p nat-candle --example divergence_compare -- \
  probe-metal-f32.json \
  crates/nat-candle/reference-probes/gb10-cuda-f32.json \
  crates/nat-candle/reference-probes/gb10-cpu-f32.json
```

**Compare f32 to f32.** Both references are f32, and a f32-vs-bf16 comparison
measures the dtype rather than the backend — which is a different (also
interesting) question, but not the one the settlement tolerance turns on.

`divergence_compare` refuses to compare probes whose `job` block differs, so a
probe from a modified build is detected rather than quietly producing a wrong
number.

## What these two already establish

| measure | CPU vs CUDA |
|---|---|
| loss after training | identical to full precision |
| max absolute weight delta | **1.19e-07** (~1 ULP of f32) |
| delta ÷ Q16 grid step | 0.0078 — **128× below** one step |
| zone shares | identical to 6 decimals |
| q16 commitments | **differ anyway** |

The last two rows together are the finding: divergence sits far below the grid
resolution and the commitments *still* disagree, because quantization boundaries
are knife-edges — a coordinate near a grid edge rounds either way under a 1-ULP
nudge, and ~0.8% of coordinates sit near one. So **exact commitment equality
cannot be the settlement primitive across heterogeneous hardware.** It has to be
a tolerance, and these two files are the only evidence we have for where to put
it.

They are also the most *favourable* pairing possible — one machine, one vendor's
float pipeline, two backends. Treat 1.19e-07 as a **lower bound** until a
genuinely different vendor (Metal, another CUDA architecture) reports in. That
measurement is the open question these files exist to help answer.

## Answered: a different vendor reported in

Measured 2026-07-30, M2 Max Metal against both GB10 references. This is the
cross-vendor number the lower bound was waiting on — different silicon vendor,
different machine, different OS, different float pipeline.

| pair | max abs delta | Δ ÷ Q16 step | Δ loss |
|---|---|---|---|
| **Metal ↔ CUDA** | **1.155e-07** | 0.0076 | exactly 0 |
| Metal ↔ GB10 CPU | 1.341e-07 | 0.0088 | exactly 0 |
| CUDA ↔ CPU (the old baseline) | 1.192e-07 | 0.0078 | exactly 0 |

**Crossing vendors did not make it worse.** Metal↔CUDA is the *tightest* pair of
the three, and the worst pair anywhere in the set is 1.341e-07 — 12.5% above the
single-machine lower bound, still **113.8× below one Q16 grid step**. Zone shares
agree to 6 decimals on all three machines, so no zone dies on one vendor's
hardware and not another's.

The lower bound held. `1e-6` remains a sound settlement tolerance: ~7.5× above
the worst honest divergence now measured, and still 4–5 orders of magnitude below
any real cheat.

What this does *not* cover: other CUDA architectures, other Apple chips, x86
CPUs, and dtypes other than f32. The set is two machines.

## The grid is coarse on purpose — do not "improve" it

The numbers above only work because the Q16 grid step (`2^-16` = 1.53e-05) is
**~114× coarser** than the worst honest divergence (1.34e-07). Quantization is
what absorbs cross-backend drift. That is a property of the *ratio*, so it is
lost by making the grid finer just as surely as by making the hardware noisier.

`Q16` is stored in an `i64` (widened from `i32`), which leaves spare bits. They
are for **range, not resolution**. The share of coordinates landing on opposite
sides of a grid boundary is roughly `divergence / grid_step`:

| fractional bits | grid step | straddling coordinates |
|---|---|---|
| 16 (today) | 1.53e-05 | ~0.9% |
| 32 (`Q32.32`) | 2.33e-10 | **all of them, by ~565 steps** |

At 32 fractional bits the grid is finer than the hardware's noise floor, so it
stops quantizing the disagreement away and starts faithfully recording it. Every
honest member's commitment would differ, by hundreds of raw units instead of one.

Raising precision does not help either. The divergence measured here is about one
ULP of f32 — the smallest disagreement the format can express — so there is no
sloppiness left to recover. f64 would shrink it to ~1e-16, but different
reduction orders still disagree at *any* precision, so a tolerance is required
regardless; and f64 is not available on Apple hardware at all (Metal has no f64
ALU — `mlx matmul doesn't support F64`), so it would exclude every Mac in the
co-op from accelerated work.

Worse, f64 would make exact equality *almost* work — straddling would drop to
roughly one coordinate in `1e11`, so commitments would match nearly every time
and then disagree unpredictably at scale. A primitive that fails rarely and
silently is more dangerous than one that fails immediately: today's "commitments
always differ" is honest, and forces the correct design straight away.

The invariant is pinned by a test in `citrate-fed-types`
(`the_grid_step_is_pinned_at_2_pow_minus_16`).

## What bf16 on Metal costs

f32 on Metal reproduces itself exactly; bf16 does not (3/3 runs differ). So a
bf16 Metal contribution **cannot be verified by recomputation** — not by another
machine, and not by the same machine a second later.

Pinning Apple hardware to f32 fixes that, and on this machine it is close to
free: 2,116 tok/s at f32 versus 2,143 at bf16 on the 8M-parameter bench, a 1.3%
difference. bf16 buys no measurable speed here and forfeits verifiability.

## Adding a probe

Send the JSON, and name it `<machine>-<backend>-<dtype>.json`. It carries no
personal data: hardware class, timings, and weights trained from synthetic
tokens generated from a fixed seed.
