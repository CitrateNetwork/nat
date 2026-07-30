# Reference probes

Real `divergence_probe` output, committed so anyone measuring a new backend has
something to compare against without needing access to the machine that produced
it. These are measurements, not fixtures — nothing here was hand-written.

| file | machine | backend | dtype | tok/s | self-repeat |
|---|---|---|---|---|---|
| `gb10-cpu-f32.json` | GB10 (DGX Spark, aarch64) | `candle-cpu` | f32 | 7,786 | identical |
| `gb10-cuda-f32.json` | GB10, CUDA 12.8 / `CUDA_COMPUTE_CAP=120` | `candle-cuda` | f32 | 71,098 | identical |

Produced 2026-07-30 on the same physical machine, which is what makes the pair
useful: it isolates the *backend* from every other variable.

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

## Adding a probe

Send the JSON, and name it `<machine>-<backend>-<dtype>.json`. It carries no
personal data: hardware class, timings, and weights trained from synthetic
tokens generated from a fixed seed.
