# nat

> Citrate Neuroarchitectural Transformer — a zone-partitioned, GGUF/ONNX-compatible transformer that emits an on-chain-verifiable provenance trace and trains in a federated cycle on Citrate.

## What it is

nat (RFC-CIT-NAT-0001) is a research transformer whose hidden representation is split into six named zones — Sensorimotor, Cerebellar, Hippocampal, Prefrontal, Codec, and an MCP harness — each running its own attention or state-space core over a fixed, learned-router-modulated topology, combined by an attention-scored noise-pruned merge. Every forward pass emits a structured, hashable trace of which zones fired and why, and all merge/reward math runs on Q16.16 fixed-point (never f32) so results are bit-reproducible across nodes.

It is an explicit research bet: the load-bearing question **H-01** is whether zone partitioning costs capability per parameter versus an equal-size dense baseline, tested cheaply up a scale ladder before an expensive ~10B run. Per training step nat emits a metered contribution that [citrate-compute-pool](https://github.com/CitrateNetwork/citrate-compute-pool) turns into a participant payout. This is a **public** repo (still BUSL-licensed — see below). Concept overview: https://docs.citrate.ai/research.

## Prerequisites

nat is a pure Rust / [Candle](https://github.com/huggingface/candle) project — **no Python**. The default build is CPU-only; the GPU path is opt-in.

```bash
# Rust 1.96.0 (pinned by rust-toolchain.toml — rustup auto-installs it)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# GPU path only (optional): NVIDIA driver + CUDA 12.8 TOOLKIT SPECIFICALLY (not 13 —
# candle 0.8's cudarc hard-rejects newer toolkits). Validated on DGX Spark GB10 (aarch64).
sudo apt-get install -y cuda-toolkit-12-8      # installs to /usr/local/cuda-12.8

# For scripts/ci-local.sh: Docker.
```

Building fetches a private git dependency (`citrate-fed-types`); `.cargo/config.toml` sets `git-fetch-with-cli = true` so system git (and your SSH access) is used.

## Build from source

```bash
git clone https://github.com/CitrateNetwork/nat.git
cd nat

cargo build --workspace          # CPU, no GPU required
cargo test  --workspace          # runs fully on CPU
cargo clippy --workspace --all-targets
```

The workspace has 15 crates under `crates/`. Release profile is `lto = true`, `codegen-units = 1`. Model artifacts and corpora are git-ignored (`*.safetensors`, `*.gguf`, `/corpus/`).

Local CI (org GitHub Actions not yet running):

```bash
scripts/ci-local.sh              # fmt + clippy + tests + cargo-deny in a rust:1.96 container (needs Docker)
```

## Run locally

CPU / illustrative (slow, but no GPU needed):

```bash
cargo run -p nat-ablation --example ablation        # H-01 ablation on synthetic data
cargo run -p nat-candle   --example train_corpus    # train a 3-zone byte-LM on the seed corpus
```

There is **no serving daemon** — "inference" is the forward-pass examples plus GGUF export (intended to run in Ollama once export lands). Build the corpus tool with `cargo build --release -p nat-data --bin nat-corpus`.

The GPU path is wrapped by `scripts/dgx-gpu.sh` (sets the CUDA 12.8 env + `CUDA_COMPUTE_CAP=120`):

```bash
scripts/dgx-gpu.sh build                             # cargo build -p nat-candle --features cuda
scripts/dgx-gpu.sh probe                             # asserts a live CUDA GPU
scripts/dgx-gpu.sh run -p nat-candle --features cuda --example scale_ladder -- <corpus-dir>
scripts/dgx-gpu.sh run -p nat-ablation --features cuda --example ablation      # the real H-01 bet
```

Verify a CPU build is healthy: `cargo test --workspace` passes with no GPU.

## Connect it locally

nat is the model + corpus layer of the Citrate stack; it does not settle rewards itself.

1. **Corpus** — build a deterministic, content-addressed corpus with the corpus scripts, then train against it:

   ```bash
   scripts/build-corpus-v6.sh                        # sized to feed the 64M H-01 rung
   scripts/dgx-gpu.sh run -p nat-candle --features cuda --example train_corpus
   ```

   A trained 64M checkpoint ships at `checkpoints-64m/nat-seed2/` for reference.

2. **Settlement (downstream)** — each training step emits `nat_train::StepContribution { compute_metered, data_quality, tokens, provenance_hash }` with `reward_weight = compute_metered × data_quality`. [citrate-compute-pool](https://github.com/CitrateNetwork/citrate-compute-pool) consumes that to compute payout on chain 40204. The interface is specified in `docs/SETTLEMENT_SEAM.md` (ADR-0007).

3. **Federation** — `nat-aggregate` (verifiable DiLoCo gradient aggregation, trimmed-mean in Q16), `nat-federated` (federated distillation), and `nat-weightspace` (weight-space commitment) implement the federated cycle. On-chain provenance verification and multi-node signed gather are Gate 4 (not done yet).

For the full multi-repo bring-up see `LOCAL_STACK.md` in [citrate-docs](https://github.com/CitrateNetwork/citrate-docs).

## Configuration

No `.env` file. Model configs are Rust constructors (`NatTrainConfig::byte_lm_3zone() / byte_lm_medium() / byte_lm_large()`), not YAML. The scale ladder rungs are S/M 3-zone and L 5-zone toward a ~10B L2 target (owner-gated).

| Variable | Default | Purpose |
|---|---|---|
| `NAT_CUDA_HOME` | `/usr/local/cuda-12.8` | override the CUDA toolkit path (GPU builds) |
| `CUDA_COMPUTE_CAP` | `120` (set by `dgx-gpu.sh`) | compile virtual `compute_120` PTX for GB10 |
| `WIKI_CHARS` / `CORPUS_OUT` / `BPE_VOCAB` | — | corpus-build script knobs |

`trace.backend` records the real device (`toy-l0` / `candle-cpu` / `candle-cuda`) in every provenance trace.

## Links

- Docs: https://docs.citrate.ai/research
- Depends on: `citrate-fed-types` (shared Q16 boundary kernel) · Consumed by: [citrate-compute-pool](https://github.com/CitrateNetwork/citrate-compute-pool) (reward settlement)
- Contributing (DCO): CONTRIBUTING.md · Security: SECURITY.md · License: LICENSE

## License

Source-available (BUSL-1.1) — free for personal/non-commercial use; commercial use requires a Citrate membership. This repo is **public**, but BUSL is **not** an open-source license.
