#!/usr/bin/env bash
# Build / test / run the nat GPU path on the DGX (GB10, compute capability sm_121).
#
# Why this script exists — two toolchain facts that are not obvious:
#
#   1. candle 0.8.4 pins cudarc 0.13.9, whose build script HARD-REJECTS any CUDA
#      toolkit newer than 12.8. A box with only CUDA 13 cannot build the GPU path.
#      Fix: install the 12.8 toolkit side-by-side (does not touch the driver):
#          sudo apt-get install -y cuda-toolkit-12-8
#      It lands in /usr/local/cuda-12.8; the CUDA-13 driver stays in place.
#
#   2. nvcc 12.8 knows sm_120 but not sm_121 (sm_121 needs nvcc >= 12.9). So we
#      compile virtual `compute_120` PTX (CUDA_COMPUTE_CAP=120) and let the
#      CUDA-13 driver JIT it up to sm_121 at load. This runs on the GB10.
#
# The aarch64 +fp16 rustflag is supplied by .cargo/config.toml, not here.
#
# Usage:
#   scripts/dgx-gpu.sh build      # cargo build -p nat-candle --features cuda
#   scripts/dgx-gpu.sh test       # cargo test  -p nat-candle --features cuda
#   scripts/dgx-gpu.sh probe      # run examples/gpu_probe (asserts the GPU is live)
#   scripts/dgx-gpu.sh <args...>  # cargo <args...> with the CUDA 12.8 env applied
set -euo pipefail

CUDA="${NAT_CUDA_HOME:-/usr/local/cuda-12.8}"
if [[ ! -x "$CUDA/bin/nvcc" ]]; then
  echo "error: CUDA 12.8 toolkit not found at $CUDA" >&2
  echo "       install it with: sudo apt-get install -y cuda-toolkit-12-8" >&2
  echo "       (or set NAT_CUDA_HOME to a 12.x toolkit cudarc 0.13.9 accepts)" >&2
  exit 1
fi

export CUDA_PATH="$CUDA" CUDA_ROOT="$CUDA" CUDA_TOOLKIT_ROOT_DIR="$CUDA"
export PATH="$CUDA/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_COMPUTE_CAP=120

case "${1:-}" in
  build) exec cargo build -p nat-candle --features cuda ;;
  test)  exec cargo test  -p nat-candle --features cuda ;;
  probe) exec cargo run   -p nat-candle --features cuda --example gpu_probe ;;
  "")    echo "usage: scripts/dgx-gpu.sh {build|test|probe|<cargo args...>}" >&2; exit 2 ;;
  *)     exec cargo "$@" ;;
esac
