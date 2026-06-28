#!/usr/bin/env bash
# Local CI for nat — runs the full gate set on the host, on the pinned toolchain.
#
# GitHub Actions is intentionally NOT used for nat right now (paid minutes are deferred
# until production; the workflow in .github/workflows/ci.yml is gated to manual dispatch).
# This script is the gate in the meantime: same checks a runner would do, plus the TLC
# formal suite, run locally where the dev box already has the CUDA toolchain (nat-candle)
# and authenticated git access to the private `citrate-fed-types` kernel.
#
# Usage:
#   scripts/ci-local.sh            # full gate: fmt, clippy, test, cargo-deny, TLC
#   scripts/ci-local.sh --fast     # skip the two slow gates (cargo-deny + TLC)
#   scripts/ci-local.sh --no-tlc   # skip only the TLC formal suite
#   scripts/ci-local.sh --no-deny  # skip only the supply-chain gate
set -uo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

# cargo must shell out to system git so the private `citrate-fed-types` kernel fetches
# with the developer's credentials / deploy key (libgit2 ignores the ssh config).
export CARGO_NET_GIT_FETCH_WITH_CLI=true
export CARGO_TERM_COLOR=always
# NB: do NOT export RUSTFLAGS="-D warnings" — a global RUSTFLAGS *replaces* (does not
# merge with) the `[target.aarch64] rustflags = +fp16` in .cargo/config.toml, which
# gemm-common (via candle) needs to assemble its fp16 kernels on the GB10 DGX. Warnings
# are denied via clippy's lint flag below (which covers every target), not via RUSTFLAGS.

run_tlc=1
run_deny=1
for arg in "$@"; do
  case "$arg" in
    --fast)    run_tlc=0; run_deny=0 ;;
    --no-tlc)  run_tlc=0 ;;
    --no-deny) run_deny=0 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

fail=0
stage() { printf '\n\033[1m>> %s\033[0m\n' "$1"; }
guard() { # guard <name> <cmd...>
  local name="$1"; shift
  local start; start=$(date +%s)
  if "$@"; then
    printf '   \033[32m✓ %s\033[0m (%ss)\n' "$name" "$(( $(date +%s) - start ))"
  else
    printf '   \033[31m✗ %s FAILED\033[0m\n' "$name"
    fail=1
  fi
}

stage "[1] rustfmt --check";  guard fmt    cargo fmt --all -- --check
stage "[2] clippy (-D warnings)"; guard clippy cargo clippy --workspace --all-targets -- -D warnings
stage "[3] test --workspace"; guard test   cargo test --workspace

if [[ "$run_deny" == 1 ]]; then
  stage "[4] cargo-deny (supply chain)"
  if command -v cargo-deny >/dev/null 2>&1; then
    guard cargo-deny cargo deny check advisories bans licenses sources
  else
    printf '   \033[33m• cargo-deny not installed — `cargo install cargo-deny --locked` (skipped)\033[0m\n'
  fi
fi

if [[ "$run_tlc" == 1 ]]; then
  stage "[5] TLC formal suite (nat/formal)"
  if command -v java >/dev/null 2>&1; then
    guard tlc bash scripts/run-tlc.sh
  else
    printf '   \033[33m• java not found — TLC formal gate skipped\033[0m\n'
  fi
fi

echo
if [[ "$fail" == 0 ]]; then
  printf '\033[32m=== ALL LOCAL CI CHECKS PASSED ===\033[0m\n'
else
  printf '\033[31m=== LOCAL CI FAILED ===\033[0m\n'
fi
exit "$fail"
