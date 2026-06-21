#!/usr/bin/env bash
# Run the CI checks locally in a clean Linux container — the same commands the
# GitHub Actions `ci.yml` runs, on the pinned toolchain, so CI can be confirmed
# green without depending on GitHub-hosted runners (useful when Actions is
# blocked, e.g. a billing cap).
#
# Requires a Docker engine (Docker Desktop, or `colima start`). Uses the official
# rust:1.96 image, which matches rust-toolchain.toml.
#
# Usage: scripts/ci-local.sh
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
IMAGE="rust:1.96-bookworm"

echo ">> Pulling $IMAGE (first run only)"
docker pull -q "$IMAGE" >/dev/null

echo ">> Running the CI command set in a clean container"
docker run --rm -t \
  -v "$ROOT":/work -w /work \
  -e CARGO_TERM_COLOR=always \
  -e RUSTFLAGS="-D warnings" \
  -e CARGO_TARGET_DIR=/tmp/nat-target \
  "$IMAGE" bash -euo pipefail -c '
    echo "### toolchain"; rustc --version; cargo --version
    rustup component add rustfmt clippy >/dev/null 2>&1 || true
    echo "### [1/4] fmt --check";  cargo fmt --all -- --check
    echo "### [2/4] clippy";       cargo clippy --workspace --all-targets -- -D warnings
    echo "### [3/4] test";         cargo test --workspace
    echo "### [4/4] cargo-deny";   cargo install cargo-deny --locked >/dev/null 2>&1; cargo deny check advisories bans licenses sources
    echo "### ALL CI CHECKS PASSED (linux container)"
  '
