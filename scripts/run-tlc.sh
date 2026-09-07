#!/usr/bin/env bash
# Model-check the TLA+ modules with TLC (Gate-1 g1-formal). Needs a JRE.
# tla2tools.jar is fetched on first run to ./.tlc/ (gitignored) unless $TLA_TOOLS
# points at an existing jar.
#
#   scripts/run-tlc.sh
#
# Supply-chain (NAT2-B-012): the jar is fetched from a PINNED release tag (never
# `releases/latest`, a mutable pointer) and its SHA-256 is VERIFIED against the
# reviewed digest recorded in scripts/tla2tools.jar.sha256 before it is executed
# with `java -cp`. A changed or compromised upstream artifact — the input that
# generates the Gate-1 formal evidence and runs as the developer — fails closed
# here, honoring deny.toml's "no unverified binary" posture.
#
# Recording/rotating the pin (a reviewed, one-time action, like accepting a
# cargo-deny advisory):
#   1. edit TLA_VERSION below,
#   2. fetch the jar for that tag,
#   3. `sha256sum` it and write the 64-hex digest to scripts/tla2tools.jar.sha256.
# Absent that pin file (or on a mismatch) the script REFUSES to run TLC — set
# TLA_ALLOW_UNVERIFIED=1 only for a deliberate, local, throwaway run.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
JAR="${TLA_TOOLS:-$ROOT/.tlc/tla2tools.jar}"

# Pinned tla2tools release. Bumping this is a reviewed edit paired with a refreshed
# scripts/tla2tools.jar.sha256.
TLA_VERSION="v1.8.0"
SHA_PIN="$ROOT/scripts/tla2tools.jar.sha256"

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    echo "error: no sha256sum/shasum available to verify $1" >&2; return 1
  fi
}

if ! command -v java >/dev/null 2>&1; then
  echo "error: java not found (install a JRE)"; exit 1
fi
if [ ! -f "$JAR" ]; then
  mkdir -p "$(dirname "$JAR")"
  echo ">> fetching tla2tools.jar ($TLA_VERSION, pinned)"
  curl -fsSL -o "$JAR" \
    "https://github.com/tlaplus/tlaplus/releases/download/${TLA_VERSION}/tla2tools.jar"
fi

# Verify EVERY run (a pre-existing or $TLA_TOOLS-provided jar is untrusted too).
if [ -f "$SHA_PIN" ]; then
  expected="$(tr -d '[:space:]' < "$SHA_PIN" | cut -d' ' -f1)"
  actual="$(sha256_of "$JAR")"
  if [ "$actual" != "$expected" ]; then
    echo "error: tla2tools.jar checksum mismatch (NAT2-B-012)"
    echo "  expected $expected"
    echo "  actual   $actual"
    echo "  refusing to execute an unverified tla2tools.jar"
    exit 1
  fi
  echo ">> tla2tools.jar SHA-256 verified against scripts/tla2tools.jar.sha256"
elif [ "${TLA_ALLOW_UNVERIFIED:-0}" = "1" ]; then
  echo ">> WARNING: scripts/tla2tools.jar.sha256 absent; running UNVERIFIED (TLA_ALLOW_UNVERIFIED=1)"
else
  echo "error: scripts/tla2tools.jar.sha256 is missing — cannot verify tla2tools.jar (NAT2-B-012)"
  echo "  record the reviewed digest:  sha256sum '$JAR' | cut -d' ' -f1 > scripts/tla2tools.jar.sha256"
  echo "  or set TLA_ALLOW_UNVERIFIED=1 for a deliberate throwaway local run"
  exit 1
fi

cd "$ROOT/formal"
fail=0
for m in MergeDeterminism AsyncGather McpHarness \
         GradientAggregation GradientAggregationAdversarial UnifiedSettlement \
         WeightCommitment LoraRegistration; do
  log="/tmp/tlc-$m.log"
  if java -cp "$JAR" tlc2.TLC -metadir "/tmp/tlc-$m" -config "$m.cfg" "$m.tla" \
       >"$log" 2>&1 && grep -q 'No error has been found' "$log"; then
    states=$(grep -oE '[0-9]+ distinct states found' "$log" | tail -1)
    echo ">> $m: GREEN ($states)"
  else
    echo ">> $m: FAIL"; tail -25 "$log"; fail=1
  fi
done
exit $fail
