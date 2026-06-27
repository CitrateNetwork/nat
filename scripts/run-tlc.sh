#!/usr/bin/env bash
# Model-check the three TLA+ modules with TLC (Gate-1 g1-formal). Needs a JRE.
# tla2tools.jar is fetched on first run to ./.tlc/ (gitignored) unless $TLA_TOOLS
# points at an existing jar.
#
#   scripts/run-tlc.sh
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
JAR="${TLA_TOOLS:-$ROOT/.tlc/tla2tools.jar}"

if ! command -v java >/dev/null 2>&1; then
  echo "error: java not found (install a JRE)"; exit 1
fi
if [ ! -f "$JAR" ]; then
  mkdir -p "$(dirname "$JAR")"
  echo ">> fetching tla2tools.jar"
  curl -fsSL -o "$JAR" \
    https://github.com/tlaplus/tlaplus/releases/latest/download/tla2tools.jar
fi

cd "$ROOT/formal"
fail=0
for m in MergeDeterminism AsyncGather McpHarness \
         GradientAggregation GradientAggregationAdversarial UnifiedSettlement; do
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
