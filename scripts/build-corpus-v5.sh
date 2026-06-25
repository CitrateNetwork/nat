#!/usr/bin/env bash
# Build corpus-v5 = the curated v3/v4 pillars (values-spine + code-craft + latex
# primaries) PLUS a SCALED Project Gutenberg PD volume haul (scripts/fetch-corpus-volume.sh
# with a larger MAX_BOOKS), then train a BPE-16384 tokenizer on the combined input.
# corpus-v5 is a strict SUPERSET of corpus-v4 — same sources, more PD books — sized to
# unblock the 16M..64M H-01 ladder rungs (SCALE-S1 WP-S9; the 8M rung neared corpus-v4's
# ceiling). Recipe committed; data lands in gitignored ./corpus/.
#
#   MAX_BOOKS=1500 scripts/fetch-corpus-volume.sh && scripts/build-corpus-v5.sh
#
# The bigger BPE vocab (16384 vs v4's 4096) is for token EFFICIENCY at scale — fewer
# steps to cover hundreds of millions of tokens — and so partitioning governs a larger
# share of a bigger model's budget (turns H-01 from a core-only signal toward a
# whole-model one as d grows).
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

OUT="${CORPUS_OUT:-$ROOT/corpus}"
WORK="$OUT/values-spine"
COMBINED="$WORK/corpus-v5-input.jsonl"
VOCAB="${BPE_VOCAB:-16384}"

echo ">> building nat-corpus (release)"
cargo build --release -q -p nat-data --bin nat-corpus
BIN="$ROOT/target/release/nat-corpus"

# The v3/v4 pillars. values-spine.jsonl already includes the CC0 seed.
VS="$WORK/values-spine.jsonl"
LATEX="$WORK/latex-primaries.jsonl"
CODE="$OUT/code-craft/code-craft.jsonl"
BULK="$WORK/bulk-gutenberg.jsonl"

[ -s "$VS" ]   || { echo "!! missing $VS — run scripts/fetch-values-spine.sh"; exit 1; }
[ -s "$BULK" ] || { echo "!! missing $BULK — run scripts/fetch-corpus-volume.sh (MAX_BOOKS=1500) first"; exit 1; }
if [ ! -s "$CODE" ]; then
  echo ">> code-craft.jsonl absent — regenerating (scripts/fetch-code-craft.sh)"
  scripts/fetch-code-craft.sh
fi

echo ">> combining inputs -> $COMBINED"
: > "$COMBINED"
for f in "$VS" "$LATEX" "$CODE" "$BULK"; do
  if [ -s "$f" ]; then
    n=$(wc -l < "$f"); printf "   + %-28s %8s passages\n" "$(basename "$f")" "$n"
    cat "$f" >> "$COMBINED"
  fi
done
echo "   = $(wc -l < "$COMBINED") passages total (pre-pipeline)"

echo ">> running the pipeline -> corpus-v5"
"$BIN" run --input "$COMBINED" --out "$WORK/corpus-v5"

echo ">> training BPE-$VOCAB on corpus-v5 input"
"$BIN" train-bpe --input "$COMBINED" --vocab "$VOCAB" --out "$WORK/bpe-$VOCAB-v5.json"

HASH=$(ls "$WORK/corpus-v5" | head -1)
echo ">> done."
echo "   corpus-v5 dir : $WORK/corpus-v5/$HASH"
echo "   manifest      : $(python3 -c "import json;m=json.load(open('$WORK/corpus-v5/$HASH/manifest.json'));print('docs',m['total_docs'],'tokens',m['total_tokens'])" 2>/dev/null)"
echo "   bpe-$VOCAB-v5  : $WORK/bpe-$VOCAB-v5.json"
echo ">> next: H-01 ladder on corpus-v5 at 16M/32M/64M (SCALE-S1 WP-S10)."
