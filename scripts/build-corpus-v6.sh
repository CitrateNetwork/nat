#!/usr/bin/env bash
# Build corpus-v6 = the corpus-v5 pillars (values-spine + code-craft + latex primaries
# + the bulk Gutenberg haul) PLUS a Wikipedia slice (scripts/fetch-wikipedia.sh,
# CC-BY-SA-4.0), then train BPE-16384 on the combined input. corpus-v6 is a strict
# SUPERSET of corpus-v5, sized to feed the 64M H-01 rung (~640M tokens at ~10 tok/param)
# without overfitting — Wikipedia is the new volume lever (SCALE-S1 WP-S6/WP-S9).
#
#   WIKI_CHARS=900000000 scripts/fetch-wikipedia.sh && scripts/build-corpus-v6.sh
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

OUT="${CORPUS_OUT:-$ROOT/corpus}"
WORK="$OUT/values-spine"
COMBINED="$WORK/corpus-v6-input.jsonl"
VOCAB="${BPE_VOCAB:-16384}"

echo ">> building nat-corpus (release)"
cargo build --release -q -p nat-data --bin nat-corpus
BIN="$ROOT/target/release/nat-corpus"

VS="$WORK/values-spine.jsonl"
LATEX="$WORK/latex-primaries.jsonl"
CODE="$OUT/code-craft/code-craft.jsonl"
BULK="$WORK/bulk-gutenberg.jsonl"
WIKI="$WORK/wikipedia.jsonl"

[ -s "$VS" ]   || { echo "!! missing $VS — run scripts/fetch-values-spine.sh"; exit 1; }
[ -s "$WIKI" ] || { echo "!! missing $WIKI — run scripts/fetch-wikipedia.sh first"; exit 1; }
[ -s "$BULK" ] || { echo "!! missing $BULK — run scripts/fetch-corpus-volume.sh"; exit 1; }
[ -s "$CODE" ] || scripts/fetch-code-craft.sh

echo ">> combining inputs -> $COMBINED"
: > "$COMBINED"
for f in "$VS" "$LATEX" "$CODE" "$BULK" "$WIKI"; do
  if [ -s "$f" ]; then
    n=$(wc -l < "$f"); printf "   + %-28s %9s docs\n" "$(basename "$f")" "$n"
    cat "$f" >> "$COMBINED"
  fi
done
echo "   = $(wc -l < "$COMBINED") docs total (pre-pipeline)"

echo ">> running the pipeline -> corpus-v6"
"$BIN" run --input "$COMBINED" --out "$WORK/corpus-v6"

echo ">> training BPE-$VOCAB on corpus-v6 input"
"$BIN" train-bpe --input "$COMBINED" --vocab "$VOCAB" --out "$WORK/bpe-$VOCAB-v6.json"

HASH=$(ls "$WORK/corpus-v6" | head -1)
echo ">> done."
echo "   corpus-v6 dir : $WORK/corpus-v6/$HASH"
echo "   manifest      : $(python3 -c "import json;m=json.load(open('$WORK/corpus-v6/$HASH/manifest.json'));print('docs',m['total_docs'],'tokens',m['total_tokens'])" 2>/dev/null)"
echo "   bpe-$VOCAB-v6  : $WORK/bpe-$VOCAB-v6.json"
echo ">> next: H-01 64M rung on corpus-v6 (bf16)."
