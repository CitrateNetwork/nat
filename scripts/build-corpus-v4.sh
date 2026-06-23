#!/usr/bin/env bash
# Build corpus-v4 = the curated values-spine + code-craft + latex primaries (the
# v3 pillars) PLUS the bulk PD volume (scripts/fetch-corpus-volume.sh), then retrain
# the BPE-4096 tokenizer on the combined input. corpus-v4 is a strict SUPERSET of
# corpus-v3 — same sources, far more tokens — so the H-01 ladder can push past 2M
# params without overfitting. Recipe committed; data in gitignored ./corpus/.
#
#   scripts/fetch-corpus-volume.sh && scripts/build-corpus-v4.sh
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

OUT="${CORPUS_OUT:-$ROOT/corpus}"
WORK="$OUT/values-spine"
COMBINED="$WORK/corpus-v4-input.jsonl"

echo ">> building nat-corpus (release)"
cargo build --release -q -p nat-data --bin nat-corpus
BIN="$ROOT/target/release/nat-corpus"

# The v3 pillars. values-spine.jsonl already includes the CC0 seed (fetch-values-spine.sh).
VS="$WORK/values-spine.jsonl"
LATEX="$WORK/latex-primaries.jsonl"
CODE="$OUT/code-craft/code-craft.jsonl"
BULK="$WORK/bulk-gutenberg.jsonl"

[ -s "$VS" ]    || { echo "!! missing $VS — run scripts/fetch-values-spine.sh"; exit 1; }
[ -s "$BULK" ]  || { echo "!! missing $BULK — run scripts/fetch-corpus-volume.sh first"; exit 1; }
if [ ! -s "$CODE" ]; then
  echo ">> code-craft.jsonl absent — regenerating (scripts/fetch-code-craft.sh)"
  scripts/fetch-code-craft.sh
fi

echo ">> combining inputs -> $COMBINED"
: > "$COMBINED"
for f in "$VS" "$LATEX" "$CODE" "$BULK"; do
  if [ -s "$f" ]; then
    n=$(wc -l < "$f"); printf "   + %-28s %6s passages\n" "$(basename "$f")" "$n"
    cat "$f" >> "$COMBINED"
  fi
done
echo "   = $(wc -l < "$COMBINED") passages total (pre-pipeline)"

echo ">> running the pipeline -> corpus-v4"
"$BIN" run --input "$COMBINED" --out "$WORK/corpus-v4"

echo ">> retraining BPE-4096 on corpus-v4 input"
"$BIN" train-bpe --input "$COMBINED" --vocab 4096 --out "$WORK/bpe-4096-v4.json"

HASH=$(ls "$WORK/corpus-v4" | head -1)
echo ">> done."
echo "   corpus-v4 dir : $WORK/corpus-v4/$HASH"
echo "   manifest      : $(python3 -c "import json;m=json.load(open('$WORK/corpus-v4/$HASH/manifest.json'));print('docs',m['total_docs'],'tokens',m['total_tokens'])" 2>/dev/null)"
echo "   bpe-4096-v4   : $WORK/bpe-4096-v4.json"
echo ">> next: re-run the H-01 ladder on corpus-v4 at higher params (4M, 8M)."
