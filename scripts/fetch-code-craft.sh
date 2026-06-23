#!/usr/bin/env bash
# Fetch the permissively-licensed "code & craft" sources (research-loop/READING_LIST.md
# pillar III — a good coder) and refine them through the nat-data pipeline. Grows the
# CX/code zone, which the latest INTENT flags as the corpus bottleneck.
#
# Two source kinds, both pre-approved (MIT/Apache, no review needed):
#   • The Rust Book (rust-lang/book) — markdown prose on the craft + idioms of code
#     ("the rules of the room" for a coder)         → `nat-corpus from-text`
#   • Idiomatic permissive crates (anyhow/itertools/serde) — real Rust source (CX
#     lexical signal)                                → `nat-corpus from-code`
#
# The recipe is committed; the data it produces lands in the gitignored ./corpus/.
# This is exactly the cycle Hermes (HERMES-S1, capsules corpus-fetch/normalize)
# automates against the daily INTENT.
#
#   scripts/fetch-code-craft.sh                       # build ./corpus/code-craft
#   CORPUS_OUT=/data scripts/fetch-code-craft.sh
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

OUT="${CORPUS_OUT:-$ROOT/corpus}"
WORK="$OUT/code-craft"
JSONL="$WORK/code-craft.jsonl"
RAW="$WORK/raw"
mkdir -p "$RAW"

echo ">> building nat-corpus (release)"
cargo build --release -q -p nat-data --bin nat-corpus
BIN="$ROOT/target/release/nat-corpus"

: > "$JSONL"

# 1) The Rust Book — dual MIT/Apache markdown. Concatenate src/**/*.md (sorted) and
#    ingest as text (no strip; markdown is prose the normalize handles).
echo ">> fetching the Rust Book (rust-lang/book, MIT/Apache)"
if [ ! -d "$RAW/rust-book" ]; then
  git clone --depth 1 -q https://github.com/rust-lang/book.git "$RAW/rust-book"
fi
find "$RAW/rust-book/src" -name '*.md' | sort | xargs cat \
  | "$BIN" from-text --input - --license MIT --source rust-lang/book \
      --id-prefix rust-book --out "$JSONL" --append --target-chars 2000

# 2) Idiomatic permissive crates — dual MIT/Apache. `from-code` walks each repo,
#    skips vendored dirs, and the SPDX id passes the fail-closed allow-list.
CRATES=(
  "dtolnay/anyhow"
  "rust-itertools/itertools"
  "serde-rs/serde"
)
echo ">> fetching ${#CRATES[@]} permissive crates"
for repo in "${CRATES[@]}"; do
  name="$(basename "$repo")"
  if [ ! -d "$RAW/$name" ]; then
    git clone --depth 1 -q "https://github.com/$repo.git" "$RAW/$name"
  fi
  # All three are "MIT OR Apache-2.0"; tag MIT (both are on the allow-list).
  "$BIN" from-code --dir "$RAW/$name" --license MIT --source "rust-lang/$name" \
    --out "$JSONL" --append --target-chars 2000 --max-line-len 400
done

echo ">> running the pipeline (code-craft only)"
"$BIN" run --input "$JSONL" --out "$WORK/corpus"

# To grow the FULL corpus, concatenate with the values-spine inputs and run once:
#   cat "$OUT/values-spine/values-spine.jsonl" \
#       "$OUT/values-spine/latex-primaries.jsonl" \
#       "$JSONL" > "$WORK/values-spine-plus-code.jsonl"
#   "$BIN" run --input "$WORK/values-spine-plus-code.jsonl" --out "$OUT/values-spine/corpus-v2"
echo ">> done. code-craft corpus under $WORK/corpus/"
