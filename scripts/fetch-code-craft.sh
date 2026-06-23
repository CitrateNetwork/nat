#!/usr/bin/env bash
# Fetch the permissively-licensed "code & craft" sources (research-loop/READING_LIST.md
# pillar III — a good coder) and refine them through the nat-data pipeline. Grows the
# CX/code zone, which the latest INTENT flags as the corpus bottleneck.
#
# Source kinds:
#   • The Rust Book (rust-lang/book, MIT/Apache) — markdown prose on the craft +
#     idioms of code ("the rules of the room")       → `nat-corpus from-text`
#   • Idiomatic permissive crates (anyhow/itertools/serde, MIT/Apache) — real Rust
#     source (CX lexical signal)                      → `nat-corpus from-code`
#   • SICP (sarabander/sicp, CC-BY-SA-4.0, owner-approved) — the canonical CS text
#     → tag-strip → `nat-corpus from-text`            (set SKIP_SICP=1 to omit)
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

# 3) SICP — Abelson & Sussman, CC-BY-SA-4.0 (owner-approved 2026-06-22). The book
#    HTML in sarabander/sicp is explicitly CC-BY-SA-4.0; strip tags → text. Set
#    SKIP_SICP=1 to omit (e.g. if a deployment wants permissive-only, no ShareAlike).
if [ "${SKIP_SICP:-0}" != "1" ]; then
  echo ">> fetching SICP (sarabander/sicp, CC-BY-SA-4.0)"
  if [ ! -d "$RAW/sicp" ]; then
    git clone --depth 1 -q https://github.com/sarabander/sicp.git "$RAW/sicp"
  fi
  find "$RAW/sicp/html" -name '*.xhtml' | sort | xargs cat | python3 -c "
import sys,re,html
t=sys.stdin.read()
t=re.sub(r'(?is)<(script|style).*?</\1>',' ',t)
t=re.sub(r'(?is)<[^>]+>',' ',t)        # drop all tags (incl MathML)
t=html.unescape(t); t=re.sub(r'[ \t]+',' ',t)
print(t)" | "$BIN" from-text --input - --license CC-BY-SA-4.0 --source sarabander/sicp \
      --id-prefix sicp --out "$JSONL" --append --target-chars 2000
fi

echo ">> running the pipeline (code-craft only)"
"$BIN" run --input "$JSONL" --out "$WORK/corpus"

# To grow the FULL corpus, concatenate with the values-spine inputs and run once:
#   cat "$OUT/values-spine/values-spine.jsonl" \
#       "$OUT/values-spine/latex-primaries.jsonl" \
#       "$JSONL" > "$WORK/values-spine-plus-code.jsonl"
#   "$BIN" run --input "$WORK/values-spine-plus-code.jsonl" --out "$OUT/values-spine/corpus-v3"
echo ">> done. code-craft corpus under $WORK/corpus/"
