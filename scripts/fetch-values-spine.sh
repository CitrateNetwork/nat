#!/usr/bin/env bash
# Fetch the public-domain "values-spine" primaries (research-loop/READING_LIST.md)
# from Project Gutenberg, refine them through the nat-data pipeline, and build a
# training corpus. The recipe is committed; the data it produces lands in the
# gitignored ./corpus/. This is exactly what Hermes (HERMES-S1) automates.
#
#   scripts/fetch-values-spine.sh            # build ./corpus/values-spine
#   CORPUS_OUT=/data scripts/fetch-values-spine.sh
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

OUT="${CORPUS_OUT:-$ROOT/corpus}"
WORK="$OUT/values-spine"
JSONL="$WORK/values-spine.jsonl"
mkdir -p "$WORK/raw"

echo ">> building nat-corpus (release)"
cargo build --release -q -p nat-data --bin nat-corpus
BIN="$ROOT/target/release/nat-corpus"

# Curated PD primaries (Gutenberg id  short-name) — the four pillars.
BOOKS=(
  "5740  wittgenstein-tractatus"
  "15114 boole-laws-of-thought"
  "5827  russell-problems-of-philosophy"
  "41654 russell-intro-math-philosophy"
  "2529  russell-analysis-of-mind"
  "37134 strunk-elements-of-style"
  "1322  whitman-leaves-of-grass"
  "11    carroll-alice"
  "3600  montaigne-essays"
  "2945  emerson-essays-first-series"
  "11339 aesop-fables"
  "1342  austen-pride-and-prejudice"
)

# Fetch one book as text: try plain-text first, then strip HTML (needs python3),
# else skip (some logic books are PDF/TeX-only on Gutenberg — a from-pdf connector
# is DATA-S1 WP-D9). Ingests via `from-gutenberg`.
fetch_book() {
  local id="$1" name="$2" raw="$WORK/raw/$id.txt"
  if curl -sfL --max-time 40 "https://www.gutenberg.org/ebooks/$id.txt.utf-8" -o "$raw" \
     && [ -s "$raw" ]; then
    "$BIN" from-gutenberg --id "$id" --input "$raw" --out "$JSONL" --append --target-chars 2000
    return 0
  fi
  # Fallback: HTML → stripped text (for books with no plain-text format).
  local html
  html=$(curl -sL --max-time 20 "https://gutendex.com/books?ids=$id" \
    | python3 -c "import sys,json;f=json.load(sys.stdin)['results'][0]['formats'];print(f.get('text/html') or f.get('text/html; charset=utf-8') or '')" 2>/dev/null || true)
  if [ -n "$html" ]; then
    curl -sL --max-time 40 "$html" | python3 -c "
import sys,re,html
t=sys.stdin.read()
t=re.sub(r'(?is)<(script|style).*?</\1>',' ',t); t=re.sub(r'(?is)<[^>]+>',' ',t)
print(html.unescape(t))" | "$BIN" from-gutenberg --id "$id" --input - --out "$JSONL" --append --target-chars 2000
    return 0
  fi
  echo "   !! $id ($name): no plain-text or HTML on Gutenberg (PDF/TeX only) — skipping; see WP-D9"
}

: > "$JSONL"   # truncate
echo ">> fetching ${#BOOKS[@]} public-domain books"
for entry in "${BOOKS[@]}"; do
  id="${entry%% *}"; name="${entry##* }"
  fetch_book "$id" "$name" || true
done

# Add the authored CC0 seed (the values-spine explainers + the eval-battery domains).
"$BIN" emit-seed --out "$WORK/seed.jsonl" >/dev/null
cat "$WORK/seed.jsonl" >> "$JSONL"

echo ">> running the pipeline"
"$BIN" run --input "$JSONL" --out "$WORK/corpus"
echo ">> done. corpus under $WORK/corpus/"
