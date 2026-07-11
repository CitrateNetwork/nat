#!/usr/bin/env bash
# Bulk public-domain VOLUME for the corpus (DATA-S1 follow-up; the H-01 ladder is
# data-limited at ~2M params / ~788K BPE tokens — more tokens is the next lever).
#
# Project Gutenberg PD is an owner-APPROVED source (research-loop/INTENT.md approval
# queue: "Approve: Project Gutenberg (PD)"). Rather than hand-list IDs, we page the
# Gutendex API for the most-downloaded ENGLISH, PUBLIC-DOMAIN (`copyright=false`)
# books and ingest their plain-text via `nat-corpus from-gutenberg`. The pipeline's
# fail-closed license gate + dedup (against the curated values-spine already in the
# corpus) do the rest. Recipe committed; data lands in gitignored ./corpus/.
#
#   scripts/fetch-corpus-volume.sh                 # ~200 popular PD books
#   MAX_BOOKS=400 scripts/fetch-corpus-volume.sh   # more volume
#
# Output: corpus/values-spine/bulk-gutenberg.jsonl  (combine + build in build-corpus-v4.sh)
set -uo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

OUT="${CORPUS_OUT:-$ROOT/corpus}"
WORK="$OUT/values-spine"
RAW="$WORK/raw-bulk"
JSONL="$WORK/bulk-gutenberg.jsonl"
MAX_BOOKS="${MAX_BOOKS:-200}"
MIN_BYTES="${MIN_BYTES:-20000}"   # skip stubs/indices
mkdir -p "$RAW"

echo ">> building nat-corpus (release)"
cargo build --release -q -p nat-data --bin nat-corpus
BIN="$ROOT/target/release/nat-corpus"

: > "$JSONL"
count=0
url="https://gutendex.com/books?languages=en&copyright=false&sort=popular"

while [ -n "$url" ] && [ "$url" != "null" ] && [ "$count" -lt "$MAX_BOOKS" ]; do
  # A 15k-book sweep is ~470 catalogue pages; one transient failure or rate-limit
  # must not end the run. Retry with exponential backoff, then politeness-pause
  # between pages so we don't trip Gutendex's limiter in the first place.
  page=""
  for attempt in 1 2 3 4 5 6; do
    page="$(curl -sfL --max-time 40 "$url" || true)"
    [ -n "$page" ] && break
    echo "   .. gutendex page fetch failed (attempt $attempt/6); backing off"
    sleep $((attempt * attempt * 2))
  done
  [ -z "$page" ] && { echo "   !! gutendex page fetch failed after 6 retries; stopping"; break; }

  # Emit "id<TAB>plain-text-url" for each PD book on this page that has a usable
  # .txt format (prefer utf-8; skip zip-only).
  rows="$(printf '%s' "$page" | python3 -c '
import sys, json
d = json.load(sys.stdin)
for b in d.get("results", []):
    f = b.get("formats", {})
    txt = (f.get("text/plain; charset=utf-8")
           or f.get("text/plain; charset=us-ascii")
           or f.get("text/plain"))
    if txt and not txt.endswith(".zip"):
        i = b["id"]
        print(f"{i}\t{txt}")
' 2>/dev/null || true)"

  while IFS=$'\t' read -r id txt; do
    [ -z "${id:-}" ] && continue
    [ "$count" -ge "$MAX_BOOKS" ] && break
    raw="$RAW/$id.txt"
    if [ ! -s "$raw" ]; then
      curl -sfL --max-time 40 "$txt" -o "$raw" || { echo "   !! $id: fetch failed"; rm -f "$raw"; continue; }
    fi
    bytes=$(wc -c < "$raw" 2>/dev/null || echo 0)
    if [ "$bytes" -lt "$MIN_BYTES" ]; then echo "   .. $id: ${bytes}B < ${MIN_BYTES} — skip"; continue; fi
    if "$BIN" from-gutenberg --id "$id" --input "$raw" --out "$JSONL" --append --target-chars 2000 >/dev/null 2>&1; then
      count=$((count + 1))
      [ $((count % 25)) -eq 0 ] && echo "   .. ingested $count books"
    else
      echo "   !! $id: from-gutenberg failed"
    fi
  done <<< "$rows"

  url="$(printf '%s' "$page" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("next") or "")' 2>/dev/null || true)"
  sleep 1
done

docs=$(wc -l < "$JSONL" 2>/dev/null || echo 0)
chars=$(wc -c < "$JSONL" 2>/dev/null || echo 0)
echo ">> done. $count books -> $docs passages, ${chars} bytes -> $JSONL"
echo ">> next: scripts/build-corpus-v4.sh  (combine with values-spine + code + latex, build + BPE-4096)"
