#!/usr/bin/env bash
# corpus-v6 lever: a bounded English Wikipedia subset as RawDoc JSONL. Wikipedia is
# CC-BY-SA-4.0 (owner-approved; on the fail-closed ALLOWED_LICENSES list — SICP precedent).
# Source: HF `wikimedia/wikipedia` (already-cleaned plain text — no wikitext parsing).
# Streams to a CHARACTER budget so we pull a slice, not the full ~20GB dump.
#
#   WIKI_CHARS=900000000 scripts/fetch-wikipedia.sh   # ~900MB text ~ ~350M BPE tokens
#
# Output: corpus/values-spine/wikipedia.jsonl  (folded in by build-corpus-v6.sh)
set -uo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

WORK="${CORPUS_OUT:-$ROOT/corpus}/values-spine"
JSONL="$WORK/wikipedia.jsonl"
CHAR_BUDGET="${WIKI_CHARS:-900000000}"
MIN_CHARS="${WIKI_MIN_CHARS:-1000}"   # skip stubs/disambig
CONFIG="${WIKI_CONFIG:-20231101.en}"
mkdir -p "$WORK"

python3 - "$JSONL" "$CHAR_BUDGET" "$MIN_CHARS" "$CONFIG" <<'PY'
import sys, json
from datasets import load_dataset
out_path, budget, min_chars, config = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
ds = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)
total = n = 0
with open(out_path, "w") as f:
    for row in ds:
        text = row.get("text", "")
        if len(text) < min_chars:
            continue
        rec = {
            "id": f"wikipedia-{row['id']}",
            "source": "wikipedia",
            "license": "CC-BY-SA-4.0",
            "fetch_date": "2026-06-28",
            "text": text,
            "modality_refs": [],
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        total += len(text); n += 1
        if n % 10000 == 0:
            print(f"  .. {n} articles, {total/1e6:.0f}M chars", flush=True)
        if total >= budget:
            break
    f.flush()
print(f">> wikipedia: {n} articles, {total/1e6:.0f}M chars -> {out_path}", flush=True)
PY
echo ">> done. $(wc -l < "$JSONL" 2>/dev/null || echo 0) articles -> $JSONL"
