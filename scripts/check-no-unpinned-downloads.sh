#!/usr/bin/env bash
# Tripwire (NAT2-B-012): no script under scripts/ may fetch a binary from a mutable
# `releases/latest` pointer, and no script may pipe a remote fetch straight into a
# shell. tla2tools.jar (the Gate-1 formal-evidence toolchain) is the artifact that
# regressed here — it was fetched from `releases/latest` with no checksum and then
# executed with `java -cp`. run-tlc.sh now pins the tag and verifies a recorded
# SHA-256; this check keeps any script from reintroducing the class.
#
#   scripts/check-no-unpinned-downloads.sh
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

fail=0
self="scripts/check-no-unpinned-downloads.sh"

# Real download/exec lines only: a curl/wget (or pipe-to-shell) statement that is
# NOT a comment and NOT this checker's own documentation.
scan() {
  grep -REn "$1" scripts/ --include='*.sh' 2>/dev/null \
    | grep -vE ":[0-9]+:[[:space:]]*#" \
    | grep -v "^$self:" || true
}

# 1. No `releases/latest` (or a bare `/latest/download`) fetch of an artifact.
hits="$(scan '(curl|wget).*(releases/latest|/latest/download)')"
if [ -n "$hits" ]; then
  echo "error: unpinned 'latest' artifact download found (pin the release tag + verify a checksum):"
  echo "$hits"
  fail=1
fi

# 2. No `curl ... | sh|bash` (remote code executed unverified).
hits="$(scan 'curl[^|]*\|[[:space:]]*(sh|bash)([[:space:]]|$)')"
if [ -n "$hits" ]; then
  echo "error: 'curl | sh' pattern found (fetch, verify, then execute — never pipe to a shell):"
  echo "$hits"
  fail=1
fi

if [ "$fail" -eq 0 ]; then
  echo ">> no unpinned/unverified downloads under scripts/ (NAT2-B-012 clean)"
fi
exit $fail
