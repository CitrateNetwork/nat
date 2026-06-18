#!/usr/bin/env bash
# Install the repo's versioned git hooks into .git/hooks (which is not itself
# versioned). Run once after cloning. Idempotent.
set -euo pipefail
root="$(git rev-parse --show-toplevel)"
src="$root/scripts/hooks"
dst="$root/.git/hooks"

for hook in "$src"/*; do
  name="$(basename "$hook")"
  cp "$hook" "$dst/$name"
  chmod +x "$dst/$name"
  echo "installed: $name"
done
echo "hooks installed into .git/hooks"
