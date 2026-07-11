#!/usr/bin/env bash
# WP-S8 — durable corpus backup (SCALE-S1). The corpus is gitignored-local-only on
# the DGX; a host crash already cost a corpus-v4 rebuild. This syncs ./corpus/ to
# the 200GiB DO volume `nat-corpus-backup` mounted on citrate-alf-gateway
# (tailnet-only host, 100.93.165.75 — public SSH is closed on that droplet).
#
#   scripts/backup-corpus.sh              # additive sync (never deletes remote)
#   scripts/backup-corpus.sh --dry-run    # extra rsync flags pass through
#
# Restore is the same rsync reversed:
#   rsync -az root@100.93.165.75:/mnt/nat-corpus-backup/nat-corpus/ ./corpus/
#
# Deliberately NO --delete by default: the backup is provenance (raw is immutable,
# per 04_DATA_OPS "provenance is immutable"); prune by hand if ever needed.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
DEST="${CORPUS_BACKUP_DEST:-root@100.93.165.75:/mnt/nat-corpus-backup/nat-corpus/}"

rsync -az --info=stats1 "$@" "$ROOT/corpus/" "$DEST"
echo ">> backup synced to $DEST"
