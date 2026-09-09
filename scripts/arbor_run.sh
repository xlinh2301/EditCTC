#!/usr/bin/env bash
set -euo pipefail

CODE=$(git rev-parse --show-toplevel)
WORKTREE_ROOT=/datastore/cndt_thangcpd/linhtruong/workspace5/worktree
mkdir -p "$WORKTREE_ROOT"

# Native Arbor uses tempfile.gettempdir() for its executor worktrees. Setting
# TMPDIR before Python starts makes that path shared with Slurm compute nodes.
export ARBOR_WORKTREE_ROOT="$WORKTREE_ROOT"
export TMPDIR="$WORKTREE_ROOT"

exec "$CODE/.arbor-venv/bin/arbor" "$@"
