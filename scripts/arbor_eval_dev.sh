#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/arbor_eval_dev.sh --run-name NAME --node-id ID [--checkpoint PATH]

Runs the B_dev evaluator through Slurm from the current Git worktree.
EOF
}

RUN_NAME=""
NODE_ID=""
CHECKPOINT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-name) RUN_NAME="$2"; shift 2 ;;
        --node-id) NODE_ID="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$RUN_NAME" || -z "$NODE_ID" ]]; then
    echo "--run-name and --node-id are required" >&2
    usage >&2
    exit 2
fi

CODE=$(git rev-parse --show-toplevel)
WS=/datastore/cndt_thangcpd/linhtruong/workspace5
RUN_ROOT="$WS/Data/EditCTC_arbor_runs/$RUN_NAME/$NODE_ID"
OUT_DIR="$RUN_ROOT/eval_dev"
BASE_CHECKPOINT="$WS/release_EditCTC/checkpoints/s1024/best_accuracy"
NODE_CHECKPOINT="$RUN_ROOT/checkpoints/best_accuracy"

branch=$(git -C "$CODE" branch --show-current)
if [[ "$branch" == "main" || "$branch" == "master" ]]; then
    if [[ "${ARBOR_ALLOW_MAIN:-0}" != "1" && "$RUN_NAME" != baseline-* ]]; then
        echo "Refusing B_dev evaluation on protected branch '$branch'." >&2
        echo "Create an Arbor worktree, or set ARBOR_ALLOW_MAIN=1 for an explicit baseline." >&2
        exit 3
    fi
fi

if [[ -z "$CHECKPOINT" && -f "$NODE_CHECKPOINT.pdparams" ]]; then
    CHECKPOINT="$NODE_CHECKPOINT"
fi
if [[ -z "$CHECKPOINT" ]]; then
    CHECKPOINT="$BASE_CHECKPOINT"
fi
export ARBOR_CHECKPOINT="$CHECKPOINT"

mkdir -p "$OUT_DIR"
job_output="$OUT_DIR/slurm-%j.out"
job_error="$OUT_DIR/slurm-%j.err"

sbatch --wait --parsable \
    --output="$job_output" \
    --error="$job_error" \
    --export=ALL,ARBOR_CODE="$CODE",ARBOR_OUT_DIR="$OUT_DIR",ARBOR_CHECKPOINT="$ARBOR_CHECKPOINT" \
    "$CODE/scripts/arbor_eval_dev_job.sh" >/dev/null

SUMMARY="$OUT_DIR/summary.json"
if [[ ! -f "$SUMMARY" ]]; then
    echo "Missing evaluator summary: $SUMMARY" >&2
    exit 4
fi

"${PYTHON:-python3}" - "$SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print("score: {:.8f}".format(summary["nerd_final_accuracy"]))
print(json.dumps({
    key: summary[key]
    for key in (
        "evaluated", "ctc_accuracy", "nrtr_accuracy", "lcb_length_accuracy",
        "nerd_final_accuracy", "nerd_changed", "nerd_helped", "nerd_hurt",
    )
}, ensure_ascii=False))
PY
