#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/arbor_train.sh --config CONFIG --run-name NAME --node-id ID

Submits one isolated experiment training job to Slurm. The checkpoint is
written to Data/EditCTC_arbor_runs/NAME/ID/checkpoints/best_accuracy.
EOF
}

CONFIG=""
RUN_NAME=""
NODE_ID=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --run-name) RUN_NAME="$2"; shift 2 ;;
        --node-id) NODE_ID="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$CONFIG" || -z "$RUN_NAME" || -z "$NODE_ID" ]]; then
    echo "--config, --run-name, and --node-id are required" >&2
    usage >&2
    exit 2
fi

CODE=$(git rev-parse --show-toplevel)
WS=/datastore/cndt_thangcpd/linhtruong/workspace5
RUN_ROOT="$WS/Data/EditCTC_arbor_runs/$RUN_NAME/$NODE_ID"
OUT_DIR="$RUN_ROOT/train"

branch=$(git -C "$CODE" branch --show-current)
if [[ "$branch" == "main" || "$branch" == "master" ]]; then
    echo "Refusing training on protected branch '$branch'. Use an Arbor worktree." >&2
    exit 3
fi

if [[ "$CONFIG" != /* ]]; then
    CONFIG="$CODE/$CONFIG"
fi
mkdir -p "$OUT_DIR"

sbatch --wait --parsable \
    --output="$OUT_DIR/slurm-%j.out" \
    --error="$OUT_DIR/slurm-%j.err" \
    --export=ALL,ARBOR_CODE="$CODE",ARBOR_OUT_DIR="$OUT_DIR",ARBOR_CONFIG="$CONFIG" \
    "$CODE/scripts/arbor_train_job.sh" >/dev/null

CHECKPOINT="$RUN_ROOT/checkpoints/best_accuracy"
if [[ ! -f "$CHECKPOINT.pdparams" ]]; then
    echo "Training finished without checkpoint: $CHECKPOINT.pdparams" >&2
    exit 4
fi
echo "checkpoint: $CHECKPOINT"
