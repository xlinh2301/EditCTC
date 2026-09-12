#!/usr/bin/env bash
# Frozen CTC span index for targeted one-character online blur augmentation.
# Reads Indomain train only; Cross-data is never used here.
#SBATCH --job-name=e41_ctc_spans
#SBATCH --partition=defq
#SBATCH --gres=mps:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/datastore/cndt_thangcpd/linhtruong/workspace5/slurm/logs/%x_%j.out

set -euo pipefail
WS=/datastore/cndt_thangcpd/linhtruong/workspace5
CODE="$WS/release_EditCTC/code"
PY="$CODE/.venv/bin/python"
CFG="$CODE/config/PP-OCRv6_small_rec_s1024_uncertainty_random_50ep.yml"
CKPT="$WS/release_EditCTC/checkpoints/s1024/best_accuracy"
DATA="$WS/Data/Indomain/crops/train"
LABEL="$WS/Data/Indomain/crops/train_label.txt"
OUT="$WS/Data/EditCTC_arbor_runs/editctc-nerd-lcb/e41/ctc_span_index_train.json"

REQUIRED_VRAM=1200 source "$WS/slurm/gpu_setup.sh"
cd "$CODE"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
"$PY" tools/build_ctc_span_index.py \
  --config "$CFG" --checkpoint "$CKPT" --data-dir "$DATA" \
  --label-file "$LABEL" --output "$OUT" --batch-size 32
