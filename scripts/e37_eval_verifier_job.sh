#!/bin/bash
# E37-G verifier evaluation. No Cross-data is read during training.
#SBATCH --job-name=e37_eval
#SBATCH --partition=defq
#SBATCH --gres=mps:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --output=/datastore/cndt_thangcpd/linhtruong/workspace5/slurm/logs/%x_%j.out

set -euo pipefail
WS=/datastore/cndt_thangcpd/linhtruong/workspace5
CODE="$WS/release_EditCTC/code"
PY="$CODE/.venv/bin/python"
CFG="$CODE/config/PP-OCRv6_small_rec_s1024_e44_highres.yml"
BASE="$WS/release_EditCTC/checkpoints/s1024/best_accuracy"
VERIFIER="$WS/Data/EditCTC_arbor_runs/editctc-nerd-lcb/e37/verifier"
: "${E37_IMAGE_DIR:?E37_IMAGE_DIR is required}"
: "${E37_LABEL_FILE:?E37_LABEL_FILE is required}"
OUT="${E37_EVAL_OUT:?E37_EVAL_OUT is required}"
REQUIRED_VRAM=2500 source "$WS/slurm/gpu_setup.sh"
mkdir -p "$(dirname "$OUT")"
cd "$CODE"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
"$PY" tools/eval_e37_verifier.py \
  --config "$CFG" --checkpoint "$BASE" --verifier "$VERIFIER" \
  --image-dir "$E37_IMAGE_DIR" --label-file "$E37_LABEL_FILE" --out "$OUT" \
  --thresholds "${E37_THRESHOLDS:-0,0.05,0.1,0.2,0.5}"
echo "E37 verifier evaluation complete"
