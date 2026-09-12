#!/bin/bash
# E37-B/C: independent sequence visual verifier. Cross-data is evaluation-only.
#SBATCH --job-name=e37_verifier
#SBATCH --partition=defq
#SBATCH --gres=mps:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/datastore/cndt_thangcpd/linhtruong/workspace5/slurm/logs/%x_%j.out

set -euo pipefail
WS=/datastore/cndt_thangcpd/linhtruong/workspace5
CODE="$WS/release_EditCTC/code"
PY="$CODE/.venv/bin/python"
CFG="$CODE/config/PP-OCRv6_small_rec_s1024_e44_highres.yml"
BASE="$WS/release_EditCTC/checkpoints/s1024/best_accuracy"
BANK="$WS/Data/EditCTC_arbor_runs/editctc-nerd-lcb/e37/counterfactual_bank.jsonl"
OUT="${E37_OUT:-$WS/Data/EditCTC_arbor_runs/editctc-nerd-lcb/e37/verifier}"
REQUIRED_VRAM=2500 source "$WS/slurm/gpu_setup.sh"
mkdir -p "$(dirname "$OUT")"
cd "$CODE"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
"$PY" tools/train_e37_verifier.py \
  --config "$CFG" --checkpoint "$BASE" --bank "$BANK" --out "$OUT" \
  --epochs "${E37_EPOCHS:-20}" --batch-size "${E37_BATCH_SIZE:-16}"
echo "E37 verifier training complete"
