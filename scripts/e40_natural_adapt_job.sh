#!/bin/bash
# E40-B: adapt the synthetic visual edit head on natural Indomain CTC seeds.
# Backbone and CTC remain frozen; Cross is never read.
#SBATCH --job-name=e40_natadapt
#SBATCH --partition=defq
#SBATCH --gres=mps:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:30:00
#SBATCH --output=/datastore/cndt_thangcpd/linhtruong/workspace5/slurm/logs/%x_%j.out

set -euo pipefail
WS=/datastore/cndt_thangcpd/linhtruong/workspace5
CODE="$WS/release_EditCTC/code"
PY="$CODE/.venv/bin/python"
CFG="$CODE/config/PP-OCRv6_small_rec_s1024_uncertainty_random_50ep.yml"
BASE="$WS/release_EditCTC/checkpoints/s1024/best_accuracy"
INIT="$WS/Data/EditCTC_arbor_runs/editctc-nerd-lcb/e40/visual_synthetic/editrefine_visual_pretrained"
BANK="$WS/Data/EditCTC_arbor_runs/editctc-nerd-lcb/e40/natural_ctc_bank"
OUT="$WS/Data/EditCTC_arbor_runs/editctc-nerd-lcb/e40/natural_adapt/editrefine_natural_adapted"

REQUIRED_VRAM=1200 source "$WS/slurm/gpu_setup.sh"
mkdir -p "$(dirname "$OUT")"
cd "$CODE"
"$PY" tools/pretrain_editrefine_visual.py \
  --config "$CFG" \
  --checkpoint "$BASE" \
  --init-edit-checkpoint "$INIT" \
  --manifest "$BANK/manifest.jsonl" \
  --out "$OUT" \
  --epochs 8 \
  --batch-size 128 \
  --lr 1e-4 \
  --positive-repeat 10
echo "natural adaptation complete: $OUT"
