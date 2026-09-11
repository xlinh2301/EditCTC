#!/usr/bin/env bash
#SBATCH --job-name=e40_nerd_visual
#SBATCH --partition=defq
#SBATCH --gres=mps:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=08:00:00
set -euo pipefail

WS=/datastore/cndt_thangcpd/linhtruong/workspace5
CODE="$WS/release_EditCTC/code"
MANIFEST="$WS/Data/EditCTC_synth/visual_edit_v1/manifest.jsonl"
CFG="$CODE/config/PP-OCRv6_small_rec_s1024_uncertainty_random_50ep.yml"
CKPT="$WS/release_EditCTC/checkpoints/s1024/best_accuracy"
TEXT="$WS/Data/EditCTC_arbor_runs/editctc-nerd-lcb/e40/text_mechanics/editrefine_pretrained"
OUT="$WS/Data/EditCTC_arbor_runs/editctc-nerd-lcb/e40/visual_synthetic/editrefine_visual_pretrained"
PY="${EDITCTC_PYTHON:-$CODE/.venv/bin/python}"
REQUIRED_VRAM=1800 source "$WS/slurm/gpu_setup.sh"
mkdir -p "$(dirname "$OUT")"
while [[ ! -f "$WS/Data/EditCTC_synth/visual_edit_v1/summary.json" || ! -f "$TEXT.pdparams" ]]; do
  sleep 30
done
cd "$CODE"
"$PY" tools/pretrain_editrefine_visual.py \
  --config "$CFG" \
  --checkpoint "$CKPT" \
  --text-checkpoint "$TEXT" \
  --manifest "$MANIFEST" \
  --out "$OUT" \
  --max-samples 200000 \
  --epochs 5 \
  --batch-size 128
