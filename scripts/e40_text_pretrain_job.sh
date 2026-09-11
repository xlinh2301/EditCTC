#!/usr/bin/env bash
#SBATCH --job-name=e40_nerd_text
#SBATCH --partition=defq
#SBATCH --gres=mps:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
set -euo pipefail

WS=/datastore/cndt_thangcpd/linhtruong/workspace5
CODE="$WS/release_EditCTC/code"
MANIFEST="$WS/Data/EditCTC_synth/visual_edit_v1/manifest.jsonl"
OUT="$WS/Data/EditCTC_arbor_runs/editctc-nerd-lcb/e40/text_mechanics/editrefine_pretrained"
PY="${EDITCTC_PYTHON:-$CODE/.venv/bin/python}"
REQUIRED_VRAM=1200 source "$WS/slurm/gpu_setup.sh"
mkdir -p "$(dirname "$OUT")"
while [[ ! -f "$WS/Data/EditCTC_synth/visual_edit_v1/summary.json" ]]; do
  sleep 30
done
cd "$CODE"
"$PY" tools/pretrain_editrefine_text.py \
  --manifest "$MANIFEST" \
  --out "$OUT" \
  --max-samples 200000 \
  --epochs 6 \
  --batch-size 256
