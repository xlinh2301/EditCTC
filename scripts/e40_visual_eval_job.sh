#!/usr/bin/env bash
#SBATCH --job-name=e40_nerd_veval
#SBATCH --partition=defq
#SBATCH --gres=mps:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
set -euo pipefail
WS=/datastore/cndt_thangcpd/linhtruong/workspace5
CODE="$WS/release_EditCTC/code"
PY="${EDITCTC_PYTHON:-$CODE/.venv/bin/python}"
CKPT="$WS/Data/EditCTC_arbor_runs/editctc-nerd-lcb/e40/visual_synthetic/editrefine_visual_pretrained"
MANIFEST="$WS/Data/EditCTC_synth/visual_edit_v1/manifest.jsonl"
CFG="$CODE/config/PP-OCRv6_small_rec_s1024_uncertainty_random_50ep.yml"
OUT="$WS/Data/EditCTC_arbor_runs/editctc-nerd-lcb/e40/visual_synthetic/metrics.json"
REQUIRED_VRAM=1800 source "$WS/slurm/gpu_setup.sh"
cd "$CODE"
"$PY" tools/eval_editrefine_visual.py --config "$CFG" --checkpoint "$CKPT" --manifest "$MANIFEST" --out "$OUT" --max-samples 4000 --batch-size 128
