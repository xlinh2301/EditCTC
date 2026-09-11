#!/bin/bash
# Dump frozen CTC/NERD branch diagnostics on Indomain train images.  The
# resulting branch audit is converted into a natural-error/hard-KEEP manifest
# for E40-B edit-head adaptation; no Cross images are read here.
#SBATCH --job-name=e40_natbank
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
CKPT="$WS/release_EditCTC/checkpoints/s1024/best_accuracy"
IMG="$WS/Data/Indomain/crops/train"
LABEL="$WS/Data/Indomain/crops/train_label.txt"
OUT="$WS/Data/EditCTC_arbor_runs/editctc-nerd-lcb/e40/natural_ctc_bank"

REQUIRED_VRAM=1200 source "$WS/slurm/gpu_setup.sh"
mkdir -p "$OUT"
awk -F '\t' 'NF >= 2 {print $1}' "$LABEL" > "$OUT/infer_list.txt"
cd "$CODE"
"$PY" tools/infer_rec.py \
  -c "$CFG" \
  -o Global.checkpoints="$CKPT" \
     Global.pretrained_model=None \
     Global.edit_refine_pretrained=None \
     Global.infer_img="$IMG" \
     Global.infer_list="$OUT/infer_list.txt" \
     Global.save_res_path="$OUT/predictions.txt" \
     Global.branch_log_path="$OUT/branch_audit.jsonl" \
     Architecture.Head.branch_debug=True
echo "natural CTC bank dump complete: $OUT"
