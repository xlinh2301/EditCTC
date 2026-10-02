#!/bin/bash
#SBATCH --job-name=e44_eval
#SBATCH --partition=defq
#SBATCH --gres=mps:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/datastore/cndt_thangcpd/linhtruong/workspace5/slurm/logs/%x_%j.out

set -euo pipefail
WS=/datastore/cndt_thangcpd/linhtruong/workspace5
CODE="$WS/release_EditCTC/code"
PY="$CODE/.venv/bin/python"
CFG="$CODE/config/PP-OCRv6_small_rec_s1024_e44_highres_curated.yml"
CKPT="$WS/Data/EditCTC_arbor_runs/curated_runs/e44_highres_curated/best_accuracy"

REQUIRED_VRAM=1500 source "$WS/slurm/gpu_setup.sh"
cd "$CODE"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"

echo "=================================================================="
echo "1. Evaluating on FULL test set (586 samples, including new crops)"
echo "=================================================================="
"$PY" tools/eval.py -c "$CFG" \
  -o Global.checkpoints="$CKPT" \
     Global.pretrained_model=None \
     Eval.dataset.data_dir="$WS/Data/Indomain_curated/crops/test" \
     Eval.dataset.label_file_list="['$WS/Data/Indomain_curated/crops/test_label.txt']"

echo ""
echo "=================================================================="
echo "2. Evaluating on NEWLY ADDED test subset (24 samples)"
echo "=================================================================="
"$PY" tools/eval.py -c "$CFG" \
  -o Global.checkpoints="$CKPT" \
     Global.pretrained_model=None \
     Eval.dataset.data_dir="$WS/Data/Indomain_curated/crops/test" \
     Eval.dataset.label_file_list="['$WS/Data/Indomain_curated/crops/test_newly_added_label.txt']"

echo ""
echo "All evaluations completed successfully."
