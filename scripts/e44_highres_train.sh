#!/bin/bash
# E44-A: shared high-resolution visual memory, full controlled training.
# Cross-data is evaluation-only and is never read by this job.
#SBATCH --job-name=e44_highres
#SBATCH --partition=defq
#SBATCH --gres=mps:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/datastore/cndt_thangcpd/linhtruong/workspace5/slurm/logs/%x_%j.out

set -euo pipefail
WS=/datastore/cndt_thangcpd/linhtruong/workspace5
CODE="$WS/release_EditCTC/code"
PY="$CODE/.venv/bin/python"
CFG="$CODE/config/PP-OCRv6_small_rec_s1024_e44_highres.yml"
BASE="$WS/release_EditCTC/checkpoints/s1024/best_accuracy"
OUT="${E44_OUT:-$WS/Data/EditCTC_arbor_runs/editctc-nerd-lcb/e44/highres_shared}"
REQUIRED_VRAM=1200 source "$WS/slurm/gpu_setup.sh"
mkdir -p "$OUT"
cd "$CODE"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
"$PY" tools/train.py -c "$CFG" \
  -o Global.pretrained_model="$BASE" \
     Global.save_model_dir="$OUT" \
     Global.epoch_num=30 \
     Global.distributed=False \
     Train.loader.batch_size_per_card=64 \
     Train.sampler.first_bs=64 \
     Train.sampler.fix_bs=True
echo "E44-A high-resolution full training complete"
