#!/bin/bash
# E44-A one-epoch training smoke test. No checkpoint is promoted.
#SBATCH --job-name=e44_smoke
#SBATCH --partition=defq
#SBATCH --gres=mps:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/datastore/cndt_thangcpd/linhtruong/workspace5/slurm/logs/%x_%j.out

set -euo pipefail
WS=/datastore/cndt_thangcpd/linhtruong/workspace5
CODE="$WS/release_EditCTC/code"
PY="$CODE/.venv/bin/python"
CFG="$CODE/config/PP-OCRv6_small_rec_s1024_e44_highres.yml"
BASE="$WS/release_EditCTC/checkpoints/s1024/best_accuracy"
OUT="$WS/Data/EditCTC_arbor_runs/editctc-nerd-lcb/e44/highres_smoke"
REQUIRED_VRAM=1200 source "$WS/slurm/gpu_setup.sh"
mkdir -p "$OUT"
cd "$CODE"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
"$PY" tools/train.py -c "$CFG" \
  -o Global.pretrained_model="$BASE" \
     Global.save_model_dir="$OUT" \
     Global.epoch_num=1 \
     Global.distributed=False \
     Train.loader.batch_size_per_card=8 \
     Train.sampler.first_bs=8 \
     Train.sampler.fix_bs=True
echo "E44-A high-resolution smoke complete"
