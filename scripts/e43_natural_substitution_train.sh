#!/bin/bash
# E43-B: natural same-length substitution-only adaptation.
# Cross-data is evaluation-only and never enters this loader.
#SBATCH --job-name=e43_nat_sub
#SBATCH --partition=defq
#SBATCH --gres=mps:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/datastore/cndt_thangcpd/linhtruong/slurm/logs/%x_%j.out
set -euo pipefail
WS=/datastore/cndt_thangcpd/linhtruong/workspace5
CODE="$WS/release_EditCTC/code"
PY="$CODE/.venv/bin/python"
CFG="$CODE/config/PP-OCRv6_small_rec_s1024_token_refine.yml"
BASE="$WS/Data/EditCTC_arbor_runs/editctc-nerd-lcb/e42/token_refine_retry/best_accuracy"
BANK="$WS/Data/EditCTC_arbor_runs/editctc-nerd-lcb/e40/natural_ctc_bank/manifest.jsonl"
OUT="$WS/Data/EditCTC_arbor_runs/editctc-nerd-lcb/e43/natural_substitution/editor"
REQUIRED_VRAM=1200 source "$WS/slurm/gpu_setup.sh"
mkdir -p "$(dirname "$OUT")"
cd "$CODE"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
"$PY" tools/adapt_nrtr_token_natural.py --config "$CFG" --checkpoint "$BASE" \
  --manifest "$BANK" --out "$OUT" --epochs 12 --batch-size 64 \
  --positive-repeat 30 --same-length-only
