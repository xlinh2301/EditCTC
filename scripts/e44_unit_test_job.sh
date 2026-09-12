#!/bin/bash
# E44-A architecture/unit-test gate. No training is performed here.
#SBATCH --job-name=e44_unit
#SBATCH --partition=defq
#SBATCH --gres=mps:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=/datastore/cndt_thangcpd/linhtruong/workspace5/slurm/logs/%x_%j.out

set -euo pipefail
WS=/datastore/cndt_thangcpd/linhtruong/workspace5
CODE="$WS/release_EditCTC/code"
PY="$CODE/.venv/bin/python"
REQUIRED_VRAM=1200 source "$WS/slurm/gpu_setup.sh"
cd "$CODE"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -m unittest -v tests/test_e44_architecture.py
"$PY" -m py_compile \
  ppocr/modeling/heads/rec_edit_refine_nrtr_head.py \
  ppocr/modeling/architectures/base_model.py
echo "E44-A unit-test gate passed"
