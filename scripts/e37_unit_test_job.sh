#!/bin/bash
# E37 independent sequence verifier unit-test gate. No training is performed.
#SBATCH --job-name=e37_unit
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
"$PY" -m unittest -v tests/test_e37_verifier.py tests/test_e44_architecture.py
"$PY" -m py_compile ppocr/modeling/heads/sequence_verifier.py tools/program.py
echo "E37 verifier unit-test gate passed"
