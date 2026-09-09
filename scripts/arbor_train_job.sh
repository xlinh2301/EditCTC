#!/usr/bin/env bash
# Slurm payload for scripts/arbor_train.sh. ARBOR_CODE is the worktree
# assigned to one Arbor node.
#SBATCH --job-name=arbor_editctc_train
#SBATCH --partition=defq
#SBATCH --gres=mps:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

CODE="${ARBOR_CODE:?ARBOR_CODE is required}"
OUT_DIR="${ARBOR_OUT_DIR:?ARBOR_OUT_DIR is required}"
CFG="${ARBOR_CONFIG:?ARBOR_CONFIG is required}"
WS=/datastore/cndt_thangcpd/linhtruong/workspace5
PY="${EDITCTC_PYTHON:-$WS/release_EditCTC/code/.venv/bin/python}"
PRETRAINED="$WS/workdir_text_rec/PPOCRv6/checkpoints/ppocrv6_small_rec_pretrained"
EDIT_PRETRAINED="$WS/workdir_text_rec/PPOCRv6/checkpoints_editrefine_textpretrain/editrefine_pretrained"
TRAIN_IMAGES="$WS/Data/Indomain/crops/train"
TRAIN_LABELS="$WS/Data/Indomain/crops/train_label.txt"
VALID_IMAGES="$WS/Data/Indomain/crops/valid"
VALID_LABELS="$WS/Data/Indomain/crops/valid_label.txt"

REQUIRED_VRAM=1200 source "$WS/slurm/gpu_setup.sh"
for required in "$PY" "$CFG" "$PRETRAINED" "$TRAIN_IMAGES" "$TRAIN_LABELS" "$VALID_IMAGES" "$VALID_LABELS"; do
    if [[ ! -e "$required" ]]; then
        echo "Missing required path: $required" >&2
        exit 1
    fi
done

mkdir -p "$OUT_DIR" "$OUT_DIR/../checkpoints"
cd "$CODE"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"

"$PY" tools/train.py -c "$CFG" \
    -o Global.save_model_dir="$OUT_DIR/../checkpoints" \
       Global.checkpoints=None \
       Global.pretrained_model="$PRETRAINED" \
       Global.edit_refine_pretrained="$EDIT_PRETRAINED" \
       Global.distributed=False \
       Train.dataset.data_dir="$TRAIN_IMAGES" \
       Train.dataset.label_file_list="[$TRAIN_LABELS]" \
       Eval.dataset.data_dir="$VALID_IMAGES" \
       Eval.dataset.label_file_list="[$VALID_LABELS]"
