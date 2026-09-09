#!/usr/bin/env bash
# Slurm payload for scripts/arbor_eval_dev.sh. ARBOR_CODE is the worktree
# assigned to one Arbor node.
#SBATCH --job-name=arbor_editctc_dev
#SBATCH --partition=defq
#SBATCH --gres=mps:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00

set -euo pipefail

CODE="${ARBOR_CODE:?ARBOR_CODE is required}"
OUT_DIR="${ARBOR_OUT_DIR:?ARBOR_OUT_DIR is required}"
CHECKPOINT="${ARBOR_CHECKPOINT:?ARBOR_CHECKPOINT is required}"
WS=/datastore/cndt_thangcpd/linhtruong/workspace5
PY="${EDITCTC_PYTHON:-$WS/release_EditCTC/code/.venv/bin/python}"
CFG="${ARBOR_CONFIG:-$CODE/config/PP-OCRv6_small_rec_s1024_uncertainty_random_50ep.yml}"
IMAGE_DIR="$WS/Data/Indomain/crops/valid"
LABEL_FILE="$WS/Data/Indomain/crops/valid_label.txt"
INFER_LIST="$OUT_DIR/eval_list.txt"
PREDICTIONS="$OUT_DIR/predictions.txt"
BRANCH_LOG="$OUT_DIR/branch_audit.jsonl"
SUMMARY="$OUT_DIR/summary.json"
WRONG_CASES="$OUT_DIR/wrong_cases.jsonl"
CHANGED_CASES="$OUT_DIR/edit_changes.jsonl"
MISSING="$OUT_DIR/skipped_missing_images.txt"

REQUIRED_VRAM=1200 source "$WS/slurm/gpu_setup.sh"

for required in "$PY" "$CFG" "$CHECKPOINT.pdparams" "$IMAGE_DIR" "$LABEL_FILE"; do
    if [[ ! -e "$required" ]]; then
        echo "Missing required path: $required" >&2
        exit 1
    fi
done

mkdir -p "$OUT_DIR"
: > "$INFER_LIST"
: > "$MISSING"
while IFS=$'\t' read -r file gt; do
    [[ -z "$file" ]] && continue
    if [[ -f "$IMAGE_DIR/$file" ]]; then
        printf '%s\t%s\n' "$file" "$gt" >> "$INFER_LIST"
    else
        printf '%s\t%s\n' "$file" "$gt" >> "$MISSING"
    fi
done < "$LABEL_FILE"

cd "$CODE"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
"$PY" tools/infer_rec.py \
    -c "$CFG" \
    -o Global.checkpoints="$CHECKPOINT" \
       Global.pretrained_model=None \
       Global.infer_img="$IMAGE_DIR" \
       Global.infer_list="$INFER_LIST" \
       Global.save_res_path="$PREDICTIONS" \
       Global.branch_log_path="$BRANCH_LOG" \
       Architecture.Head.branch_debug=True

"$PY" - "$LABEL_FILE" "$BRANCH_LOG" "$SUMMARY" "$WRONG_CASES" "$CHANGED_CASES" "$MISSING" <<'PY'
import json
import sys
from pathlib import Path

label_path, branch_path, summary_path, wrong_path, changed_path, missing_path = sys.argv[1:]
labels = {}
for line in Path(label_path).read_text(encoding="utf-8").splitlines():
    if line.strip():
        name, text = line.split("\t", 1)
        labels[Path(name).name] = text

rows = [
    json.loads(line)
    for line in Path(branch_path).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
wrong = []
changed = []
op_counts = {name: 0 for name in ("KEEP", "REPLACE", "DELETE", "INSERT_AFTER")}
metrics = {
    "total_labeled": len(labels),
    "evaluated": 0,
    "skipped_missing_images": sum(
        bool(line.strip()) for line in Path(missing_path).read_text().splitlines()
    ),
    "ctc_correct": 0,
    "seed_correct": 0,
    "nrtr_correct": 0,
    "nerd_final_correct": 0,
    "lcb_length_correct": 0,
    "nerd_changed": 0,
    "nerd_helped": 0,
    "nerd_hurt": 0,
}

for row in rows:
    filename = Path(row["file"]).name
    gt = labels.get(filename)
    if gt is None:
        continue
    metrics["evaluated"] += 1
    ctc = row["ctc"]["text"]
    seed = row["nerd"]["seed_text"]
    final = row["final"]["text"]
    nrtr = row["nrtr"]["text"] if row.get("nrtr") else None
    lcb = row["lcb"]["predicted_length"] if row.get("lcb") else None
    changed_flag = seed != final or row["nerd"]["changed_positions"] > 0

    metrics["ctc_correct"] += ctc == gt
    metrics["seed_correct"] += seed == gt
    metrics["nrtr_correct"] += nrtr == gt
    metrics["nerd_final_correct"] += final == gt
    metrics["lcb_length_correct"] += lcb == len(gt)
    metrics["nerd_changed"] += changed_flag
    metrics["nerd_helped"] += seed != gt and final == gt
    metrics["nerd_hurt"] += seed == gt and final != gt

    for operation in row["nerd"]["operations"]:
        op_counts[operation["operation"]] += 1

    audit = {
        "ground_truth": gt,
        "filename": filename,
        "ctc_correct": ctc == gt,
        "seed_correct": seed == gt,
        "nrtr_correct": nrtr == gt,
        "nerd_final_correct": final == gt,
        "nerd_changed": changed_flag,
        "lcb_length_correct": lcb == len(gt),
        "branch_audit": row,
    }
    if final != gt:
        wrong.append(audit)
    if changed_flag:
        changed.append(audit)

denom = max(metrics["evaluated"], 1)
metrics.update({
    "ctc_accuracy": metrics["ctc_correct"] / denom,
    "seed_accuracy": metrics["seed_correct"] / denom,
    "nrtr_accuracy": metrics["nrtr_correct"] / denom,
    "nerd_final_accuracy": metrics["nerd_final_correct"] / denom,
    "lcb_length_accuracy": metrics["lcb_length_correct"] / denom,
    "wrong_cases": len(wrong),
    "operation_counts": op_counts,
    "files": {
        "branch_audit": str(branch_path),
        "wrong_cases": str(wrong_path),
        "edit_changes": str(changed_path),
    },
})

Path(wrong_path).write_text(
    "\n".join(json.dumps(x, ensure_ascii=False) for x in wrong)
    + ("\n" if wrong else ""),
    encoding="utf-8",
)
Path(changed_path).write_text(
    "\n".join(json.dumps(x, ensure_ascii=False) for x in changed)
    + ("\n" if changed else ""),
    encoding="utf-8",
)
Path(summary_path).write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(metrics, ensure_ascii=False))
PY
