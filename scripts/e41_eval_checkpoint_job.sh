#!/bin/bash
# Evaluate one E41 checkpoint on a read-only labeled split.
# EVAL_IMAGE_DIR, EVAL_LABEL_FILE, EVAL_OUT_DIR and EVAL_CHECKPOINT are exported by sbatch.
# Cross-data is accepted only here for evaluation; this script never trains.
#SBATCH --job-name=e41_eval
#SBATCH --partition=defq
#SBATCH --gres=mps:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/datastore/cndt_thangcpd/linhtruong/workspace5/slurm/logs/%x_%j.out

set -euo pipefail
WS=/datastore/cndt_thangcpd/linhtruong/workspace5
CODE=/datastore/cndt_thangcpd/linhtruong/workspace5/release_EditCTC/code
PY="$CODE/.venv/bin/python"
CFG="${EVAL_CONFIG:-$CODE/config/PP-OCRv6_small_rec_s1024_nrtr_blur_online_w1.yml}"
: "${EVAL_IMAGE_DIR:?EVAL_IMAGE_DIR is required}"
: "${EVAL_LABEL_FILE:?EVAL_LABEL_FILE is required}"
: "${EVAL_OUT_DIR:?EVAL_OUT_DIR is required}"
: "${EVAL_CHECKPOINT:?EVAL_CHECKPOINT is required}"

REQUIRED_VRAM=1200 source "$WS/slurm/gpu_setup.sh"
mkdir -p "$EVAL_OUT_DIR"
EXTRA_OVERRIDES=()
if [[ -n "${EVAL_GATE_THRESHOLD:-}" ]]; then
  EXTRA_OVERRIDES+=("Architecture.Head.edit_gate_threshold=$EVAL_GATE_THRESHOLD")
fi
if [[ -n "${EVAL_DELTA_THRESHOLD:-}" ]]; then
  EXTRA_OVERRIDES+=("Architecture.Head.edit_delta_threshold=$EVAL_DELTA_THRESHOLD")
fi
if [[ -n "${EVAL_EDIT_PRETRAINED:-}" ]]; then
  EXTRA_OVERRIDES+=("Global.edit_refine_pretrained=$EVAL_EDIT_PRETRAINED")
fi
LIST="$EVAL_OUT_DIR/eval_list.txt"
MISSING="$EVAL_OUT_DIR/skipped_missing_images.txt"
BRANCH="$EVAL_OUT_DIR/branch_audit.jsonl"
PRED="$EVAL_OUT_DIR/predictions.txt"
SUMMARY="$EVAL_OUT_DIR/summary.json"
: > "$LIST"
: > "$MISSING"
while IFS=$'\t' read -r file gt; do
  [[ -z "$file" ]] && continue
  if [[ -f "$EVAL_IMAGE_DIR/$file" ]]; then
    printf '%s\t%s\n' "$file" "$gt" >> "$LIST"
  else
    printf '%s\t%s\n' "$file" "$gt" >> "$MISSING"
  fi
done < "$EVAL_LABEL_FILE"

cd "$CODE"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
"$PY" tools/infer_rec.py -c "$CFG" \
  -o Global.checkpoints="$EVAL_CHECKPOINT" \
     Global.pretrained_model=None \
     Global.infer_img="$EVAL_IMAGE_DIR" \
     Global.infer_list="$LIST" \
     Global.save_res_path="$PRED" \
     Global.branch_log_path="$BRANCH" \
     Architecture.Head.branch_debug=True \
     "${EXTRA_OVERRIDES[@]}"

"$PY" - "$EVAL_LABEL_FILE" "$BRANCH" "$PRED" "$MISSING" "$SUMMARY" <<'PY'
import json, sys
from pathlib import Path
labels = {}
label_path, branch_path, pred_path, missing_path, summary_path = map(Path, sys.argv[1:])
for line in label_path.read_text(encoding='utf-8').splitlines():
    if line.strip():
        name, text = line.split('\t', 1)
        labels[Path(name).name] = text
m = {k: 0 for k in ('evaluated','ctc_correct','seed_correct','nrtr_correct','final_correct','changed','helped','hurt')}
if branch_path.exists() and branch_path.stat().st_size:
    rows = [json.loads(x) for x in branch_path.read_text(encoding='utf-8').splitlines() if x.strip()]
    for row in rows:
        name = Path(row['file']).name
        gt = labels.get(name)
        if gt is None: continue
        m['evaluated'] += 1
        ctc = row['ctc']['text']; seed = row['nerd']['seed_text']; final = row['final']['text']
        nrtr = row.get('nrtr') or {}
        m['ctc_correct'] += ctc == gt
        m['seed_correct'] += seed == gt
        m['nrtr_correct'] += nrtr.get('text') == gt
        m['final_correct'] += final == gt
        changed = seed != final or row['nerd']['changed_positions'] > 0
        m['changed'] += changed
        m['helped'] += seed != gt and final == gt
        m['hurt'] += seed == gt and final != gt
else:
    # Integrated NRTR head currently returns only final CTC-vocabulary output;
    # retain a valid final-accuracy summary even without branch diagnostics.
    for line in pred_path.read_text(encoding='utf-8').splitlines():
        parts = line.split('\t')
        if len(parts) < 2: continue
        gt = labels.get(Path(parts[0]).name)
        if gt is None: continue
        m['evaluated'] += 1
        m['final_correct'] += parts[1] == gt
d = max(m['evaluated'], 1)
m.update({k + '_accuracy': m[k + '_correct'] / d for k in ('ctc','seed','nrtr','final')})
m['skipped_missing_images'] = sum(1 for x in missing_path.read_text().splitlines() if x.strip())
m['missing_files'] = [x.split('\t',1)[0] for x in missing_path.read_text().splitlines() if x.strip()]
summary_path.write_text(json.dumps(m, indent=2) + '\n', encoding='utf-8')
print(json.dumps(m, indent=2))
PY
