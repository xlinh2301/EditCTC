#!/bin/bash
#SBATCH --job-name=inspect24
#SBATCH --partition=defq
#SBATCH --gres=mps:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/datastore/cndt_thangcpd/linhtruong/workspace5/slurm/logs/%x_%j.out

set -euo pipefail
WS=/datastore/cndt_thangcpd/linhtruong/workspace5
CODE="$WS/release_EditCTC/code"
PY="$CODE/.venv/bin/python"
CFG="$CODE/config/PP-OCRv6_small_rec_s1024_e44_highres_curated.yml"
CKPT="$WS/Data/EditCTC_arbor_runs/curated_runs/e44_highres_curated/best_accuracy"
TMP_DIR="/tmp/test_24_crops_job$SLURM_JOB_ID"
OUT_TXT="$WS/Data/Indomain_curated/crops/test_newly_added_preds.txt"

rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"
while IFS= read -r p; do
  [ -f "$p" ] && ln -sf "$p" "$TMP_DIR/"
done < "$WS/Data/Indomain_curated/crops/test_newly_added_paths.txt"

REQUIRED_VRAM=1500 source "$WS/slurm/gpu_setup.sh"
cd "$CODE"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"

echo "Running inference on 24 new test samples in $TMP_DIR..."
"$PY" tools/infer_rec.py -c "$CFG" \
  -o Global.checkpoints="$CKPT" \
     Global.pretrained_model=None \
     Global.infer_img="$TMP_DIR" \
     Global.save_res_path="$OUT_TXT"

echo ""
echo "Comparison with Ground Truth:"
python3 -c '
gt_map = {}
with open("/datastore/cndt_thangcpd/linhtruong/workspace5/Data/Indomain_curated/crops/test_newly_added_label.txt") as f:
    for line in f:
        if line.strip():
            fn, lbl = line.strip().split("\t")
            gt_map[fn] = lbl

with open("'"$OUT_TXT"'") as f:
    correct = 0
    total = 0
    for line in f:
        if not line.strip(): continue
        parts = line.strip().split("\t")
        full_path = parts[0]
        fn = full_path.split("/")[-1]
        pred = parts[1] if len(parts) > 1 else ""
        conf = parts[2] if len(parts) > 2 else ""
        gt = gt_map.get(fn, "UNKNOWN")
        match = "✓ MATCH" if pred == gt else "✗ WRONG"
        if pred == gt: correct += 1
        total += 1
        print(f"{match:8} | GT: {gt:10} | Pred: {pred:10} | Conf: {conf[:5]:5} | {fn}")

print("-" * 65)
print(f"Accuracy on 24 new crops: {correct}/{total} ({correct/total*100:.2f}%)")
'
rm -rf "$TMP_DIR"
