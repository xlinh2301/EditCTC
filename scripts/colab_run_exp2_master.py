#!/usr/bin/env python3
"""Master Exp-2 (ARCH-9 Adaptive Spatial Sigma) Training & Evaluation Runner on Colab GPU T4.
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime

print("=" * 70)
print("🚀 STARTING EDITCTC EXP-2 (ARCH-9 ADAPTIVE SPATIAL SIGMA) MASTER RUNNER")
print("=" * 70)

# 1. Mount Google Drive if not mounted
print("\n[1/6] Checking Google Drive...")
if not os.path.exists('/content/drive/MyDrive'):
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        print("Google Drive mounted successfully.")
    except Exception as e:
        print("Drive mount exception:", e)

# 2. Sync Repository
print("\n[2/6] Syncing EditCTC repository (branch exp/arch9-adaptive-spatial-bias)...")
WORKDIR = "/tmp/EditCTC_code"
os.makedirs(WORKDIR, exist_ok=True)
if not os.path.exists(f"{WORKDIR}/.git"):
    subprocess.run(["git", "clone", "https://github.com/xlinh2301/EditCTC.git", WORKDIR], check=True)

os.chdir(WORKDIR)
subprocess.run(["git", "checkout", "exp/arch9-adaptive-spatial-bias"], check=False)
subprocess.run(["git", "pull", "origin", "exp/arch9-adaptive-spatial-bias"], check=False)

# 3. Unpack Datasets
print("\n[3/6] Unpacking curated datasets...")
os.makedirs("/tmp/curated_eval", exist_ok=True)
subprocess.run("tar -xzf /content/drive/MyDrive/research/EditCTC/DATA/Indomain_curated.tar.gz -C /tmp/curated_eval/ || true", shell=True)
subprocess.run("tar -xzf /content/drive/MyDrive/research/EditCTC/DATA/Cross-data_curated.tar.gz -C /tmp/curated_eval/ || true", shell=True)

# Find label files
INDOMAIN_DIR = "/tmp/curated_eval/indomain" if os.path.exists("/tmp/curated_eval/indomain") else "/tmp/curated_eval/Indomain_curated"
CROSSDATA_DIR = "/tmp/curated_eval/crossdata" if os.path.exists("/tmp/curated_eval/crossdata") else "/tmp/curated_eval/Cross-data_curated"

print(f"Indomain path: {INDOMAIN_DIR}")
print(f"Crossdata path: {CROSSDATA_DIR}")

# 4. Verify GPU & Unit Tests
print("\n[4/6] Verifying GPU & Running Unit Tests...")
subprocess.run([sys.executable, "-m", "unittest", "tests/test_arch9_adaptive_spatial_bias.py"], check=True)

# 5. Launch Training (Fine-tuning beta from SOTA checkpoint)
print("\n[5/6] Launching ARCH-9 Training (Exp-2)...")
CONFIG_PATH = "config/PP-OCRv6_small_rec_s1024_e45_arch9_adaptive_spatial_bias.yml"
PRETRAINED_MODEL = "/content/drive/MyDrive/research/EditCTC/DATA/Checkpoints/EditCTC_ARCH4_AlignCrossAttn_SOTA_s1024/best_accuracy"
SAVE_MODEL_DIR = "/content/drive/MyDrive/research/EditCTC/DATA/Checkpoints/EditCTC_ARCH9_AdaptiveSpatial_s1024"

train_cmd = [
    sys.executable, "tools/train.py",
    "-c", CONFIG_PATH,
    "-o", f"Global.pretrained_model={PRETRAINED_MODEL}",
    "-o", f"Global.save_model_dir={SAVE_MODEL_DIR}",
    "-o", "Global.epoch_num=10",
]

print("Running command:", " ".join(train_cmd))
train_res = subprocess.run(train_cmd, capture_output=True, text=True)
print("Training returncode:", train_res.returncode)
print("Training output tail:\n", train_res.stdout[-1500:] if train_res.stdout else train_res.stderr[-1500:])

# 6. Evaluation Benchmark
print("\n[6/6] Running Benchmark Evaluation on Curated Test Sets...")
EVAL_CKPT = f"{SAVE_MODEL_DIR}/best_accuracy" if os.path.exists(f"{SAVE_MODEL_DIR}/best_accuracy.pdparams") else PRETRAINED_MODEL

def eval_dataset(name, data_dir, label_file):
    cmd = [
        sys.executable, "tools/eval.py",
        "-c", CONFIG_PATH,
        "-o", f"Global.pretrained_model={EVAL_CKPT}",
        "-o", f"Eval.dataset.data_dir={data_dir}",
        "-o", f"Eval.dataset.label_file_list=['{label_file}']",
    ]
    st = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    el = time.time() - st
    acc = None
    cer = None
    stdout = res.stdout + "\n" + res.stderr
    for line in stdout.splitlines():
        if "acc:" in line:
            try:
                acc = float(line.split("acc:")[1].split(",")[0].strip())
            except Exception:
                pass
        if "norm_edit_dis:" in line:
            try:
                cer = 1.0 - float(line.split("norm_edit_dis:")[1].split(",")[0].strip())
            except Exception:
                pass
    print(f" -> {name}: Acc = {acc}, CER = {cer} ({el:.2f}s)")
    return {"acc": acc, "cer": cer, "latency_s": el}

# Find label txt
in_label = f"{INDOMAIN_DIR}/labels.txt" if os.path.exists(f"{INDOMAIN_DIR}/labels.txt") else f"{INDOMAIN_DIR}/test_label.txt"
if not os.path.exists(in_label):
    in_label = "/tmp/curated_eval/indomain/Indomain_curated/crops/test_label.txt"

cross_label = f"{CROSSDATA_DIR}/labels.txt" if os.path.exists(f"{CROSSDATA_DIR}/labels.txt") else f"{CROSSDATA_DIR}/crossdata_label.txt"
if not os.path.exists(cross_label):
    cross_label = "/tmp/curated_eval/crossdata/Cross-data_curated/crops/crossdata_label.txt"

in_metrics = eval_dataset("In-Domain Curated", INDOMAIN_DIR, in_label)
cross_metrics = eval_dataset("Cross-Data Curated", CROSSDATA_DIR, cross_label)

out_summary = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model": "EditCTC ARCH-9 (Exp-2 Adaptive Spatial Sigma)",
    "checkpoint": EVAL_CKPT,
    "indomain": in_metrics,
    "crossdata": cross_metrics,
}

out_file = "/content/drive/MyDrive/research/EditCTC/logs/2026-10-04/exp2_arch9_benchmark_results.json"
os.makedirs(os.path.dirname(out_file), exist_ok=True)
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(out_summary, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 70)
print(f"🎉 EXP-2 BENCHMARK COMPLETED SUCCESSFULLY! Results saved to {out_file}")
print("=" * 70)
