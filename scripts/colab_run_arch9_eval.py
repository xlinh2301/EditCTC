#!/usr/bin/env python3
"""Run ARCH-9 evaluation (Exp-1 vectorized, Exp-2 adaptive sigma, Exp-3 confidence gating)
across In-domain and Cross-data test sets on Colab GPU.
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime

os.chdir('/tmp/EditCTC_code')
subprocess.run(['git', 'checkout', 'exp/arch9-adaptive-spatial-bias'], check=False)
subprocess.run(['git', 'pull', 'origin', 'exp/arch9-adaptive-spatial-bias'], check=False)

# Dataset paths
INDOMAIN_DIR = "/tmp/curated_eval/indomain"
CROSSDATA_DIR = "/tmp/curated_eval/crossdata"

# Pretrained checkpoint to load
BASE_CKPT = "/content/drive/MyDrive/research/EditCTC/DATA/Checkpoints/EditCTC_ARCH4_AlignCrossAttn_SOTA_s1024/best_accuracy"
CONFIG_ARCH9 = "config/PP-OCRv6_small_rec_s1024_e45_arch9_adaptive_spatial_bias.yml"

def run_eval_pass(config_path, pretrained_model, data_dir, label_file, extra_opts=None):
    cmd = [
        "python3", "tools/eval.py",
        "-c", config_path,
        "-o", f"Global.pretrained_model={pretrained_model}",
        "-o", f"Eval.dataset.data_dir={data_dir}",
        "-o", f"Eval.dataset.label_file_list=['{label_file}']",
    ]
    if extra_opts:
        cmd.extend(extra_opts)

    start_t = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_t

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

    return {
        "acc": acc,
        "cer": cer,
        "elapsed_s": elapsed,
        "returncode": res.returncode,
        "raw_output": stdout[-1500:] if res.returncode != 0 else stdout[-500:],
    }

def main():
    print("=" * 70)
    print("STARTING EDITCTC ARCH-9 BENCHMARK EVALUATION (EXP-1, EXP-2, EXP-3)")
    print("=" * 70)

    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiments": {}
    }

    # 1. Exp-1 / Exp-2: Adaptive Spatial Sigma Baseline
    print("\n[1/3] Running ARCH-9 Baseline (Adaptive Sigma enabled)...")
    res_in = run_eval_pass(
        CONFIG_ARCH9,
        BASE_CKPT,
        INDOMAIN_DIR,
        f"{INDOMAIN_DIR}/labels.txt",
    )
    res_cross = run_eval_pass(
        CONFIG_ARCH9,
        BASE_CKPT,
        CROSSDATA_DIR,
        f"{CROSSDATA_DIR}/labels.txt",
    )
    results["experiments"]["ARCH-9_adaptive_sigma"] = {
        "indomain": res_in,
        "crossdata": res_cross,
    }
    print(f" -> Indomain: Acc={res_in['acc']}, CER={res_in['cer']}")
    print(f" -> Crossdata: Acc={res_cross['acc']}, CER={res_cross['cer']}")

    # 2. Exp-3: CTC Confidence Gating Margin Sweep
    for margin_thresh in [0.70, 0.85, 0.90, 0.95]:
        exp_name = f"ARCH-9_conf_gate_margin_{margin_thresh}"
        print(f"\n[2/3] Running {exp_name}...")
        opts = [f"-o Architecture.Head.ctc_confidence_gate_margin={margin_thresh}"]
        r_in = run_eval_pass(CONFIG_ARCH9, BASE_CKPT, INDOMAIN_DIR, f"{INDOMAIN_DIR}/labels.txt", opts)
        r_cross = run_eval_pass(CONFIG_ARCH9, BASE_CKPT, CROSSDATA_DIR, f"{CROSSDATA_DIR}/labels.txt", opts)
        results["experiments"][exp_name] = {
            "indomain": r_in,
            "crossdata": r_cross,
        }
        print(f" -> Indomain: Acc={r_in['acc']}, CER={r_in['cer']}")
        print(f" -> Crossdata: Acc={r_cross['acc']}, CER={r_cross['cer']}")

    out_json = "/content/drive/MyDrive/research/EditCTC/logs/2026-10-04/arch9_benchmark_results.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] Results saved to: {out_json}")

if __name__ == "__main__":
    main()
EOF
