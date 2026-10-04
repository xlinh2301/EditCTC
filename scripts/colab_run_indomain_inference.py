#!/usr/bin/env python3
"""
Indomain Validation & Evaluation Runner for EditCTC on Colab GPU
Evaluates EditCTC model across seeds and collects detailed predictions for visualization.
"""

import os
import sys
import json
import statistics
import time

def run_indomain_evaluation():
    print("==========================================================")
    print("⚡ EditCTC In-Domain V4 Evaluation & Inference on Colab GPU")
    print("==========================================================")

    base_dir = "/content/drive/MyDrive/research/EditCTC/release_EditCTC"
    gt_path = os.path.join(base_dir, "data", "in-domain", "gt_test_v4.json")
    preds_dir = os.path.join(base_dir, "eval", "results_indomain_v4", "preds")
    out_summary_path = os.path.join(base_dir, "eval", "indomain_val_comparison.json")

    if not os.path.exists(gt_path):
        print(f"[!] Ground truth file not found at: {gt_path}")
        return

    print(f"[*] Loading Ground Truth: {gt_path}")
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    gt_by_id = {g["id"]: g["text"] for g in gt_data}
    total_samples = len(gt_data)
    print(f"[+] Total validation samples: {total_samples}")

    seeds = [1024, 2048, 4096, 8192, 16384]
    
    models = {
        "EditCTC (proposed)": {
            "type": "paddle",
            "fn": lambda s: f"preds_editctc_proposed_s{s}.json"
        },
        "PP-OCRv6": {
            "type": "paddle",
            "fn": lambda s: f"preds_v6_s{s}.json"
        },
        "PP-OCRv5": {
            "type": "paddle",
            "fn": lambda s: f"preds_v5_s{s}.json"
        },
        "PP-OCRv4": {
            "type": "paddle",
            "fn": lambda s: f"preds_v4_s{s}.json"
        },
        "SVTRv2": {
            "type": "paddle",
            "fn": lambda s: f"preds_svtrv2_s{s}.json"
        },
        "SATRN": {
            "type": "mmocr",
            "fn": lambda s: f"preds_satrn_indomain_s{s}_V4.json"
        },
        "ABINet": {
            "type": "mmocr",
            "fn": lambda s: f"preds_abinet_indomain_s{s}_V4.json"
        },
        "MASTER": {
            "type": "mmocr",
            "fn": lambda s: f"preds_master_indomain_s{s}_V4.json"
        },
        "SAR": {
            "type": "mmocr",
            "fn": lambda s: f"preds_sar_indomain_s{s}_V4.json"
        }
    }

    results_summary = {}
    model_predictions_s1024 = {}

    for model_name, cfg in models.items():
        seed_accs = []
        for seed in seeds:
            pred_file = os.path.join(preds_dir, cfg["fn"](seed))
            if os.path.exists(pred_file):
                with open(pred_file, "r", encoding="utf-8") as f:
                    blob = json.load(f)
                items = blob["results"] if isinstance(blob, dict) and "results" in blob else blob
                
                correct = 0
                preds_map = {}
                for item in items:
                    sample_id = item.get("id")
                    pred_txt = item.get("text", "") or ""
                    gt_txt = gt_by_id.get(sample_id, "")
                    preds_map[sample_id] = pred_txt
                    if pred_txt == gt_txt:
                        correct += 1
                
                acc = (correct / total_samples) * 100.0
                seed_accs.append(acc)

                if seed == 1024:
                    model_predictions_s1024[model_name] = preds_map
            else:
                print(f"[!] Warning: Missing prediction file: {pred_file}")

        if seed_accs:
            mean_acc = statistics.mean(seed_accs)
            std_acc = statistics.pstdev(seed_accs)
            results_summary[model_name] = {
                "mean_accuracy": round(mean_acc, 2),
                "std": round(std_acc, 2),
                "per_seed": [round(a, 2) for a in seed_accs]
            }
            print(f" - {model_name:20s}: {mean_acc:6.2f}% ± {std_acc:4.2f}%  (seeds: {seed_accs})")

    # Build per-sample detailed comparison
    detailed_samples = []
    for g in gt_data:
        sid = g["id"]
        gt_text = g["text"]
        sample_info = {
            "id": sid,
            "ground_truth": gt_text,
            "predictions": {m: model_predictions_s1024.get(m, {}).get(sid, "") for m in models.keys()},
            "is_editctc_correct": model_predictions_s1024.get("EditCTC (proposed)", {}).get(sid, "") == gt_text,
            "is_v6_correct": model_predictions_s1024.get("PP-OCRv6", {}).get(sid, "") == gt_text,
            "is_svtrv2_correct": model_predictions_s1024.get("SVTRv2", {}).get(sid, "") == gt_text,
        }
        detailed_samples.append(sample_info)

    export_payload = {
        "dataset": "in-domain-v4",
        "total_samples": total_samples,
        "summary": results_summary,
        "samples": detailed_samples
    }

    with open(out_summary_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2, ensure_ascii=False)

    print(f"\n[+] Exported comparison JSON to: {out_summary_path} ({os.path.getsize(out_summary_path)/(1024):.1f} KB)")
    print("==========================================================")
    print("🎉 In-domain evaluation completed successfully!")
    print("==========================================================")

if __name__ == "__main__":
    run_indomain_evaluation()
