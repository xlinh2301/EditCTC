"""Fine-grained Inference Sweep & Pareto Curve Generator (CorrectionRecall vs OverCorrectionRate).

Sweeps gate threshold in [0.40, 0.65] and delta margin in [0.00, 0.10].
Computes:
  - Exact Match & Accuracy
  - CER (Character Error Rate)
  - CorrectionRecall = sum(repaired) / sum(REPLACE)
  - OverCorrectionRate = sum(over_corrected) / sum(KEEP)
  - CorrectSeedPreserveRate = 1.0 - OverCorrectionRate
Identifies the Pareto optimal frontier and the knee point.
"""
from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import os
import sys
import numpy as np

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

import paddle
from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.losses.rec_edit_loss import levenshtein_ops, KEEP, REPLACE, IGNORE_INDEX
import tools.program as program


def levenshtein_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


def main():
    config, device, logger, vdl_writer = program.preprocess()

    save_dir = config["Global"].get("save_res_path", "./output/pareto_sweep")
    os.makedirs(save_dir, exist_ok=True)

    post_process_class = build_post_process(config["PostProcess"], config["Global"])
    char_list = getattr(post_process_class, "character", [])
    vocab_size = len(char_list)
    char_dict = {c: i for i, c in enumerate(char_list)}

    # Ensure out_channels_list is set up for multi-head architectures
    char_num = len(char_list)
    if config["PostProcess"]["name"] == "SARLabelDecode":
        char_num = char_num - 2
    if config["PostProcess"]["name"] == "NRTRLabelDecode":
        char_num = char_num - 3
    out_channels_list = {
        "CTCLabelDecode": char_num,
        "SARLabelDecode": char_num + 2,
        "NRTRLabelDecode": char_num + 3,
    }
    config["Architecture"]["Head"]["out_channels_list"] = out_channels_list
    config["Architecture"]["Head"]["branch_debug"] = True
    config["Architecture"]["Head"]["edit_allowed_ops"] = [0, 1]  # seed-only substitution

    model = build_model(config["Architecture"])
    load_model(config, model)
    model.eval()

    # Create data transforms
    transforms = []
    for op in config["Eval"]["dataset"]["transforms"]:
        op_name = list(op)[0]
        if "Label" in op_name:
            continue
        elif op_name in ["RecResizeImg"]:
            op[op_name]["infer_mode"] = True
        elif op_name == "KeepKeys":
            op[op_name]["keep_keys"] = ["image"]
        transforms.append(op)
    ops = create_operators(transforms, config["Global"])

    # Load labels
    infer_list_path = config["Global"]["infer_list"]
    with open(infer_list_path, "r", encoding="utf-8") as f:
        lines = [line.strip().split("\t") for line in f if line.strip()]

    infer_img_dir = config["Global"]["infer_img"]
    samples = []
    for parts in lines:
        fn = parts[0]
        gt_text = parts[1] if len(parts) > 1 else ""
        img_path = os.path.join(infer_img_dir, os.path.basename(fn))
        if os.path.exists(img_path):
            samples.append((img_path, gt_text))
        elif os.path.exists(fn):
            samples.append((fn, gt_text))

    logger.info(f"Loaded {len(samples)} evaluation samples from {infer_list_path}")

    # Step 1: Run inference and extract logits/seeds
    records = []
    logger.info("Extracting model representations for evaluation set...")

    for idx, (img_path, gt_str) in enumerate(samples):
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        data = {"image": img_bytes}
        batch = transform(data, ops)
        images = paddle.to_tensor(np.expand_dims(batch[0], axis=0))

        with paddle.no_grad():
            preds = model(images)

        branch_debug = preds.get("branch_debug", {})
        seeds_np = branch_debug.get("seed_ids")[0]   # shape: [max_len]
        lens_np = int(branch_debug.get("seed_lens")[0])
        tok_logits = branch_debug.get("edit_tok_logits")[0] # shape: [max_len, vocab]

        # Truncate to true seed length
        seed_ids = seeds_np[:lens_np].copy()
        tok_logits = tok_logits[:lens_np].copy()

        # Compute softmax probabilities over vocab
        tok_logits_shifted = tok_logits - tok_logits.max(axis=-1, keepdims=True)
        exp_logits = np.exp(tok_logits_shifted)
        tok_probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

        best_ids = np.argmax(tok_probs, axis=-1)
        best_probs = np.max(tok_probs, axis=-1)

        seed_clip = np.clip(seed_ids, 0, tok_probs.shape[-1] - 1)
        seed_probs = tok_probs[np.arange(lens_np), seed_clip]

        # Decode ground truth text to token IDs
        gt_ids = [char_dict[c] for c in gt_str if c in char_dict]

        # Compute optimal Levenshtein alignment
        op_t, tok_t = levenshtein_ops(seed_ids.tolist(), gt_ids)
        op_t = np.array(op_t, dtype=np.int32)
        tok_t = np.array(tok_t, dtype=np.int64)

        # Baseline CTC decoded string
        ctc_chars = [char_list[i] for i in seed_ids if 0 <= i < len(char_list) and i != 0]
        ctc_str = "".join(ctc_chars)

        # Check if explicit change head or ctc confs are present
        change_probs_all = branch_debug.get("change_probs")
        if change_probs_all is not None:
            change_probs = change_probs_all[0][:lens_np].copy()
        else:
            change_probs = None

        confs_all = branch_debug.get("ctc_confs")
        if confs_all is not None:
            ctc_confs = confs_all[0][:lens_np].copy()
        else:
            ctc_confs = None

        records.append({
            "path": img_path,
            "gt_str": gt_str,
            "ctc_str": ctc_str,
            "seed_ids": seed_ids,
            "lens": lens_np,
            "best_ids": best_ids,
            "best_probs": best_probs,
            "seed_probs": seed_probs,
            "change_probs": change_probs,
            "ctc_confs": ctc_confs,
            "op_t": op_t,
            "tok_t": tok_t,
        })

    logger.info("Extraction complete. Beginning 2D Grid Sweep over (threshold, delta)...")

    # Step 2: Grid sweep
    gate_thresholds = np.round(np.arange(0.40, 0.651, 0.01), 2)
    deltas = np.round(np.arange(0.00, 0.101, 0.01), 2)

    total_samples = len(records)
    total_gt_chars = sum(len(r["gt_str"]) for r in records)

    # Count total REPLACE and KEEP positions across the dataset
    total_replace_pos = sum(np.sum(r["op_t"] == REPLACE) for r in records)
    total_keep_pos = sum(np.sum(r["op_t"] == KEEP) for r in records)

    results = []

    for gate in gate_thresholds:
        for delta in deltas:
            exact_count = 0
            total_char_dist = 0
            total_repaired = 0
            total_over_corr = 0
            total_fires = 0

            for r in records:
                seed_ids = r["seed_ids"]
                best_ids = r["best_ids"]
                best_probs = r["best_probs"]
                seed_probs = r["seed_probs"]
                op_t = r["op_t"]
                tok_t = r["tok_t"]
                gt_str = r["gt_str"]

                # Fire condition
                ch_p = r["change_probs"] if r["change_probs"] is not None else best_probs
                fire = (best_ids != seed_ids) & (ch_p >= gate) & ((best_probs - seed_probs) >= delta)
                pred_ids = np.where(fire, best_ids, seed_ids)

                # Decode string
                pred_chars = [char_list[i] for i in pred_ids if 0 <= i < len(char_list) and i != 0]
                pred_str = "".join(pred_chars)

                if pred_str == gt_str:
                    exact_count += 1
                total_char_dist += levenshtein_distance(pred_str, gt_str)

                # Diagnostics
                repaired = fire & (op_t == REPLACE) & (pred_ids == tok_t)
                over_corr = fire & (op_t == KEEP)

                total_repaired += int(np.sum(repaired))
                total_over_corr += int(np.sum(over_corr))
                total_fires += int(np.sum(fire))

            acc = exact_count / total_samples
            cer = total_char_dist / total_gt_chars
            recall = total_repaired / max(total_replace_pos, 1)
            over_corr_rate = total_over_corr / max(total_keep_pos, 1)
            preserve_rate = 1.0 - over_corr_rate

            results.append({
                "gate": float(gate),
                "delta": float(delta),
                "exact": exact_count,
                "total": total_samples,
                "accuracy": acc * 100.0,
                "cer": cer * 100.0,
                "recall": recall * 100.0,
                "over_corr_rate": over_corr_rate * 100.0,
                "preserve_rate": preserve_rate * 100.0,
                "fires": total_fires,
            })

    # Step 3: Save CSV
    csv_path = os.path.join(save_dir, "pareto_grid_sweep.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "gate", "delta", "exact", "total", "accuracy", "cer", "recall", "over_corr_rate", "preserve_rate", "fires"
        ])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    logger.info(f"Saved complete sweep results to {csv_path}")

    # Step 4: Extract Pareto Frontier (Recall vs OverCorrectionRate)
    # Sort by OverCorrectionRate ascending, then Recall descending
    sorted_results = sorted(results, key=lambda x: (x["over_corr_rate"], -x["recall"]))
    pareto_frontier = []
    max_recall_so_far = -1.0

    for pt in sorted_results:
        if pt["recall"] > max_recall_so_far:
            pareto_frontier.append(pt)
            max_recall_so_far = pt["recall"]

    # Step 5: Find Knee Point
    # Knee point maximizes (Recall - lambda * OverCorrectionRate) or trade-off angle
    # Normalized knee detection:
    best_tradeoff_score = -1e9
    knee_point = None
    for pt in pareto_frontier:
        # Weighting: 1% recall gain is valuable if it costs <= 0.2% over-correction
        # Equivalent to score = Recall - 5 * OverCorrectionRate
        score = pt["recall"] - 5.0 * pt["over_corr_rate"]
        if score > best_tradeoff_score:
            best_tradeoff_score = score
            knee_point = pt

    # Print summary
    print("\n" + "=" * 90)
    print("PARETO CURVE & INFERENCE THRESHOLD SWEEP SUMMARY")
    print("=" * 90)
    print(f"Total Evaluated Combinations: {len(results)}")
    print(f"Target PP-OCRv4: 93.68% (548 / 585)")
    print("-" * 90)

    # Baseline CTC
    ctc_exact = sum(1 for r in records if r["ctc_str"] == r["gt_str"])
    ctc_cer = sum(levenshtein_distance(r["ctc_str"], r["gt_str"]) for r in records) / total_gt_chars
    print(f"Baseline CTC Seed Alone: {ctc_exact}/{total_samples} ({ctc_exact/total_samples*100:.2f}%), CER: {ctc_cer*100:.2f}%\n")

    # Top Accuracy configurations
    top_acc = sorted(results, key=lambda x: (-x["accuracy"], x["cer"]))[:5]
    print("--- TOP 5 CONFIGURATIONS BY ACCURACY ---")
    for r in top_acc:
        star = " 🎯 TARGET REACHED!" if r["exact"] >= 548 else ""
        print(f"gate={r['gate']:.2f}, delta={r['delta']:.2f} -> Acc: {r['accuracy']:.2f}% ({r['exact']}/{r['total']}), CER: {r['cer']:.2f}%, Recall: {r['recall']:.2f}%, OverCorr: {r['over_corr_rate']:.2f}%{star}")

    print("\n--- PARETO FRONTIER: CorrectionRecall vs OverCorrectionRate ---")
    print(f"{'Gate':<6} {'Delta':<6} {'Exact':<10} {'Acc (%)':<9} {'CER (%)':<9} {'Recall (%)':<12} {'OverCorr (%)':<14} {'Preserve (%)':<13}")
    print("-" * 85)
    for pt in pareto_frontier:
        is_knee = " ⭐ [KNEE POINT]" if pt == knee_point else ""
        print(f"{pt['gate']:<6.2f} {pt['delta']:<6.2f} {pt['exact']}/{pt['total']:<5} {pt['accuracy']:<9.2f} {pt['cer']:<9.2f} {pt['recall']:<12.2f} {pt['over_corr_rate']:<14.2f} {pt['preserve_rate']:<13.2f}{is_knee}")

    if knee_point:
        print("\n" + "=" * 90)
        print(f"IDENTIFIED KNEE POINT: gate={knee_point['gate']:.2f}, delta={knee_point['delta']:.2f}")
        print(f"Accuracy: {knee_point['accuracy']:.2f}% ({knee_point['exact']}/{knee_point['total']}), CER: {knee_point['cer']:.2f}%")
        print(f"CorrectionRecall: {knee_point['recall']:.2f}%, OverCorrectionRate: {knee_point['over_corr_rate']:.2f}%, PreserveRate: {knee_point['preserve_rate']:.2f}%")
        print("=" * 90 + "\n")

    # Save summary json
    summary_path = os.path.join(save_dir, "pareto_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "target": {"acc": 93.68, "exact": 548, "total": 585},
            "baseline_ctc": {"exact": ctc_exact, "total": total_samples, "acc": ctc_exact / total_samples * 100, "cer": ctc_cer * 100},
            "top_acc": top_acc,
            "knee_point": knee_point,
            "pareto_frontier": pareto_frontier,
        }, f, indent=2)
    logger.info(f"Saved summary JSON to {summary_path}")


if __name__ == "__main__":
    main()
