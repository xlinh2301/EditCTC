#!/usr/bin/env python3
"""
Master Live Forward-Pass Evaluator for All EditCTC & Baseline Checkpoints on Indomain_curated Test Set
Evaluates 16+ checkpoints directly on GPU T4 and records live predictions, exact accuracy, CER, and error audit logs.
"""

import os
import sys
import time
import json
import yaml
import numpy as np

def compute_levenshtein(gt: str, pred: str) -> int:
    dp = [[0] * (len(pred) + 1) for _ in range(len(gt) + 1)]
    for i in range(len(gt) + 1):
        dp[i][0] = i
    for j in range(len(pred) + 1):
        dp[0][j] = j
    for i in range(1, len(gt) + 1):
        for j in range(1, len(pred) + 1):
            if gt[i - 1] == pred[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[len(gt)][len(pred)]

def run_master_checkpoints_eval():
    sys.path.insert(0, '/tmp/EditCTC_code')
    sys.path.insert(0, '/tmp/bundle_essentials')
    
    import paddle
    import cv2
    from ppocr.modeling.architectures import build_model

    logs = []
    def log(msg):
        logs.append(str(msg))

    log("=================================================================")
    log("🚀 RUNNING LIVE FORWARD PASS ON ALL CHECKPOINTS (INDOMAIN CURATED)")
    log(f"   Date: 2026-10-04 | GPU: {paddle.device.get_device()}")
    log("=================================================================")

    # 1. Setup Dictionary
    dict_path = "/tmp/EditCTC_code/ppocr/utils/dict/ppocrv6_dict.txt"
    if not os.path.exists(dict_path):
        dict_path = "/content/drive/MyDrive/research/EditCTC/source/ppocr/utils/dict/ppocrv6_dict.txt"

    char_list = []
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            char_list.append(line.strip("\n\r"))
    char_list.append(" ")
    vocab_size = len(char_list) + 1

    # 2. Load Indomain Curated Images (585 samples)
    test_img_dir = "/tmp/curated_eval/indomain/Indomain_curated/crops/test"
    label_file = "/tmp/bundle_essentials/workdir_eval/recognition/curated_eval_labels/indomain_test_label.present.txt"
    if not os.path.exists(label_file):
        label_file = "/tmp/curated_eval/indomain/Indomain_curated/crops/test_label.txt"

    samples = []
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                img_name = os.path.basename(parts[0])
                gt_text = parts[1]
                full_img_path = os.path.join(test_img_dir, img_name)
                if os.path.exists(full_img_path):
                    samples.append((full_img_path, img_name, gt_text))

    log(f"[+] Loaded {len(samples)} physical test images from {test_img_dir}")

    # Preload and preprocess all images into memory for fair speed benchmarking
    def preprocess_image(img_path, target_h=48, target_w=320):
        img = cv2.imread(img_path)
        if img is None:
            return None
        resized = cv2.resize(img, (target_w, target_h))
        normalized = (resized.astype("float32") / 255.0 - 0.5) / 0.5
        chw = normalized.transpose((2, 0, 1))
        return chw

    preprocessed_tensors = []
    valid_samples = []
    for s in samples:
        t = preprocess_image(s[0])
        if t is not None:
            preprocessed_tensors.append(t)
            valid_samples.append(s)

    total_images = len(valid_samples)
    log(f"[+] Preprocessed {total_images} images ready in memory.")

    # 3. Checkpoints to Evaluate
    checkpoints_dir = "/content/drive/MyDrive/research/EditCTC/DATA/Checkpoints"
    
    checkpoints_suite = [
        # --- Top 10 EditCTC Checkpoints ---
        {
            "name": "EditCTC EXP-18B (Canonical Baseline)",
            "file": os.path.join(checkpoints_dir, "EditCTC_Top10_Best_Checkpoints", "EditCTC_EXP18B_CanonicalBaseline_Best_s3024.pdparams"),
            "head": "MultiHeadEditRefineNRTR",
            "align_guided": False,
            "use_explicit": True
        },
        {
            "name": "EditCTC ARCH-4 (Align-Guided Cross-Attn)",
            "file": os.path.join(checkpoints_dir, "EditCTC_Top10_Best_Checkpoints", "EditCTC_ARCH4_AlignCrossAttn_SOTA_s1024.pdparams"),
            "head": "MultiHeadEditRefineNRTR",
            "align_guided": True,
            "use_explicit": True
        },
        {
            "name": "EditCTC ARCH-4C (Align+Conf4D)",
            "file": os.path.join(checkpoints_dir, "EditCTC_Top10_Best_Checkpoints", "EditCTC_ARCH4C_AlignConf4D_Best_s2024.pdparams"),
            "head": "MultiHeadEditRefineNRTR",
            "align_guided": True,
            "use_explicit": True
        },
        {
            "name": "EditCTC ARCH-2B (Full Conf4D)",
            "file": os.path.join(checkpoints_dir, "EditCTC_Top10_Best_Checkpoints", "EditCTC_ARCH2B_FullConf4D_Best_s3024.pdparams"),
            "head": "MultiHeadEditRefineNRTR",
            "align_guided": False,
            "use_explicit": True
        },
        {
            "name": "EditCTC ARCH-3 (Temporal Align)",
            "file": os.path.join(checkpoints_dir, "EditCTC_Top10_Best_Checkpoints", "EditCTC_ARCH3_TemporalAlign_Best_s2024.pdparams"),
            "head": "MultiHeadEditRefineNRTR",
            "align_guided": False,
            "use_explicit": True
        },
        {
            "name": "EditCTC ARCH-5 (Local Visual Refine)",
            "file": os.path.join(checkpoints_dir, "EditCTC_Top10_Best_Checkpoints", "EditCTC_ARCH5_LocalVisualRefine_Best_s5024.pdparams"),
            "head": "MultiHeadEditRefineNRTR",
            "align_guided": False,
            "use_explicit": True
        },
        {
            "name": "EditCTC ARCH-7 (Gated Fusion)",
            "file": os.path.join(checkpoints_dir, "EditCTC_Top10_Best_Checkpoints", "EditCTC_ARCH7_GatedFusion_Best_s3024.pdparams"),
            "head": "MultiHeadEditRefineNRTR",
            "align_guided": False,
            "use_explicit": True
        },
        {
            "name": "EditCTC ARCH-1 (Explicit Head)",
            "file": os.path.join(checkpoints_dir, "EditCTC_Top10_Best_Checkpoints", "EditCTC_ARCH1_ExplicitHead_Best_s4024.pdparams"),
            "head": "MultiHeadEditRefineNRTR",
            "align_guided": False,
            "use_explicit": True
        },
        {
            "name": "EditCTC ARCH-2A (CTC Conf2D)",
            "file": os.path.join(checkpoints_dir, "EditCTC_Top10_Best_Checkpoints", "EditCTC_ARCH2A_CTCConf2D_Best_s5024.pdparams"),
            "head": "MultiHeadEditRefineNRTR",
            "align_guided": False,
            "use_explicit": True
        },
        {
            "name": "EditCTC ARCH-4B (Align+Append)",
            "file": os.path.join(checkpoints_dir, "EditCTC_Top10_Best_Checkpoints", "EditCTC_ARCH4B_AlignAppend_Best_s1024.pdparams"),
            "head": "MultiHeadEditRefineNRTR",
            "align_guided": True,
            "use_explicit": True
        },
        # --- Baselines ---
        {
            "name": "PP-OCRv4 Server (Baseline)",
            "file": os.path.join(checkpoints_dir, "Benchmark_Baselines_Best_Checkpoints", "PPOCRv4_curated_s1024.pdparams"),
            "head": "CTCHead",
            "align_guided": False,
            "use_explicit": False
        },
        {
            "name": "PP-OCRv6 Server (Baseline)",
            "file": os.path.join(checkpoints_dir, "Benchmark_Baselines_Best_Checkpoints", "PPOCRv6_curated_s1024.pdparams"),
            "head": "CTCHead",
            "align_guided": False,
            "use_explicit": False
        },
    ]

    benchmark_results = {}
    batch_size = 32

    for model_meta in checkpoints_suite:
        m_name = model_meta["name"]
        cp_file = model_meta["file"]
        
        if not os.path.exists(cp_file):
            log(f"[!] Warning: Checkpoint not found: {cp_file}")
            continue

        log(f"\n[*] Evaluating: {m_name} from {os.path.basename(cp_file)}...")
        
        # Build specific model architecture
        if model_meta["head"] == "CTCHead":
            arch_cfg = {
                "model_type": "rec",
                "algorithm": "SVTR_LCNet",
                "Backbone": {"name": "PPLCNetV4", "model_size": "small"},
                "Head": {
                    "name": "CTCHead",
                    "out_channels": vocab_size,
                    "Neck": {"name": "lightsvtr", "dims": 120, "depth": 2, "mlp_ratio": 2.0, "local_kernel": 7},
                    "Head": {"fc_decay": 1e-05}
                }
            }
        else:
            arch_cfg = {
                "model_type": "rec",
                "algorithm": "SVTR_LCNet",
                "Backbone": {"name": "PPLCNetV4", "model_size": "small"},
                "Head": {
                    "name": "MultiHeadEditRefineNRTR",
                    "nrtr_dim": 384,
                    "nrtr_layers": 4,
                    "length_hidden": 128,
                    "length_max": 25,
                    "edit_mode": "token_refine",
                    "use_highres_visual": True,
                    "use_length_head": True,
                    "use_explicit_change_head": model_meta["use_explicit"],
                    "use_align_guided_cross_attn": model_meta["align_guided"],
                    "align_spatial_weight": 1.0,
                    "align_spatial_sigma": 0.15,
                    "out_channels_list": {
                        "CTCLabelDecode": vocab_size,
                        "NRTRLabelDecode": vocab_size + 3
                    },
                    "head_list": [
                        {
                            "CTCHead": {
                                "Neck": {"name": "lightsvtr", "dims": 120, "depth": 2, "mlp_ratio": 2.0, "local_kernel": 7},
                                "Head": {"fc_decay": 1e-05}
                            }
                        },
                        {
                            "NRTRHead": {"nrtr_dim": 384, "max_text_length": 25}
                        }
                    ]
                }
            }

        try:
            model = build_model(arch_cfg)
            model.eval()
            state_dict = paddle.load(cp_file)
            model.set_state_dict(state_dict)
        except Exception as e:
            log(f"[!] Model build/load failed for {m_name}: {e}")
            continue

        correct_count = 0
        total_chars = 0
        total_char_dist = 0
        model_errors = []
        
        t0 = time.time()
        
        for i in range(0, total_images, batch_size):
            batch_slice = preprocessed_tensors[i:i + batch_size]
            batch_samples = valid_samples[i:i + batch_size]
            
            inp = paddle.to_tensor(np.array(batch_slice, dtype="float32"))
            with paddle.no_grad():
                preds = model(inp)
                
            ctc_out = preds["ctc"] if isinstance(preds, dict) and "ctc" in preds else (preds[0] if isinstance(preds, (list, tuple)) else preds)
            ctc_probs = paddle.nn.functional.softmax(ctc_out, axis=2).numpy()
            
            for b_idx, s in enumerate(batch_samples):
                _, img_name, gt_text = s
                prob_mat = ctc_probs[b_idx]
                pred_indices = np.argmax(prob_mat, axis=1)
                
                # CTC decode
                pred_chars = []
                prev_idx = -1
                for idx in pred_indices:
                    if idx != 0 and idx != prev_idx and (idx - 1) < len(char_list):
                        pred_chars.append(char_list[idx - 1])
                    prev_idx = idx
                ctc_pred = "".join(pred_chars)
                
                # Refine Gating if available
                final_pred = ctc_pred
                if isinstance(preds, dict) and "change_gate" in preds and "edit_tok" in preds:
                    gate_logits = preds["change_gate"][b_idx].numpy()
                    edit_tok_logits = preds["edit_tok"][b_idx].numpy()
                    gate_prob = 1.0 / (1.0 + np.exp(-gate_logits))
                    refined_chars = list(ctc_pred)
                    for pos in range(min(len(refined_chars), len(gate_prob))):
                        if gate_prob[pos] >= 0.5:
                            best_tok_idx = np.argmax(edit_tok_logits[pos])
                            if best_tok_idx < len(char_list):
                                refined_chars[pos] = char_list[best_tok_idx]
                    final_pred = "".join(refined_chars)

                # Accuracy & CER calculation
                dist = compute_levenshtein(gt_text, final_pred)
                total_char_dist += dist
                total_chars += len(gt_text)
                
                if final_pred == gt_text:
                    correct_count += 1
                else:
                    model_errors.append({
                        "image": img_name,
                        "ground_truth": gt_text,
                        "prediction": final_pred,
                        "ctc_prediction": ctc_pred
                    })

        t_elapsed = time.time() - t0
        acc = (correct_count / total_images) * 100.0
        cer = (total_char_dist / total_chars) * 100.0 if total_chars > 0 else 0.0
        fps = total_images / t_elapsed if t_elapsed > 0 else 0.0

        benchmark_results[m_name] = {
            "checkpoint_file": os.path.basename(cp_file),
            "exact_accuracy": round(acc, 2),
            "cer": round(cer, 2),
            "correct_matches": correct_count,
            "total_images": total_images,
            "fps": round(fps, 1),
            "latency_ms": round(1000.0 / fps, 1) if fps > 0 else 0.0,
            "errors_count": len(model_errors),
            "sample_errors": model_errors[:10]
        }

        log(f"  --> Acc: {acc:6.2f}% ({correct_count}/{total_images}) | CER: {cer:5.2f}% | Latency: {1000.0/fps:.1f}ms ({fps:.1f} FPS)")

    # 4. Print Ranked Leaderboard
    log("\n=================================================================")
    log("🏆 LIVE BENCHMARK RANKING ON INDOMAIN CURATED (585 TEST SAMPLES):")
    log("=================================================================")
    sorted_ranks = sorted(benchmark_results.items(), key=lambda x: -x[1]["exact_accuracy"])
    for rank, (name, metrics) in enumerate(sorted_ranks, start=1):
        log(f" #{rank:2d} | {name:40s} | Acc: {metrics['exact_accuracy']:6.2f}% | CER: {metrics['cer']:5.2f}% | Latency: {metrics['latency_ms']}ms")

    # 5. Save Full Audit Log to Drive
    drive_log_dir = "/content/drive/MyDrive/research/EditCTC/logs/2026-10-04"
    os.makedirs(drive_log_dir, exist_ok=True)
    out_json = os.path.join(drive_log_dir, "all_models_live_indomain_eval.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "eval_title": "Master Live Checkpoint Evaluation (Indomain Curated 585)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "accelerator": "NVIDIA T4 GPU",
            "total_samples": total_images,
            "results": benchmark_results
        }, f, indent=2, ensure_ascii=False)

    log(f"\n[+] Saved full master evaluation results to: {out_json}")
    log("=================================================================")
    return "\n".join(logs)

if __name__ == "__main__":
    out = run_master_checkpoints_eval()
    out
