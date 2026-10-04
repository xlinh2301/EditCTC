#!/usr/bin/env python3
"""
Live Forward-Pass Inference Engine on GPU T4
Loads checkpoint .pdparams weights, processes images dynamically from disk,
computes forward pass predictions, measures latency, and dumps live error statistics.
"""

import os
import sys
import time
import json
import numpy as np

def run_live_forward_evaluation():
    sys.path.insert(0, '/tmp/EditCTC_code')
    sys.path.insert(0, '/tmp/bundle_essentials')
    
    import paddle
    import cv2
    from ppocr.modeling.architectures import build_model
    from ppocr.utils.save_load import load_model

    lines = []
    def log(msg):
        lines.append(str(msg))

    log("=================================================================")
    log("⚡ RUNNING LIVE FORWARD PASS INFERENCE FROM CHECKPOINT WEIGHTS")
    log(f"   Date: 2026-10-04 | Device: {paddle.device.get_device()}")
    log("=================================================================")

    # 1. Checkpoint location
    cp_path = "/content/drive/MyDrive/research/EditCTC/release_EditCTC/checkpoints/s1024/best_accuracy"
    log(f"[*] Checkpoint weight path: {cp_path}.pdparams")
    
    # 2. Dictionary setup
    dict_path = "/tmp/EditCTC_code/ppocr/utils/dict/ppocrv6_dict.txt"
    if not os.path.exists(dict_path):
        dict_path = "/tmp/bundle_essentials/workdir_eval/dict/ppocrv6_dict.txt"
        
    char_list = []
    if os.path.exists(dict_path):
        with open(dict_path, "r", encoding="utf-8") as f:
            for line in f:
                char_list.append(line.strip("\n\r"))
    else:
        char_list = list("0123456789")
    char_list.append(" ")
    vocab_size = len(char_list) + 1
    log(f"[+] Character vocabulary size: {vocab_size}")

    # 3. Model Architecture Construction
    arch_cfg = {
        "model_type": "rec",
        "algorithm": "SVTR_LCNet",
        "Backbone": {
            "name": "PPLCNetV4",
            "model_size": "small"
        },
        "Head": {
            "name": "MultiHeadEditRefineNRTR",
            "nrtr_dim": 384,
            "nrtr_layers": 4,
            "length_hidden": 128,
            "length_max": 25,
            "edit_mode": "token_refine",
            "use_highres_visual": True,
            "use_length_head": True,
            "use_explicit_change_head": True,
            "use_align_guided_cross_attn": True,
            "align_spatial_weight": 1.0,
            "align_spatial_sigma": 0.15,
            "out_channels_list": {
                "CTCLabelDecode": vocab_size,
                "NRTRLabelDecode": vocab_size + 3
            },
            "head_list": [
                {
                    "CTCHead": {
                        "Neck": {
                            "name": "lightsvtr",
                            "dims": 120,
                            "depth": 2,
                            "mlp_ratio": 2.0,
                            "local_kernel": 7
                        },
                        "Head": {
                            "fc_decay": 1e-05
                        }
                    }
                },
                {
                    "NRTRHead": {
                        "nrtr_dim": 384,
                        "max_text_length": 25
                    }
                }
            ]
        }
    }

    log("[*] Building EditCTC model layers...")
    model = build_model(arch_cfg)
    model.eval()

    # Load checkpoint
    log(f"[*] Loading state_dict into model parameters...")
    state_dict = paddle.load(cp_path + ".pdparams")
    model.set_state_dict(state_dict)
    log(f"[+] Successfully loaded {len(state_dict)} parameter weights into GPU memory!")

    # 4. Prepare Dataset Images & Ground Truth
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

    log(f"[+] Loaded {len(samples)} actual image files from disk for testing.")

    # 5. Live Forward Pass Processing
    def preprocess_image(img_path, target_h=48, target_w=320):
        img = cv2.imread(img_path)
        if img is None:
            return None
        resized = cv2.resize(img, (target_w, target_h))
        normalized = (resized.astype("float32") / 255.0 - 0.5) / 0.5
        chw = normalized.transpose((2, 0, 1))
        return chw

    correct_count = 0
    total_count = 0
    error_cases = []
    batch_size = 32
    
    start_time = time.time()
    log(f"[*] Executing live forward pass inference on GPU...")

    for i in range(0, len(samples), batch_size):
        batch_slice = samples[i:i + batch_size]
        batch_tensors = []
        valid_samples = []
        
        for s in batch_slice:
            t = preprocess_image(s[0])
            if t is not None:
                batch_tensors.append(t)
                valid_samples.append(s)
                
        if not batch_tensors:
            continue
            
        inp = paddle.to_tensor(np.array(batch_tensors, dtype="float32"))
        with paddle.no_grad():
            preds = model(inp)
            
        ctc_out = preds["ctc"] if isinstance(preds, dict) and "ctc" in preds else (preds[0] if isinstance(preds, (list, tuple)) else preds)
        ctc_probs = paddle.nn.functional.softmax(ctc_out, axis=2).numpy()
        
        for b_idx, s in enumerate(valid_samples):
            _, img_name, gt_text = s
            prob_mat = ctc_probs[b_idx]
            pred_indices = np.argmax(prob_mat, axis=1)
            
            # CTC greedy decode
            pred_chars = []
            prev_idx = -1
            for idx in pred_indices:
                if idx != 0 and idx != prev_idx and (idx - 1) < len(char_list):
                    pred_chars.append(char_list[idx - 1])
                prev_idx = idx
                
            pred_text = "".join(pred_chars)
            total_count += 1
            if pred_text == gt_text:
                correct_count += 1
            else:
                error_cases.append({
                    "image": img_name,
                    "ground_truth": gt_text,
                    "prediction": pred_text
                })

    elapsed = time.time() - start_time
    live_acc = (correct_count / total_count) * 100.0 if total_count > 0 else 0.0
    fps = total_count / elapsed if elapsed > 0 else 0.0

    log("\n=================================================================")
    log(f"🎉 LIVE FORWARD-PASS EVALUATION COMPLETED:")
    log(f"   - Evaluated Images: {total_count}")
    log(f"   - Correct Matches: {correct_count}")
    log(f"   - Exact Match Accuracy: {live_acc:.2f}%")
    log(f"   - Error Count: {len(error_cases)}")
    log(f"   - Execution Time: {elapsed:.2f}s ({fps:.1f} FPS)")
    log("=================================================================")
    log("\n[*] Sample Live Error Cases (Image | GT vs PRED):")
    for err in error_cases[:12]:
        log(f"   - {err['image']}: GT='{err['ground_truth']}' -> PRED='{err['prediction']}'")

    # 6. Save Live Results to Drive
    drive_out_dir = "/content/drive/MyDrive/research/EditCTC/logs/2026-10-04"
    os.makedirs(drive_out_dir, exist_ok=True)
    live_results_path = os.path.join(drive_out_dir, "live_forward_eval_run.json")
    with open(live_results_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "checkpoint_loaded": cp_path + ".pdparams",
            "total_samples": total_count,
            "correct_samples": correct_count,
            "live_accuracy": live_acc,
            "fps": fps,
            "errors": error_cases
        }, f, indent=2, ensure_ascii=False)
        
    log(f"\n[+] Live results and error logs saved to: {live_results_path}")
    return "\n".join(lines)

if __name__ == "__main__":
    res = run_live_forward_evaluation()
    res
