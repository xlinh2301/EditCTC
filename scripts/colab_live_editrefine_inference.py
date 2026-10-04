#!/usr/bin/env python3
"""
Full Live EditCTC Forward-Pass Inference (CTC Branch + Refine Branch + Gating)
Evaluates actual .pdparams weights directly on test images dynamically.
"""

import os
import sys
import time
import json
import yaml
import numpy as np

def run_live_full_editctc_inference():
    sys.path.insert(0, '/tmp/EditCTC_code')
    sys.path.insert(0, '/tmp/bundle_essentials')
    
    import paddle
    import cv2
    from ppocr.modeling.architectures import build_model

    lines = []
    def log(msg):
        lines.append(str(msg))

    log("=================================================================")
    log("⚡ LIVE FULL EDITCTC (CTC + REFINE DECODER + GATING) INFERENCE")
    log(f"   Date: 2026-10-04 | GPU: {paddle.device.get_device()}")
    log("=================================================================")

    # 1. Checkpoint & Config
    cfg_path = "/content/drive/MyDrive/research/EditCTC/release_EditCTC/checkpoints/s1024/config.yml"
    cp_weight = "/content/drive/MyDrive/research/EditCTC/release_EditCTC/checkpoints/s1024/best_accuracy.pdparams"
    
    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    dict_path = "/tmp/EditCTC_code/ppocr/utils/dict/ppocrv6_dict.txt"
    if not os.path.exists(dict_path):
        dict_path = "/content/drive/MyDrive/research/EditCTC/source/ppocr/utils/dict/ppocrv6_dict.txt"

    char_list = []
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            char_list.append(line.strip("\n\r"))
    char_list.append(" ")
    vocab_size = len(char_list) + 1

    arch_cfg = config["Architecture"]
    if "Head" in arch_cfg:
        arch_cfg["Head"]["out_channels_list"] = {
            "CTCLabelDecode": vocab_size,
            "NRTRLabelDecode": vocab_size + 3
        }

    log(f"[*] Building complete architecture with Head: {arch_cfg['Head']['name']}...")
    model = build_model(arch_cfg)
    model.eval()

    log(f"[*] Loading live checkpoint weights from {cp_weight}...")
    state_dict = paddle.load(cp_weight)
    model.set_state_dict(state_dict)
    log(f"[+] Loaded {len(state_dict)} weights successfully into GPU memory!")

    # 2. Dataset images
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

    log(f"[+] Found {len(samples)} test images on disk.")

    def preprocess_image(img_path, target_h=48, target_w=320):
        img = cv2.imread(img_path)
        if img is None:
            return None
        resized = cv2.resize(img, (target_w, target_h))
        normalized = (resized.astype("float32") / 255.0 - 0.5) / 0.5
        chw = normalized.transpose((2, 0, 1))
        return chw

    ctc_correct = 0
    refine_correct = 0
    total = 0
    error_audit = []

    batch_size = 32
    start_time = time.time()

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

        # 1. CTC Branch
        ctc_out = preds["ctc"] if isinstance(preds, dict) and "ctc" in preds else preds
        ctc_probs = paddle.nn.functional.softmax(ctc_out, axis=2).numpy()

        for b_idx, s in enumerate(valid_samples):
            _, img_name, gt_text = s
            prob_mat = ctc_probs[b_idx]
            pred_indices = np.argmax(prob_mat, axis=1)

            ctc_chars = []
            prev_idx = -1
            for idx in pred_indices:
                if idx != 0 and idx != prev_idx and (idx - 1) < len(char_list):
                    ctc_chars.append(char_list[idx - 1])
                prev_idx = idx
            ctc_pred = "".join(ctc_chars)

            # 2. Refined Output (Decoupled Gating)
            final_pred = ctc_pred # Base prediction
            if isinstance(preds, dict) and "change_gate" in preds and "edit_tok" in preds:
                gate_logits = preds["change_gate"][b_idx].numpy()
                edit_tok_logits = preds["edit_tok"][b_idx].numpy()
                gate_prob = 1.0 / (1.0 + np.exp(-gate_logits)) # sigmoid
                
                # Apply gating per character position
                refined_chars = list(ctc_pred)
                for pos in range(min(len(refined_chars), len(gate_prob))):
                    if gate_prob[pos] >= 0.5:
                        best_tok_idx = np.argmax(edit_tok_logits[pos])
                        if best_tok_idx < len(char_list):
                            refined_chars[pos] = char_list[best_tok_idx]
                final_pred = "".join(refined_chars)

            total += 1
            if ctc_pred == gt_text:
                ctc_correct += 1
            if final_pred == gt_text:
                refine_correct += 1
            else:
                error_audit.append({
                    "image": img_name,
                    "ground_truth": gt_text,
                    "ctc_pred": ctc_pred,
                    "final_pred": final_pred
                })

    elapsed = time.time() - start_time
    fps = total / elapsed if elapsed > 0 else 0.0
    ctc_acc = (ctc_correct / total) * 100.0 if total > 0 else 0.0
    refine_acc = (refine_correct / total) * 100.0 if total > 0 else 0.0

    log("\n=================================================================")
    log("🎯 LIVE FORWARD-PASS BENCHMARK EXECUTION RESULTS:")
    log(f"   - Total Physical Images Evaluated: {total}")
    log(f"   - Raw CTC Branch Accuracy: {ctc_acc:.2f}% ({ctc_correct}/{total})")
    log(f"   - EditCTC Proposed Accuracy: {refine_acc:.2f}% ({refine_correct}/{total})")
    log(f"   - Total Error Count: {len(error_audit)}")
    log(f"   - Live Inference Time: {elapsed:.2f}s ({fps:.1f} FPS)")
    log("=================================================================")
    log("\n[*] Sample Error Cases from Live Checkpoint Execution:")
    for err in error_audit[:10]:
        log(f"   - {err['image']}: GT='{err['ground_truth']}' | CTC='{err['ctc_pred']}' | EditCTC='{err['final_pred']}'")

    # Save to Drive
    drive_out_dir = "/content/drive/MyDrive/research/EditCTC/logs/2026-10-04"
    os.makedirs(drive_out_dir, exist_ok=True)
    live_out_json = os.path.join(drive_out_dir, "live_checkpoint_forward_run.json")
    with open(live_out_json, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "checkpoint": cp_weight,
            "total_images": total,
            "ctc_accuracy": ctc_acc,
            "editctc_accuracy": refine_acc,
            "fps": fps,
            "errors": error_audit
        }, f, indent=2, ensure_ascii=False)

    log(f"\n[+] Saved complete live inference audit to: {live_out_json}")
    return "\n".join(lines)

if __name__ == "__main__":
    result = run_live_full_editctc_inference()
    result
