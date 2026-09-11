#!/usr/bin/env python3
"""Evaluate an explicit-seed visual edit checkpoint on held-out synthetic rows."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import paddle
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppocr.losses.rec_edit_loss import EditLoss
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.modeling.heads.rec_edit_refine_head import apply_edit_ops
from ppocr.utils.save_load import load_pretrained_params, load_model

DIGIT_IDS = {str(i): 33 + i for i in range(10)}
ID_DIGITS = {v: k for k, v in DIGIT_IDS.items()}


def read_rows(manifest, root, max_samples):
    rows = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split") == "val" and (root / row["image"]).exists():
            rows.append(row)
        if max_samples and len(rows) >= max_samples:
            break
    return rows


def image_tensor(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (320, 48), interpolation=cv2.INTER_AREA)
    image = (image.astype("float32") / 255.0 - 0.5) / 0.5
    return image.transpose(2, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--base-checkpoint", type=Path, required=True,
                    help="frozen CTC/backbone checkpoint used during visual pretraining")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-samples", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()
    paddle.set_device("gpu" if paddle.is_compiled_with_cuda() else "cpu")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cfg["Global"]["distributed"] = False
    post = build_post_process(cfg["PostProcess"], cfg["Global"])
    chars = len(getattr(post, "character"))
    cfg["Architecture"]["Head"]["out_channels_list"] = {
        "CTCLabelDecode": chars, "NRTRLabelDecode": chars + 3
    }
    cfg["Global"]["checkpoints"] = str(args.base_checkpoint)
    cfg["Global"]["pretrained_model"] = None
    model = build_model(cfg["Architecture"])
    load_model(cfg, model)
    load_pretrained_params(model, str(args.checkpoint))
    model.eval()
    edit = model.head.edit_refine_head
    edit.eval()
    rows = read_rows(args.manifest, args.manifest.parent, args.max_samples)
    loss_fn = EditLoss(max_length=25)
    counts = {"rows": 0, "wrong_seed": 0, "seed_exact": 0, "refined_exact": 0,
              "helped": 0, "hurt": 0, "changed": 0, "valid_ops": 0,
              "nonkeep": 0, "nonkeep_correct": 0, "keep_target": 0,
              "keep_pred": 0, "keep_true_positive": 0, "token_valid": 0,
              "token_correct": 0}
    batch = max(1, args.batch_size)
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        images, seeds, seed_lens, gts, gt_lens = [], [], [], [], []
        for row in chunk:
            s = [DIGIT_IDS[c] for c in row["seed"] if c in DIGIT_IDS][:25]
            g = [DIGIT_IDS[c] for c in row["gt"] if c in DIGIT_IDS][:25]
            sp, gp = np.zeros(25, "int64"), np.zeros(25, "int64")
            sp[:len(s)], gp[:len(g)] = s, g
            images.append(image_tensor(args.manifest.parent / row["image"]))
            seeds.append(sp); seed_lens.append(len(s)); gts.append(gp); gt_lens.append(len(g))
        images = paddle.to_tensor(np.asarray(images))
        seed_ids = paddle.to_tensor(np.asarray(seeds))
        seed_lens_t = paddle.to_tensor(np.asarray(seed_lens, "int64"))
        with paddle.no_grad():
            feat = model.backbone(images)
            head = model.head
            if head.use_pool:
                feat = head.pool(feat.reshape([0, 3, -1, head.in_channels]).transpose([0, 3, 1, 2]))
            memory = head.ctc_encoder(feat)
            length_logits = head.length_head(memory) if head.use_length_head else None
            pred = edit(memory=memory, length_logits=length_logits, seed_ids=seed_ids, seed_lens=seed_lens_t)
        gt_arr, gl_arr = np.asarray(gts), np.asarray(gt_lens, "int64")
        op_t, tok_t = loss_fn.build_targets(np.asarray(seeds), np.asarray(seed_lens, "int64"), gt_arr, gl_arr)
        op_p, tok_p = pred["op_logits"].argmax(axis=-1).numpy(), pred["tok_logits"].argmax(axis=-1).numpy()
        for b, row in enumerate(chunk):
            n, seed, gt = int(seed_lens[b]), row["seed"], row["gt"]
            refined = apply_edit_ops(np.asarray(seeds)[b, :n], op_p[b, :n], tok_p[b, :n])
            refined_text = "".join(ID_DIGITS.get(int(x), "?") for x in refined)
            counts["rows"] += 1; counts["wrong_seed"] += seed != gt; counts["seed_exact"] += seed == gt
            counts["refined_exact"] += refined_text == gt
            counts["helped"] += seed != gt and refined_text == gt
            counts["hurt"] += seed == gt and refined_text != gt
            counts["changed"] += refined_text != seed
        valid = op_t != -100; nonkeep = valid & (op_t != 0); keep = valid & (op_t == 0); tv = tok_t != -100
        counts["valid_ops"] += int(valid.sum()); counts["nonkeep"] += int(nonkeep.sum()); counts["nonkeep_correct"] += int((nonkeep & (op_p == op_t)).sum())
        counts["keep_target"] += int(keep.sum()); counts["keep_pred"] += int((valid & (op_p == 0)).sum()); counts["keep_true_positive"] += int((keep & (op_p == 0)).sum())
        counts["token_valid"] += int(tv.sum()); counts["token_correct"] += int((tv & (tok_p == tok_t)).sum())
    result = {
        "experiment": "E40-visual-synthetic-eval", "checkpoint": str(args.checkpoint),
        "base_checkpoint": str(args.base_checkpoint),
        "rows": counts["rows"], "wrong_seed": counts["wrong_seed"],
        "seed_exact": counts["seed_exact"], "refined_exact": counts["refined_exact"],
        "changed": counts["changed"], "helped": counts["helped"], "hurt": counts["hurt"],
        "nonkeep_recall": counts["nonkeep_correct"] / max(counts["nonkeep"], 1),
        "keep_precision": counts["keep_true_positive"] / max(counts["keep_pred"], 1),
        "token_accuracy": counts["token_correct"] / max(counts["token_valid"], 1),
        "cross_data_used_for_training": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
