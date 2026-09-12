#!/usr/bin/env python3
"""Build normalized collapsed CTC token spans for online blur augmentation.

The recognizer is frozen and used only to produce frame argmax spans.  The
result is an index consumed by ``CTCSpanBlurAug``; no Cross-data path is read
by this tool.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import paddle
import yaml

from ppocr.data.imaug import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model


def make_model(cfg, checkpoint):
    global_cfg = cfg["Global"]
    post = build_post_process(cfg["PostProcess"], global_cfg)
    char_num = len(post.character)
    head = cfg["Architecture"]["Head"]
    if head["name"].startswith("MultiHead"):
        head["out_channels_list"] = {
            "CTCLabelDecode": char_num,
            "SARLabelDecode": char_num + 2,
            "NRTRLabelDecode": char_num + 3,
        }
    else:
        head["out_channels"] = char_num
    cfg["Global"]["checkpoints"] = checkpoint
    cfg["Global"]["pretrained_model"] = None
    model = build_model(cfg["Architecture"])
    load_model(cfg, model)
    model.eval()
    return model


def make_ops(cfg):
    ops_cfg = []
    for item in cfg["Eval"]["dataset"]["transforms"]:
        name = next(iter(item))
        if "Label" in name:
            continue
        item = {name: dict(item[name] or {})}
        if name == "RecResizeImg":
            item[name]["infer_mode"] = True
        if name == "KeepKeys":
            item[name]["keep_keys"] = ["image"]
        ops_cfg.append(item)
    return create_operators(ops_cfg, cfg["Global"])


def collapse_spans(frame_ids, blank=0):
    spans = []
    prev = blank
    start = None
    total = max(len(frame_ids), 1)
    for t, token in enumerate(frame_ids):
        token = int(token)
        if token == blank:
            if start is not None:
                spans.append([start / total, t / total])
                start = None
            prev = blank
            continue
        if token != prev:
            if start is not None:
                spans.append([start / total, t / total])
            start = t
        prev = token
    if start is not None:
        spans.append([start / total, 1.0])
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--label-file", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    paddle.set_device("gpu" if paddle.is_compiled_with_cuda() else "cpu")
    model = make_model(cfg, args.checkpoint)
    ops = make_ops(cfg)

    rows = []
    with open(args.label_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            name, gt = line.rstrip("\n").split("\t", 1)
            path = Path(args.data_dir) / name
            if path.is_file():
                rows.append((name, gt, path))

    out = {}
    for offset in range(0, len(rows), args.batch_size):
        batch_rows = rows[offset : offset + args.batch_size]
        images = []
        for name, gt, path in batch_rows:
            data = {"img_path": str(path), "image": path.read_bytes()}
            data = transform(data, ops)
            if data is not None:
                images.append(np.asarray(data[0], dtype="float32"))
        if not images:
            continue
        x = paddle.to_tensor(np.stack(images, axis=0))
        with paddle.no_grad():
            feat = model.backbone(x)
            if model.use_neck:
                feat = model.neck(feat)
            enc = model.head.ctc_encoder(feat)
            probs = model.head.ctc_head(enc)
        ids = probs.argmax(axis=-1).numpy()
        for j, (name, gt, path) in enumerate(batch_rows[: len(ids)]):
            spans = collapse_spans(ids[j].tolist())
            out[name] = {"spans": spans, "ctc_tokens": len(spans), "label": gt}
        if (offset // args.batch_size) % 10 == 0:
            print(f"processed {min(offset + len(batch_rows), len(rows))}/{len(rows)}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "indexed": len(out), "output": args.output}))


if __name__ == "__main__":
    main()
