#!/usr/bin/env python3
"""Adapt the integrated NRTR editor on cached natural CTC seeds.

The CTC/backbone and NRTR decoder stay frozen.  Unlike joint training, the
editor receives the seed recorded by an out-of-sample/frozen CTC audit, so the
training distribution is the actual natural CTC error distribution.
"""
from __future__ import absolute_import, division, print_function

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import paddle
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppocr.losses.rec_edit_loss_token_refine import EditLossTokenRefine
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model


DIGIT_IDS = {str(i): 33 + i for i in range(10)}


class NaturalDataset(paddle.io.Dataset):
    def __init__(self, rows, max_len=25):
        self.rows, self.max_len = rows, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = cv2.imread(row["image"], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(row["image"])
        image = cv2.resize(image, (320, 48), interpolation=cv2.INTER_AREA)
        image = ((image.astype("float32") / 255.0) - 0.5) / 0.5
        image = image.transpose(2, 0, 1)
        seed = [DIGIT_IDS[c] for c in row["seed"] if c in DIGIT_IDS][: self.max_len]
        gt = [DIGIT_IDS[c] for c in row["gt"] if c in DIGIT_IDS][: self.max_len]
        s, g = np.zeros(self.max_len, "int64"), np.zeros(self.max_len, "int64")
        s[: len(seed)], g[: len(gt)] = seed, gt
        return image, s, np.int64(len(seed)), g, np.int64(len(gt))


def load_rows(path, positive_repeat):
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    train = [r for r in rows if r.get("split") == "train" and r.get("wrong_seed")]
    # Keep all hard KEEP rows, but oversample real errors.  Same-length errors
    # are especially useful for this substitution-only token head.
    hard = [r for r in rows if r.get("split") == "train" and not r.get("wrong_seed")]
    train = hard + train * int(positive_repeat)
    return train, [r for r in rows if r.get("split") == "val"]


def build_cfg(path):
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["Global"]["distributed"] = False
    post = build_post_process(cfg["PostProcess"], cfg["Global"])
    chars = len(getattr(post, "character"))
    cfg["Architecture"]["Head"]["out_channels_list"] = {
        "CTCLabelDecode": chars, "NRTRLabelDecode": chars + 3
    }
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--positive-repeat", type=int, default=20)
    args = ap.parse_args()
    paddle.set_device("gpu" if paddle.is_compiled_with_cuda() else "cpu")
    cfg = build_cfg(args.config)
    cfg["Global"]["checkpoints"] = str(args.checkpoint)
    cfg["Global"]["pretrained_model"] = None
    model = build_model(cfg["Architecture"])
    load_model(cfg, model)
    model.eval()
    head = model.head
    for p in model.parameters():
        p.stop_gradient = True
    for module in (head.edit_op_head, head.edit_tok_head):
        for p in module.parameters():
            p.stop_gradient = False
    edit_params = list(head.edit_op_head.parameters()) + list(head.edit_tok_head.parameters())
    opt = paddle.optimizer.Adam(learning_rate=args.lr, parameters=edit_params)
    loss_fn = EditLossTokenRefine(max_length=25)
    train_rows, val_rows = load_rows(args.manifest, args.positive_repeat)
    loader = paddle.io.DataLoader(NaturalDataset(train_rows), batch_size=args.batch_size,
                                  shuffle=True, drop_last=True, num_workers=0)
    history = []
    for epoch in range(args.epochs):
        total = 0.0
        head.train()
        for images, seed_ids, seed_lens, gt_ids, gt_lens in loader:
            with paddle.no_grad():
                feat = model.backbone(images)
                if head.use_pool:
                    feat = head.pool(feat.reshape([0, 3, -1, head.in_channels]).transpose([0, 3, 1, 2]))
                ctc_memory, memory = head._memory(feat, images)
                length_logits = head.length_head(ctc_memory) if head.use_length_head else None
            edit = head._edit_forward(None, memory, length_logits, seed_ids, seed_lens)
            result = loss_fn(edit, (gt_ids, gt_lens))
            result["loss"].backward()
            opt.step(); opt.clear_grad()
            total += float(result["loss"])
        value = total / max(len(loader), 1)
        history.append({"epoch": epoch + 1, "loss": value})
        print("epoch={}/{} loss={:.6f}".format(epoch + 1, args.epochs, value), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Save an additive checkpoint with only the edit tensors.  The loader in
    # infer_rec.py applies matching keys after the complete frozen CTC model.
    state = {"head." + k: v for k, v in head.state_dict().items()
             if k.startswith("edit_op_head.") or k.startswith("edit_tok_head.")}
    paddle.save(state, str(args.out) + ".pdparams")
    (args.out.parent / (args.out.name + ".json")).write_text(json.dumps({
        "experiment": "natural_cached_seed_token_refine", "train_rows": len(train_rows),
        "val_rows": len(val_rows), "positive_repeat": args.positive_repeat,
        "frozen_ctc_backbone": True, "cross_data_used_for_training": False,
        "history": history,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
