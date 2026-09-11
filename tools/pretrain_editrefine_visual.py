#!/usr/bin/env python3
"""Stage-1 visual pretraining for NERD with explicit synthetic CTC seeds.

The CTC/backbone stays frozen.  The edit decoder receives the rendered image
features and the manifest's explicit seed, rather than silently rebuilding a
seed from the clean image's CTC logits.  This is a standalone GPU job because
the normal joint training path intentionally derives seeds internally.
"""
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
from ppocr.utils.save_load import load_model, load_pretrained_params


DIGIT_IDS = {str(i): 33 + i for i in range(10)}


class VisualDataset(paddle.io.Dataset):
    def __init__(self, rows, root, max_len):
        self.rows, self.root, self.max_len = rows, root, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.root / row["image"]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.resize(image, (320, 48), interpolation=cv2.INTER_AREA)
        image = image.astype("float32") / 255.0
        image = (image - 0.5) / 0.5
        image = image.transpose(2, 0, 1)
        seed = [DIGIT_IDS[c] for c in row["seed"] if c in DIGIT_IDS][: self.max_len]
        gt = [DIGIT_IDS[c] for c in row["gt"] if c in DIGIT_IDS][: self.max_len]
        seed_pad = np.zeros((self.max_len,), dtype="int64")
        gt_pad = np.zeros((self.max_len,), dtype="int64")
        seed_pad[: len(seed)] = seed
        gt_pad[: len(gt)] = gt
        return image, seed_pad, np.int64(len(seed)), gt_pad, np.int64(len(gt))


def rows_from_manifest(path: Path, root: Path, max_samples: int):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (root / row["image"]).exists() and row.get("seed", "").isdigit():
            rows.append(row)
        if max_samples and len(rows) >= max_samples:
            break
    return rows


def build_cfg(path: Path):
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["Global"]["distributed"] = False
    post = build_post_process(cfg["PostProcess"], cfg["Global"])
    chars = len(getattr(post, "character"))
    cfg["Architecture"]["Head"]["out_channels_list"] = {
        "CTCLabelDecode": chars,
        "NRTRLabelDecode": chars + 3,
    }
    return cfg, chars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--text-checkpoint", type=Path, default=None)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-samples", type=int, default=200_000)
    ap.add_argument("--max-len", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    paddle.set_device("gpu" if paddle.is_compiled_with_cuda() else "cpu")
    cfg, vocab_size = build_cfg(args.config)
    cfg["Global"]["checkpoints"] = str(args.checkpoint)
    cfg["Global"]["pretrained_model"] = None
    model = build_model(cfg["Architecture"])
    load_model(cfg, model)
    if args.text_checkpoint:
        load_pretrained_params(model, str(args.text_checkpoint))
    for p in model.parameters():
        p.stop_gradient = True
    edit = model.head.edit_refine_head
    for p in edit.parameters():
        p.stop_gradient = False
    loss_fn = EditLoss(max_length=args.max_len, op_class_weights=[0.25, 2.0, 2.0, 2.0])
    root = args.manifest.parent
    rows = rows_from_manifest(args.manifest, root, args.max_samples)
    train_rows = [r for r in rows if r.get("split") != "val"]
    val_rows = [r for r in rows if r.get("split") == "val"]
    loader = paddle.io.DataLoader(VisualDataset(train_rows, root, args.max_len), batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    optimizer = paddle.optimizer.Adam(learning_rate=args.lr, parameters=edit.parameters())
    history = []
    model.eval()
    for epoch in range(args.epochs):
        edit.train()
        total = 0.0
        for images, seed_ids, seed_lens, gt_ids, gt_lens in loader:
            with paddle.no_grad():
                feat = model.backbone(images)
                head = model.head
                if head.use_pool:
                    feat = head.pool(feat.reshape([0, 3, -1, head.in_channels]).transpose([0, 3, 1, 2]))
                memory = head.ctc_encoder(feat)
                length_logits = head.length_head(memory) if head.use_length_head else None
            pred = edit(memory=memory, length_logits=length_logits, seed_ids=seed_ids, seed_lens=seed_lens)
            op_t, tok_t = loss_fn.build_targets(seed_ids.numpy(), seed_lens.numpy(), gt_ids.numpy(), gt_lens.numpy())
            loss = loss_fn(pred, (gt_ids, gt_lens, op_t, tok_t))["loss"]
            loss.backward()
            optimizer.step(); optimizer.clear_grad()
            total += float(loss)
        value = total / max(len(loader), 1)
        history.append({"epoch": epoch + 1, "loss": value})
        print(f"epoch={epoch + 1}/{args.epochs} loss={value:.5f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    state = {"head.edit_refine_head." + k: v for k, v in edit.state_dict().items()}
    paddle.save(state, str(args.out) + ".pdparams")
    (args.out.parent / (args.out.name + ".json")).write_text(json.dumps({
        "stage": "visual_synthetic_edit_pretrain", "rows": len(rows),
        "train_rows": len(train_rows), "val_rows": len(val_rows),
        "history": history, "frozen_ctc": True,
        "cross_data_used_for_training": False,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
