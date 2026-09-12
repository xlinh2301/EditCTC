#!/usr/bin/env python3
"""Train E37's independent full-sequence visual verifier."""
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
from ppocr.modeling.architectures import build_model
from ppocr.modeling.heads.sequence_verifier import SequenceVisualVerifier
from ppocr.postprocess import build_post_process
from ppocr.data.imaug.rec_img_aug import resize_norm_img
from ppocr.utils.save_load import load_model


DIGIT_IDS = {str(i): 33 + i for i in range(10)}


def encode(text, max_length):
    ids = [DIGIT_IDS[c] for c in text if c in DIGIT_IDS][:max_length]
    out = np.zeros(max_length, dtype="int64")
    out[: len(ids)] = ids
    return out, len(ids)


class GroupDataset(paddle.io.Dataset):
    def __init__(self, groups, max_length=25, max_candidates=5):
        self.groups = groups
        self.max_length = max_length
        self.max_candidates = max_candidates

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        row = self.groups[index]
        image = cv2.imread(row["image"], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(row["image"])
        image, _ = resize_norm_img(image, [3, 48, 320], padding=True)
        ids = np.zeros((self.max_candidates, self.max_length), dtype="int64")
        lens = np.zeros(self.max_candidates, dtype="int64")
        rewards = np.full(self.max_candidates, -1.0e4, dtype="float32")
        candidates = row["candidates"][: self.max_candidates]
        for i, candidate in enumerate(candidates):
            ids[i], lens[i] = encode(candidate["text"], self.max_length)
            rewards[i] = float(candidate["delta_ed"])
        return image, ids, lens, rewards


def build_cfg(path):
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["Global"]["distributed"] = False
    post = build_post_process(cfg["PostProcess"], cfg["Global"])
    chars = len(getattr(post, "character"))
    cfg["Architecture"]["Head"]["out_channels_list"] = {
        "CTCLabelDecode": chars,
        "NRTRLabelDecode": chars + 3,
    }
    return cfg, chars


def score_groups(model, verifier, batch, max_candidates):
    images, candidate_ids, candidate_lens, rewards = batch
    with paddle.no_grad():
        model.backbone.eval()
        model.backbone(images)
        feature_map = model.backbone.recon_feat.detach()
    bsz = images.shape[0]
    feature_map = feature_map.unsqueeze(1).tile([1, max_candidates, 1, 1, 1])
    feature_map = feature_map.reshape([bsz * max_candidates] + list(feature_map.shape[2:]))
    flat_ids = candidate_ids.reshape([bsz * max_candidates, candidate_ids.shape[-1]])
    flat_lens = candidate_lens.reshape([bsz * max_candidates])
    scores = verifier(feature_map, flat_ids, flat_lens).reshape([bsz, max_candidates])
    return scores, rewards


def run_epoch(model, verifier, loader, optimizer, max_candidates, train):
    total, batches = 0.0, 0
    if train:
        verifier.train()
    else:
        verifier.eval()
    for batch in loader:
        scores, rewards = score_groups(model, verifier, batch, max_candidates)
        targets = paddle.argmax(rewards, axis=1)
        loss = paddle.nn.functional.cross_entropy(scores, targets)
        if train:
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
        total += float(loss)
        batches += 1
    return total / max(batches, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-candidates", type=int, default=5)
    args = ap.parse_args()
    paddle.set_device("gpu" if paddle.is_compiled_with_cuda() else "cpu")
    cfg, vocab_size = build_cfg(args.config)
    cfg["Global"]["checkpoints"] = str(args.checkpoint)
    cfg["Global"]["pretrained_model"] = None
    model = build_model(cfg["Architecture"])
    load_model(cfg, model)
    model.eval()
    verifier = SequenceVisualVerifier(
        in_channels=384, vocab_size=vocab_size, hidden=192,
        max_length=25, layers=2, heads=4, dropout=0.1,
    )
    for p in model.parameters():
        p.stop_gradient = True
    loader_rows = [json.loads(x) for x in args.bank.read_text().splitlines() if x.strip()]
    train_rows = [r for r in loader_rows if r.get("split") == "train"]
    val_rows = [r for r in loader_rows if r.get("split") == "val"]
    train_loader = paddle.io.DataLoader(
        GroupDataset(train_rows, max_candidates=args.max_candidates),
        batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0,
    )
    val_loader = paddle.io.DataLoader(
        GroupDataset(val_rows, max_candidates=args.max_candidates),
        batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0,
    )
    optimizer = paddle.optimizer.Adam(learning_rate=args.lr, parameters=verifier.parameters())
    history = []
    for epoch in range(args.epochs):
        train_loss = run_epoch(model, verifier, train_loader, optimizer, args.max_candidates, True)
        val_loss = run_epoch(model, verifier, val_loader, optimizer, args.max_candidates, False)
        record = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}
        history.append(record)
        print("epoch={}/{} train={:.6f} val={:.6f}".format(epoch + 1, args.epochs, train_loss, val_loss), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    paddle.save(verifier.state_dict(), str(args.out) + ".pdparams")
    (args.out.parent / (args.out.name + ".json")).write_text(json.dumps({
        "experiment": "E37_sequence_visual_verifier",
        "train_groups": len(train_rows), "val_groups": len(val_rows),
        "max_candidates": args.max_candidates, "frozen_ctc_backbone": True,
        "cross_data_used_for_training": False, "history": history,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
