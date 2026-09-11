#!/usr/bin/env python3
"""Stage-0 text pretraining for the existing EditRefineDecoder.

This stage teaches the decoder the mechanics of KEEP/REPLACE/DELETE/INSERT on
confusion-aware numeric strings.  It intentionally uses a learned null visual
memory and never claims visual correction transfer.  The saved tensors match
``head.edit_refine_head.*`` so a later visual stage can load them additively.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import paddle

from ppocr.losses.rec_edit_loss import EditLoss
from ppocr.modeling.heads.rec_edit_refine_head import EditRefineDecoder


DIGIT_IDS = {str(i): 33 + i for i in range(10)}


class TextDataset(paddle.io.Dataset):
    def __init__(self, rows, max_len):
        self.rows = rows
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        seed = [DIGIT_IDS[x] for x in row["seed"] if x in DIGIT_IDS]
        gt = [DIGIT_IDS[x] for x in row["gt"] if x in DIGIT_IDS]
        seed = seed[: self.max_len]
        gt = gt[: self.max_len]
        seed_pad = np.zeros((self.max_len,), dtype="int64")
        gt_pad = np.zeros((self.max_len,), dtype="int64")
        seed_pad[: len(seed)] = seed
        gt_pad[: len(gt)] = gt
        return seed_pad, np.int64(len(seed)), gt_pad, np.int64(len(gt))


def read_rows(path: Path, max_samples: int | None):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("seed", "").isdigit() and row.get("gt", "").isdigit():
            rows.append(row)
        if max_samples and len(rows) >= max_samples:
            break
    if not rows:
        raise ValueError("manifest contains no numeric rows")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="checkpoint prefix")
    ap.add_argument("--max-samples", type=int, default=200_000)
    ap.add_argument("--max-len", type=int, default=25)
    ap.add_argument("--edit-dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=4035)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)
    paddle.set_device("gpu" if paddle.is_compiled_with_cuda() else "cpu")

    rows = read_rows(args.manifest, args.max_samples)
    train_rows = [r for r in rows if r.get("split") != "val"]
    val_rows = [r for r in rows if r.get("split") == "val"]
    # Edit class weights prevent the per-position KEEP majority from erasing
    # the operation mechanics.  Transfer gating is still done on natural
    # seeds later, never on this synthetic validation split.
    loss_fn = EditLoss(
        max_length=args.max_len,
        op_class_weights=[0.25, 2.0, 2.0, 2.0],
    )
    model = EditRefineDecoder(
        in_channels=args.edit_dim,
        vocab_size=18709,
        edit_dim=args.edit_dim,
        num_layers=args.layers,
        nhead=args.nhead,
        max_seed_len=args.max_len,
        length_max=args.max_len,
        dropout=0.1,
        head_dropout=0.1,
        use_cross_attn=False,
    )
    loader = paddle.io.DataLoader(
        TextDataset(train_rows, args.max_len),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    optimizer = paddle.optimizer.Adam(learning_rate=args.lr, parameters=model.parameters())
    history = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for seed_ids, seed_lens, gt_ids, gt_lens in loader:
            pred = model(
                memory=None,
                seed_ids=seed_ids,
                seed_lens=seed_lens,
            )
            op_t, tok_t = loss_fn.build_targets(
                seed_ids.numpy(), seed_lens.numpy(), gt_ids.numpy(), gt_lens.numpy()
            )
            loss = loss_fn(pred, (gt_ids, gt_lens, op_t, tok_t))["loss"]
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            total += float(loss)
        mean_loss = total / max(len(loader), 1)
        history.append({"epoch": epoch + 1, "loss": mean_loss})
        print(f"epoch={epoch + 1}/{args.epochs} loss={mean_loss:.5f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    state = {"head.edit_refine_head." + k: v for k, v in model.state_dict().items()}
    paddle.save(state, str(args.out) + ".pdparams")
    (args.out.parent / (args.out.name + ".json")).write_text(
        json.dumps({
            "stage": "text_mechanics",
            "rows": len(rows),
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "history": history,
            "use_cross_attn": False,
            "cross_data_used_for_training": False,
            "note": "mechanics pretraining only; visual transfer must be gated separately",
        }, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
