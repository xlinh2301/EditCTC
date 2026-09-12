#!/usr/bin/env python3
"""Evaluate E37 verifier with CTC Top-2 replacement proposals."""
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
from ppocr.modeling.heads.rec_edit_refine_nrtr_head import ctc_seed_and_margin
from ppocr.modeling.heads.sequence_verifier import SequenceVisualVerifier
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model


DIGIT_IDS = {str(i): 33 + i for i in range(10)}
ID_TO_DIGIT = {v: k for k, v in DIGIT_IDS.items()}


def encode(text, max_length):
    ids = [DIGIT_IDS[c] for c in text if c in DIGIT_IDS][:max_length]
    arr = np.zeros(max_length, dtype="int64")
    arr[: len(ids)] = ids
    return arr, len(ids)


def decode(ids):
    return "".join(ID_TO_DIGIT.get(int(x), "") for x in ids if int(x) > 0)


class ImageRows(paddle.io.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = cv2.imread(row["image"], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(row["image"])
        image = cv2.resize(image, (320, 48), interpolation=cv2.INTER_AREA)
        image = ((image.astype("float32") / 255.0) - 0.5) / 0.5
        return image.transpose(2, 0, 1), row["name"], row["gt"]


def collate(batch):
    return (
        paddle.to_tensor(np.stack([x[0] for x in batch]), dtype="float32"),
        [x[1] for x in batch],
        [x[2] for x in batch],
    )


def build_cfg(path):
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["Global"]["distributed"] = False
    post = build_post_process(cfg["PostProcess"], cfg["Global"])
    chars = len(getattr(post, "character"))
    cfg["Architecture"]["Head"]["out_channels_list"] = {
        "CTCLabelDecode": chars, "NRTRLabelDecode": chars + 3,
    }
    return cfg, chars


def make_rows(image_dir, label_file, max_length):
    rows, missing = [], []
    for line in Path(label_file).read_text().splitlines():
        if not line.strip():
            continue
        name, gt = line.split("\t", 1)
        path = Path(image_dir) / name
        if not path.exists():
            missing.append(name)
            continue
        rows.append({"image": str(path), "name": Path(name).name,
                     "gt": gt[:max_length]})
    return rows, missing


def build_candidates(seeds, alternatives, lens, max_length, max_candidates):
    bsz = len(lens)
    ids = np.zeros((bsz, max_candidates, max_length), dtype="int64")
    lengths = np.zeros((bsz, max_candidates), dtype="int64")
    for b, n_raw in enumerate(lens):
        n = min(int(n_raw), max_length)
        seed = seeds[b, :n].copy()
        ids[b, 0, :n], lengths[b, 0] = seed, n
        slot = 1
        for pos in range(n):
            alt = int(alternatives[b, pos])
            if slot >= max_candidates or alt <= 0 or alt == int(seed[pos]):
                continue
            cand = seed.copy()
            cand[pos] = alt
            ids[b, slot, :n], lengths[b, slot] = cand, n
            slot += 1
        while slot < max_candidates:
            ids[b, slot, :n], lengths[b, slot] = seed, n
            slot += 1
    return ids, lengths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--verifier", type=Path, required=True)
    ap.add_argument("--image-dir", type=Path, required=True)
    ap.add_argument("--label-file", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=25)
    ap.add_argument("--max-candidates", type=int, default=10)
    ap.add_argument("--thresholds", type=str, default="0,0.05,0.1,0.2,0.5")
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
        max_length=args.max_length, layers=2, heads=4, dropout=0.0,
    )
    verifier.set_state_dict(paddle.load(str(args.verifier) + ".pdparams"))
    verifier.eval()
    rows, missing = make_rows(args.image_dir, args.label_file, args.max_length)
    loader = paddle.io.DataLoader(
        ImageRows(rows), batch_size=args.batch_size, shuffle=False,
        drop_last=False, num_workers=0, collate_fn=collate,
    )
    records = []
    head = model.head
    with paddle.no_grad():
        for images, names, gts in loader:
            fmap = model.backbone(images)
            feature_map = model.backbone.recon_feat.detach()
            ctc_memory = head.ctc_encoder(fmap)
            ctc_logits = head.ctc_head.fc(ctc_memory)
            seeds, lens, _, alternatives = ctc_seed_and_margin(ctc_logits, args.max_length)
            cand_ids, cand_lens = build_candidates(
                seeds, alternatives, lens, args.max_length, args.max_candidates
            )
            bsz = images.shape[0]
            feat = feature_map.unsqueeze(1).tile([1, args.max_candidates, 1, 1, 1])
            feat = feat.reshape([bsz * args.max_candidates] + list(feat.shape[2:]))
            flat_ids = paddle.to_tensor(cand_ids.reshape([bsz * args.max_candidates, args.max_length]))
            flat_lens = paddle.to_tensor(cand_lens.reshape([bsz * args.max_candidates]))
            scores = verifier(feat, flat_ids, flat_lens).reshape([bsz, args.max_candidates]).numpy()
            for i, name in enumerate(names):
                records.append({
                    "file": name, "gt": gts[i],
                    "seed": decode(seeds[i, : int(lens[i])]),
                    "candidate_texts": [decode(cand_ids[i, k]) for k in range(args.max_candidates)],
                    "scores": scores[i].tolist(),
                })
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    summaries = []
    for threshold in thresholds:
        m = {"threshold": threshold, "evaluated": len(records), "ctc_correct": 0,
             "final_correct": 0, "changed": 0, "helped": 0, "hurt": 0}
        for row in records:
            best = int(np.argmax(row["scores"]))
            seed, candidate = row["seed"], row["candidate_texts"][best]
            final = candidate if row["scores"][best] - row["scores"][0] > threshold else seed
            m["ctc_correct"] += seed == row["gt"]
            m["final_correct"] += final == row["gt"]
            m["changed"] += final != seed
            m["helped"] += seed != row["gt"] and final == row["gt"]
            m["hurt"] += seed == row["gt"] and final != row["gt"]
        d = max(m["evaluated"], 1)
        m["ctc_accuracy"] = m["ctc_correct"] / d
        m["final_accuracy"] = m["final_correct"] / d
        summaries.append(m)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summaries": summaries, "missing_files": missing,
                                    "records": records}, indent=2) + "\n")
    print(json.dumps({"summaries": summaries, "missing_files": missing}, indent=2))


if __name__ == "__main__":
    main()
