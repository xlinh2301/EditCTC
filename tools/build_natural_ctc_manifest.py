#!/usr/bin/env python3
"""Build an E40 bridge manifest from a frozen CTC branch audit.

The audit contains model-produced CTC seeds but intentionally does not carry
labels.  This utility joins the locked Indomain train labels, keeps authentic
wrong seeds plus low-confidence correct seeds as hard KEEP examples, and
writes absolute image paths so the visual pretrainer can consume the bank.
Cross data is rejected by an explicit path guard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def labels(path: Path):
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name, text = line.split("\t", 1)
        out[Path(name).name] = text.strip()
    return out


def confidence(row):
    ctc = row.get("ctc", {})
    aggregate = float(ctc.get("confidence", 0.0))
    trace = ctc.get("frame_trace", [])
    margins = [float(x.get("top1_top2_margin", 1.0)) for x in trace]
    # Aggregate confidence is often saturated for CTC.  The lower tail of
    # frame margins catches ambiguous but globally correct seeds.
    low_margin = min(margins) if margins else aggregate
    return aggregate, low_margin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--image-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--keep-confidence", type=float, default=0.995)
    ap.add_argument("--keep-min-margin", type=float, default=0.80)
    args = ap.parse_args()

    # Guard against accidentally constructing a training bank from Cross.
    joined = str(args.labels).lower() + " " + str(args.image_root).lower()
    if "cross" in joined:
        raise SystemExit("refusing Cross path: natural bank is Indomain-only")
    gt_by_name = labels(args.labels)
    rows, stats = [], {
        "audit_rows": 0, "label_matches": 0, "numeric_rows": 0,
        "natural_errors": 0, "hard_keep": 0, "dropped_correct": 0,
    }
    with args.audit.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            stats["audit_rows"] += 1
            row = json.loads(line)
            name = Path(row["file"]).name
            gt = gt_by_name.get(name)
            if gt is None:
                continue
            stats["label_matches"] += 1
            seed = str(row.get("nerd", {}).get("seed_text", ""))
            if not (seed.isdigit() and gt.isdigit() and seed and gt):
                continue
            stats["numeric_rows"] += 1
            aggregate, low_margin = confidence(row)
            wrong = seed != gt
            hard_keep = (not wrong and (aggregate < args.keep_confidence or
                                        low_margin < args.keep_min_margin))
            if not wrong and not hard_keep:
                stats["dropped_correct"] += 1
                continue
            stats["natural_errors" if wrong else "hard_keep"] += 1
            image = args.image_root / name
            if not image.exists():
                continue
            # Deterministic held-out split, independent of model outputs.
            digest = hashlib.md5(name.encode("utf-8")).digest()[0]
            split = "val" if digest < 26 else "train"
            rows.append({
                "image": str(image), "seed": seed, "gt": gt,
                "ctc_text": str(row.get("ctc", {}).get("text", "")),
                "ctc_confidence": aggregate, "ctc_min_frame_margin": low_margin,
                "wrong_seed": wrong, "hard_keep": hard_keep,
                "source": "natural_ctc_indomain_train", "split": split,
            })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats.update({"kept_rows": len(rows),
                  "train_rows": sum(r["split"] == "train" for r in rows),
                  "val_rows": sum(r["split"] == "val" for r in rows),
                  "keep_confidence": args.keep_confidence,
                  "keep_min_margin": args.keep_min_margin,
                  "cross_data_used_for_training": False,
                  "labels": str(args.labels), "audit": str(args.audit)})
    summary = args.out.with_suffix(".summary.json")
    summary.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
