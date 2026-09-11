#!/usr/bin/env python3
"""Replace rule seeds with seeds emitted by a frozen CTC checkpoint.

The rendered corpus is intentionally generated with a separate rule seed so
its operation labels are controllable.  This utility creates the transfer
manifest after frozen-CTC inference: only rows with a real CTC error or a
low-margin hard KEEP are retained, and the actual CTC text becomes `seed`.
Cross-data is never an input to this command for training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--branch-audit", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hard-keep-margin", type=float, default=0.15)
    args = ap.parse_args()
    rows = {}
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[Path(row["image"]).name] = row
    stats = {"matched": 0, "natural_error": 0, "hard_keep": 0, "missing": 0}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        for line in args.branch_audit.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            audit = json.loads(line)
            name = Path(audit.get("file", "")).name
            row = rows.get(name)
            if row is None:
                stats["missing"] += 1
                continue
            stats["matched"] += 1
            gt = row["gt"]
            seed = str((audit.get("ctc") or {}).get("text", ""))
            trace = (audit.get("ctc") or {}).get("frame_trace", [])
            margins = [float(x.get("top1_top2_margin", 1.0)) for x in trace]
            min_margin = min(margins) if margins else 1.0
            wrong = seed != gt
            hard_keep = not wrong and min_margin <= args.hard_keep_margin
            if not (wrong or hard_keep):
                continue
            item = dict(row)
            item["seed"] = seed
            item["seed_source"] = "frozen_ctc"
            item["rule_seed"] = row.get("seed")
            item["ctc_confidence"] = (audit.get("ctc") or {}).get("confidence")
            item["ctc_min_frame_margin"] = min_margin
            item["ctc_edit_distance"] = edit_distance(seed, gt)
            item["kind"] = "natural_ctc_error" if wrong else "hard_keep"
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            stats["natural_error" if wrong else "hard_keep"] += 1
    summary = {
        "manifest": str(args.manifest),
        "branch_audit": str(args.branch_audit),
        "out": str(args.out),
        "hard_keep_margin": args.hard_keep_margin,
        "stats": stats,
        "cross_data_used_for_training": False,
        "note": "Use this output as the visual transfer bank; retain the raw manifest only for mechanics pretraining.",
    }
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
