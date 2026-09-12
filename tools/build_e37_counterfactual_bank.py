#!/usr/bin/env python3
"""Build complete-sequence counterfactual groups from a natural CTC bank."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


CONFUSIONS = {
    "0": "8", "1": "7", "2": "7", "3": "8", "4": "9",
    "5": "6", "6": "5", "7": "1", "8": "3", "9": "4",
}


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def make_group(row, max_alts=4):
    seed, gt = row["seed"], row["gt"]
    base = edit_distance(seed, gt)
    candidates = [{"text": seed, "kind": "seed", "delta_ed": 0}]
    seen = {seed}
    if gt != seed:
        candidates.append({"text": gt, "kind": "gt", "delta_ed": base})
        seen.add(gt)
    for pos, char in enumerate(seed):
        alt = CONFUSIONS.get(char, "0")
        text = seed[:pos] + alt + seed[pos + 1:]
        if text in seen:
            continue
        candidates.append({
            "text": text,
            "kind": "replacement",
            "delta_ed": base - edit_distance(text, gt),
        })
        seen.add(text)
        if len(candidates) >= max_alts + 1:
            break
    return {
        "image": row["image"],
        "seed": seed,
        "gt": gt,
        "wrong_seed": bool(row.get("wrong_seed", seed != gt)),
        "split": row.get("split", "train"),
        "candidates": candidates,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-alts", type=int, default=4)
    args = ap.parse_args()
    rows = [json.loads(x) for x in args.manifest.read_text().splitlines() if x.strip()]
    groups = [make_group(row, args.max_alts) for row in rows]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(x) for x in groups) + "\n")
    stats = {"groups": len(groups), "train": 0, "val": 0, "wrong": 0,
             "beneficial_candidates": 0, "harmful_candidates": 0,
             "gt_candidates": 0, "fully_fixable_groups": 0}
    for group in groups:
        stats[group["split"]] = stats.get(group["split"], 0) + 1
        stats["wrong"] += int(group["wrong_seed"])
        stats["gt_candidates"] += int(any(c["kind"] == "gt" for c in group["candidates"]))
        best = max(c["delta_ed"] for c in group["candidates"])
        stats["fully_fixable_groups"] += int(best >= edit_distance(group["seed"], group["gt"]))
        stats["beneficial_candidates"] += sum(c["delta_ed"] > 0 for c in group["candidates"])
        stats["harmful_candidates"] += sum(c["delta_ed"] < 0 for c in group["candidates"])
    (args.out.parent / (args.out.stem + ".summary.json")).write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
