#!/usr/bin/env python3
"""Read-only E37 sequence candidate audit on precomputed evaluation audits.

The exported branch audits do not contain Top-K token probabilities or encoder
spans.  This audit therefore uses the proposals that are actually persisted:
KEEP, the full NRTR string, and each NERD predicted-token replacement.  No
parameters are fitted and cross-data is never used for training.
"""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path


def ed(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def load_labels(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            name, gt = line.split("\t", 1)
            out[Path(name).name] = gt.strip()
    return out


def transition(seed: str, cand: str, gt: str) -> tuple[str, int]:
    gain = ed(seed, gt) - ed(cand, gt)
    src = "correct" if seed == gt else "wrong"
    if gain > 0:
        dst = "correct" if cand == gt else "better_wrong"
    elif gain == 0:
        dst = "correct" if cand == gt else ("same" if cand == seed else "wrong_same")
    else:
        dst = "worse"
    return f"{src}_to_{dst}", gain


def candidates(row: dict) -> list[dict]:
    seed = row["ctc"]["text"]
    out = [{"text": seed, "source": "KEEP"}]
    seen = {seed}
    nrtr = row.get("nrtr", {}).get("text", "")
    if nrtr and nrtr not in seen:
        seen.add(nrtr)
        out.append({"text": nrtr, "source": "NRTR_FULL"})
    for op in row.get("nerd", {}).get("operations", []):
        pos = op.get("position")
        pred = op.get("predicted_token")
        if not isinstance(pos, int) or not isinstance(pred, str) or len(pred) != 1:
            continue
        if not pred.isdigit() or pos < 0 or pos >= len(seed) or pred == seed[pos]:
            continue
        cand = seed[:pos] + pred + seed[pos + 1:]
        if cand not in seen:
            seen.add(cand)
            out.append({"text": cand, "source": f"NERD_REPLACE@{pos}"})
    return out


def audit(audit_path: Path, label_path: Path) -> dict:
    labels = load_labels(label_path)
    rows = []
    skipped = 0
    for line in audit_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        name = Path(r["file"]).name
        if name not in labels:
            skipped += 1
            continue
        r["_gt"] = labels[name]
        rows.append(r)
    taxonomy = Counter()
    source_taxonomy = Counter()
    error_structure = Counter()
    source_exact_wrong = Counter()
    group_stats = Counter()
    exact_groups = 0
    beneficial_groups = 0
    wrong_rows = sum(r["ctc"]["text"] != r["_gt"] for r in rows)
    oracle_gain_values = []
    for r in rows:
        seed, gt = r["ctc"]["text"], r["_gt"]
        if seed != gt:
            if len(seed) == len(gt):
                error_structure["wrong_equal_length"] += 1
            elif len(seed) < len(gt):
                error_structure["wrong_under_length"] += 1
            else:
                error_structure["wrong_over_length"] += 1
            error_structure[f"wrong_edit_distance_{ed(seed, gt)}"] += 1
        cs = candidates(r)
        gains = []
        for c in cs:
            tr, gain = transition(seed, c["text"], gt)
            c["gain"], c["transition"] = gain, tr
            taxonomy[tr] += 1
            source_taxonomy[(c["source"], tr)] += 1
            if seed != gt and c["text"] == gt:
                source_exact_wrong[c["source"]] += 1
            gains.append(gain)
        best = max(gains)
        oracle_gain_values.append(best)
        if any(c["text"] == gt for c in cs):
            exact_groups += 1
        if any(c["gain"] > 0 for c in cs[1:]):
            beneficial_groups += 1
        group_stats["groups"] += 1
        group_stats["candidates"] += len(cs)
        group_stats["groups_with_nonkeep"] += int(len(cs) > 1)
        group_stats["groups_with_beneficial"] += int(any(c["gain"] > 0 for c in cs[1:]))
        group_stats["groups_with_exact_gt"] += int(any(c["text"] == gt for c in cs[1:]))
        group_stats["groups_with_harmful"] += int(any(c["gain"] < 0 for c in cs[1:]))
    denom = max(1, wrong_rows)
    result = {
        "audit": str(audit_path), "label_file": str(label_path),
        "evaluated_rows": len(rows), "unmatched_audit_rows": skipped,
        "wrong_seed_rows": wrong_rows,
        "correct_seed_rows": len(rows) - wrong_rows,
        "candidate_definition": "KEEP + NRTR full string + unique NERD predicted-token single-position replacements persisted by branch_audit",
        "no_training": True, "cross_data_used_for_training": False,
        "group_stats": dict(group_stats),
        "wrong_row_candidate_exact_coverage": sum(
            1 for r in rows if r["ctc"]["text"] != r["_gt"] and any(c["text"] == r["_gt"] for c in candidates(r))) / denom,
        "wrong_row_beneficial_oracle_coverage": sum(
            1 for r in rows if r["ctc"]["text"] != r["_gt"] and any(transition(r["ctc"]["text"], c["text"], r["_gt"])[1] > 0 for c in candidates(r)[1:])) / denom,
        "all_group_exact_coverage": exact_groups / max(1, len(rows)),
        "all_group_beneficial_coverage": beneficial_groups / max(1, len(rows)),
        "taxonomy": dict(sorted(taxonomy.items())),
        "source_taxonomy": {f"{a}|{b}": n for (a, b), n in sorted(source_taxonomy.items())},
        "wrong_seed_error_structure": dict(sorted(error_structure.items())),
        "wrong_seed_exact_correction_by_source": dict(sorted(source_exact_wrong.items())),
        "oracle_best_gain_mean": sum(oracle_gain_values) / max(1, len(oracle_gain_values)),
        "oracle_best_gain_positive": sum(x > 0 for x in oracle_gain_values),
    }
    return result, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", required=True, type=Path)
    ap.add_argument("--label-file", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    result, rows = audit(args.audit, args.label_file)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (args.out_dir / "counterfactual_groups.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            seed, gt = r["ctc"]["text"], r["_gt"]
            cs = candidates(r)
            for c in cs:
                c["transition"], c["gain"] = transition(seed, c["text"], gt)
            f.write(json.dumps({"file": Path(r["file"]).name, "seed": seed,
                                "ground_truth": gt, "candidates": cs}, ensure_ascii=False) + "\n")
    lines = ["# Cross-data E37 sequence audit", "",
             "Read-only audit of persisted CTC/NRTR/NERD proposals. No model was fitted and no cross-data row entered training.", "",
             f"- Evaluated rows: `{result['evaluated_rows']}`; wrong CTC seeds: `{result['wrong_seed_rows']}`.",
             f"- Candidate exact coverage on wrong rows: `{result['wrong_row_candidate_exact_coverage']:.4f}`.",
             f"- Beneficial candidate oracle coverage on wrong rows: `{result['wrong_row_beneficial_oracle_coverage']:.4f}`.",
             f"- Candidate groups with any beneficial proposal: `{result['group_stats']['groups_with_beneficial']}`.",
             "", "## Transition taxonomy", "", "```json", json.dumps(result["taxonomy"], indent=2), "```", "",
             "The persisted cross audit has no encoder spans or token Top-K probabilities; this limits E37 to the real proposals saved in branch_audit (NRTR full string and NERD token proposals)."]
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
