## Context

The existing `s1024` checkpoint is the reference model for all NERD and LCB training experiments. The project already has Slurm inference and branch-audit support under `slurm/` and `tools/infer_rec.py`.

## Goals / Non-Goals

**Goals:**

- Freeze the current dataset, checkpoint, and evaluation protocol as E0.
- Preserve per-image branch evidence and wrong cases for paired comparisons.

**Non-Goals:**

- No model, loss, sampler, or inference changes.
- No claim that E0 proves or disproves the value of NERD beyond this protocol.

## Decisions

- Use exact-match accuracy as the primary metric and retain CER/edit distance as secondary metrics.
- Compare all later experiments on the same two datasets and image-label mapping.
- Store JSON summaries and JSONL audit files under a versioned experiment output directory.

## Risks / Trade-offs

- [Small error count] → Report raw counts beside percentages and use paired multi-seed runs for final conclusions.
- [Missing images or label drift] → Record skipped labels and hash or copy the evaluated file list.

