## Why

The current `s1024` checkpoint is the reference point for the planned NERD and LCB experiments. Its results and audit artifacts must be recorded before any training change is evaluated.

## What Changes

- Record the existing `s1024` checkpoint, configuration, datasets, and evaluation commands.
- Preserve branch-level metrics for CTC, NRTR, LCB, and NERD.
- Record wrong cases and NERD operation counts as the E0 comparison baseline.

## Capabilities

### New Capabilities

- `baseline-evaluation`: Reproducible baseline evaluation and branch audit for EditCTC.

### Modified Capabilities

## Impact

- Uses the existing checkpoint and evaluation scripts.
- Writes baseline summaries, predictions, branch audits, and wrong-case JSONL artifacts.
- Does not change model code or checkpoint weights.

