## Why

The current NERD training objective is dominated by `KEEP`: the evaluation audit contains only 1.87% and 1.04% non-KEEP target positions on Cross-data and Indomain respectively. Randomly selecting 10% of positions can omit the few positions that require correction, so the model converges to an always-KEEP policy.

## What Changes

- Select every valid non-KEEP target position for the edit loss.
- Subsample only valid KEEP positions at a configurable ratio, initially 2 KEEP positions per edit position.
- Keep the NERD architecture and `weight_edit: 0.15` unchanged.
- Log target and predicted operation distributions and per-operation recall.

## Capabilities

### New Capabilities

- `balanced-edit-supervision`: Train NERD with guaranteed coverage of edit targets and controlled KEEP sampling.

### Modified Capabilities

## Impact

- Affects `EditLossUncertainty` masking and its training diagnostics.
- Adds a new E1 configuration and Slurm training/evaluation entry point.
- Does not change inference decoding, NERD depth, LCB architecture, or the baseline checkpoint.

