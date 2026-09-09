## Why

If balanced sampling activates NERD but its edits remain too rare, the edit loss may still be underweighted relative to CTC and NRTR. E2a tests a moderate increase while keeping the sampling and architecture fixed.

## What Changes

- Start from the E1 balanced sampler.
- Set `weight_edit` to `0.30`.
- Evaluate operation recall, helpful versus harmful edits, and final recognition metrics.

## Capabilities

### New Capabilities

- `moderate-edit-loss-weight`: Train the balanced NERD branch with edit-loss weight 0.30.

### Modified Capabilities

## Impact

- Changes only the training loss weight and experiment configuration.
- Uses the same checkpoint initialization, data, evaluation, and inference procedure as E1.

