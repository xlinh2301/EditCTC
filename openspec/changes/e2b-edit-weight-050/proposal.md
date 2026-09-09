## Why

E2a may not provide enough gradient to overcome the KEEP prior. E2b tests a stronger edit-loss weight as a separate controlled comparison, without changing the sampler or model.

## What Changes

- Start from the E1 balanced sampler.
- Set `weight_edit` to `0.50`.
- Measure whether operation recall and useful corrections improve without increasing harmful edits.

## Capabilities

### New Capabilities

- `strong-edit-loss-weight`: Train the balanced NERD branch with edit-loss weight 0.50.

### Modified Capabilities

## Impact

- Changes only the training loss weight and experiment configuration.
- Uses the same initialization and evaluation protocol as E1 and E2a.

