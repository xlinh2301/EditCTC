## Why

Balanced sampling guarantees that edit positions are seen, but REPLACE, DELETE, and INSERT_AFTER remain rare within the edit positions. Moderate operation class weights can improve recall for these operations without immediately introducing focal loss or extreme inverse-frequency weights.

## What Changes

- Use the E1 balanced sampler and the best edit-loss weight from E2a/E2b.
- Apply operation weights `[1.0, 2.0, 2.0, 2.0]` for `[KEEP, REPLACE, DELETE, INSERT_AFTER]`.
- Report operation confusion matrices and operation-specific precision/recall.

## Capabilities

### New Capabilities

- `operation-class-weighting`: Train NERD with moderate class weighting for rare edit operations.

### Modified Capabilities

## Impact

- Extends the edit operation loss configuration.
- Requires the selected E2 checkpoint/configuration as the comparison reference.
- Does not change model layers or inference decoding.

