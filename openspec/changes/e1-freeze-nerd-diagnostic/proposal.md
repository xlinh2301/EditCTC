## Why

E1 may still fail if the edit head cannot learn from the current visual features, even after the sampling fix. A frozen-backbone diagnostic isolates edit-head learnability from moving CTC seeds and shared-backbone gradient competition.

## What Changes

- Freeze the CTC encoder/backbone and CTC head from the same E0 initialization.
- Train only the NERD edit head with E1 balanced edit supervision.
- Keep the seed fixed across epochs and compare operation recall against E1.

## Capabilities

### New Capabilities

- `frozen-nerd-diagnostic`: Isolate NERD edit learning with a fixed CTC seed and visual representation.

### Modified Capabilities

## Impact

- Adds a diagnostic training configuration and run script.
- Changes trainable parameters only for this diagnostic experiment.
- Does not define the production training recipe.

