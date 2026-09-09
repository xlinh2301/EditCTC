## Why

Natural CTC seeds are usually already correct, so current alignment targets overwhelmingly say KEEP. Synthetic one-error seeds provide direct supervision for NERD to replace, delete, and insert a token while retaining a natural-seed mixture for transfer to real inference errors.

## What Changes

- Retain a natural CTC-seed training path.
- Add synthetic single-error seeds created from ground truth with unambiguous substitution, deletion, or insertion.
- Start with a 50% natural and 50% synthetic mixture, with equal synthetic operation types.
- Evaluate only on natural CTC seeds at validation and test time.

## Capabilities

### New Capabilities

- `synthetic-one-error-seeds`: Train and audit NERD on controlled single-operation corruptions.

### Modified Capabilities

## Impact

- Adds deterministic, seed-controlled synthetic-seed generation to the training path.
- Extends batch/head/loss plumbing to accept externally supplied seed IDs and lengths.
- Keeps evaluation data and final inference behavior unchanged.

