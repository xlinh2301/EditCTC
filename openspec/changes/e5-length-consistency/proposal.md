## Why

Soft LCB conditioning gives NERD access to length uncertainty, but the two branches can still disagree about how edits affect sequence length. A small consistency objective can couple expected insertions and deletions to the ground-truth length.

## What Changes

- Start from E4.
- Add a small differentiable length-consistency loss based on seed length, expected INSERT_AFTER probability, and expected DELETE probability.
- Keep the consistency weight small initially, between 0.02 and 0.05.
- Report whether consistency improves helpful edits without increasing harmful edits.

## Capabilities

### New Capabilities

- `length-consistency-training`: Align expected NERD length-changing operations with the target sequence length.

### Modified Capabilities

## Impact

- Extends the edit loss with a differentiable auxiliary consistency term.
- Requires soft LCB conditioning from E4.
- Adds a consistency metric to training and evaluation artifacts.

