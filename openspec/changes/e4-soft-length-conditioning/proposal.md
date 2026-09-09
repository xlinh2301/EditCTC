## Why

NERD currently receives only the hard argmax class from LCB. This discards length uncertainty and prevents the NERD edit loss from providing a differentiable learning signal through the LCB length distribution.

## What Changes

- Replace hard argmax length embedding with the expected embedding under the LCB probability distribution.
- Exclude the unused length class zero before normalization.
- Keep the E3 synthetic-seed recipe and test soft conditioning independently before adding a consistency loss.

## Capabilities

### New Capabilities

- `soft-length-conditioning`: Couple NERD to the calibrated LCB length distribution through a differentiable soft embedding.

### Modified Capabilities

## Impact

- Changes `EditRefineDecoder` length conditioning and its gradient path.
- Changes no NERD layer count or operation decoding rule.
- Requires comparison against the best E3 model using identical natural-seed evaluation.

