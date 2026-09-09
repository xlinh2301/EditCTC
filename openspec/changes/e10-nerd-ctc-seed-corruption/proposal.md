## Why

E3 generated corruptions directly from ground truth. Training logs confirmed
that synthetic supervision was present, but natural-seed B_dev still returned
all KEEP. The synthetic input distribution therefore remains too different
from the deployed CTC seed.

## What Changes

- Build the natural greedy CTC seed first.
- On 50% of training rows, apply exactly one substitution, deletion, or
  insertion to that seed, then align the corrupted seed to the ground truth.
- Keep validation and inference on untouched natural CTC seeds.
- Keep the best E2a recipe: `weight_edit=0.30`, no operation class weights,
  three NERD decoder layers, and no inference threshold.
- Log corruption type, operation recall, synthetic correction rate, and
  natural-seed helped/hurt cases.

## Capabilities

### New Capabilities

- `ctc-seed-corruption-training`: NERD sees the same seed distribution used at
  inference before a controlled one-error perturbation.

### Modified Capabilities

## Impact

The change is training-only and isolated to a new configuration and seed
generator. CTC, LCB, NRTR, architecture depth, and inference post-processing
remain unchanged.
