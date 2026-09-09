## Why

E1-freeze proved that NERD can emit operations, while E3 showed that one-error
supervision alone still collapses to KEEP on natural evaluation. Joint training
lets the strong CTC objective and the edit objective compete on a tiny dataset.

## What Changes

- Stage A freezes the visual, CTC, NRTR, and LCB paths and trains only NERD on
  natural CTC seeds plus controlled one-error seed corruptions.
- Stage B keeps CTC frozen, unfreezes NERD and LCB, and fine-tunes with a low
  learning rate while retaining the natural/synthetic mixture.
- Select checkpoints using both natural NERD accuracy and synthetic correction
  rate; never select on CTC accuracy alone.
- Keep inference unchanged and report whether NERD fixes natural CTC errors
  without increasing harmful edits.

## Capabilities

### New Capabilities

- `two-stage-nerd-correction`: NERD first learns the correction mapping with a
  stable visual/CTC interface, then adapts LCB and NERD jointly.

### Modified Capabilities

## Impact

This is a training and checkpoint-selection experiment. It reuses E10's seed
corruption and adds explicit stage boundaries and selection diagnostics. It
does not change NERD depth or inference decoding.
