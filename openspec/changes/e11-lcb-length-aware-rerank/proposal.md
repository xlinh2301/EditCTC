## Why

E4 raised LCB length accuracy to `0.89024`, but NERD stayed all KEEP and the
length prediction never changed the final text. LCB currently acts as a
conditioning feature only, so its complementary information is not measured
as a decision signal.

## What Changes

- Add a bounded length-aware CTC candidate reranker.
- Score each candidate with its CTC score plus `beta * log P_LCB(length)`.
- Keep `beta=0` as the exact baseline and use a small fixed first candidate
  `beta=0.10` for the isolated experiment.
- Preserve the original CTC result when candidate generation or LCB confidence
  is unavailable.
- Log candidate length, LCB class, rerank decision, and cases where LCB fixes
  a CTC error or harms a CTC-correct result.

## Capabilities

### New Capabilities

- `length-aware-ctc-reranking`: LCB can select among existing CTC candidates
  without hard-forcing an output length.

### Modified Capabilities

## Impact

This is an inference-only decoding experiment. It does not change NERD
training or the checkpoint. It uses the same B_dev/B_test audit and stores
the original CTC text beside the reranked text for attribution.
