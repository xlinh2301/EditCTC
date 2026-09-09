## Context

The audit found examples where CTC is wrong but LCB predicts the correct
length. A final decision rule must exploit that information while avoiding a
hard length override, because LCB also makes errors on CTC-correct examples.

## Decisions

- Generate a small N-best set from the existing CTC decoder or its bounded
  prefix beam implementation; do not introduce a new neural branch.
- Use the normalized CTC candidate score and the LCB log probability of the
  candidate's decoded length.
- Expose `beta` in config and run beta zero as a regression check before beta
  `0.10`.
- Compare CTC versus reranked output per file, including LCB-helped and
  LCB-hurt counts.

## Acceptance Signals

- `beta=0` reproduces the existing CTC output exactly.
- B_dev reports `lcb_helped`, `lcb_hurt`, rerank rate, and final accuracy.
- The experiment never changes a candidate solely because its length equals
  the LCB argmax; the combined score must select it.
