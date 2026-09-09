## Decisions

- Start from the immutable `s1024` checkpoint for both stages.
- Stage A uses E1-freeze semantics and the E2a edit weight `0.30`; no class
  weighting or hard length override.
- Stage B uses the Stage A checkpoint, freezes CTC/visual/NRTR parameters, and
  unfreezes NERD plus LCB only.
- Save `stage_a` and `stage_b` checkpoints separately.
- Add synthetic and natural validation summaries so checkpoint selection can
  distinguish correction ability from KEEP behavior.

## Risks

- Stage A may learn synthetic edits but fail on natural errors → require both
  synthetic correction and natural error-fix metrics.
- Stage B may damage CTC or LCB → keep frozen CTC and compare LCB accuracy to
  E4 before accepting the branch.

## Acceptance Signals

- Stage boundaries and trainable parameter groups are printed in the log.
- Stage A has nonzero synthetic correction rate.
- Stage B has `nerd_helped > 0` and `nerd_helped > nerd_hurt`, or the report
  explicitly rejects the hypothesis with per-case evidence.
