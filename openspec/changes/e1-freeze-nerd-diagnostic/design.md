## Context

E1 changes which positions contribute to the edit loss but still permits the shared visual representation and CTC seed to move. This diagnostic isolates the edit head by fixing the source representation and seed while retaining the E1 target-selection rule.

## Goals / Non-Goals

**Goals:**

- Test whether NERD can learn corrections from the existing E0 visual features.
- Remove moving-seed and shared-backbone gradient competition from the diagnosis.

**Non-Goals:**

- This is not the final joint-training recipe.
- It does not test soft LCB, synthetic seeds, or new operation decoding.

## Decisions

- Initialize from the same E0-compatible weights used for E1.
- Freeze the visual/CTC path and train only NERD parameters, with the same optimizer budget where feasible.
- Evaluate on natural CTC seeds so the diagnostic remains comparable to E0/E1.

## Risks / Trade-offs

- [Frozen features may limit correction quality] → Treat the run as a learnability diagnosis, not as an accuracy candidate.
- [Different optimizer parameter count] → Report trainable parameter names and compare operation metrics rather than only final accuracy.

