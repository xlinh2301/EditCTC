## Context

The natural CTC seed is correct on most training examples. E3's GT-derived
synthetic strings taught an artificial correction task without making NERD
useful on natural CTC errors.

## Decisions

- Generate the greedy seed from `ctc_logits` and detach it before corruption.
- Corrupt only non-empty seeds and guarantee one edit relative to that seed.
- Use REPLACE, DELETE, and INSERT_AFTER with equal requested probability;
  fall back to an available operation at sequence boundaries.
- Use the original GT label as the correction target. Align every synthetic
  seed with the same `levenshtein_ops` implementation used by the loss.
- Preserve a 50% natural row probability so the decoder does not forget the
  deployed input distribution.

## Acceptance Signals

- Training diagnostics report approximately 50% synthetic rows and balanced
  corruption types.
- Reconstruction checks show one edit before training begins.
- Natural B_dev reports NERD accuracy, CTC accuracy, changed/helped/hurt,
  operation recall, error-fix rate, and edit precision.
- No inference-time seed or post-processing change is present.
