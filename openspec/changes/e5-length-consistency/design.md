## Context

E4 provides a soft LCB signal but does not explicitly align length-changing operations with the target sequence length. E5 adds a small auxiliary objective after E4 so its effect can be attributed to consistency rather than to sampling or conditioning changes.

## Goals / Non-Goals

**Goals:**

- Couple expected INSERT_AFTER and DELETE behavior to target length.
- Keep operation classification and token losses unchanged.
- Measure residual length error alongside recognition outcomes.

**Non-Goals:**

- No hard length override at inference.
- No new NERD layers, focal loss, or multi-error synthetic corruption.

## Decisions

- Use `L_seed + sum(p_insert) - sum(p_delete) - L_gt` as the residual over valid seed positions.
- Start with a small configurable consistency weight, `0.02` or `0.05`.
- Use a smooth robust penalty if absolute-value gradients are unstable, while retaining the same residual definition.

## Risks / Trade-offs

- [Expected operations do not equal discrete output length exactly] → Evaluate both residual and actual refined length.
- [Consistency may suppress valid replacements] → Replacements do not affect length; monitor operation-specific recall and helpful/hurt counts.
- [Extra coupling can destabilize training] → Add the term only after E4 is stable and compare raw loss components.

