## Context

E1 isolates target sampling. If it activates NERD but the edit branch remains weak, increasing `weight_edit` tests whether the branch is underweighted relative to the CTC and NRTR objectives.

## Goals / Non-Goals

**Goals:**

- Measure the effect of changing `weight_edit` from 0.15 to 0.30.
- Preserve the E1 balanced sampler and all architecture settings.

**Non-Goals:**

- No class weighting, focal loss, synthetic seed, soft LCB, or inference change.

## Decisions

- Train E2a from the same base initialization and protocol as E1.
- Compare paired predictions and branch metrics, not only aggregate exact match.
- Retain E2a only if useful edits improve without an unacceptable hurt increase.

## Risks / Trade-offs

- [Over-editing] → Track `nerd_hurt`, operation precision, and CTC-preserving regression counts.
- [Training stochasticity] → Keep the first run to one seed and repeat only shortlisted settings.

