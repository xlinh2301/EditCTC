## Context

E2b is the stronger edit-loss comparison after E1. It tests whether NERD needs a larger optimization signal before adding class weighting or synthetic corruption.

## Goals / Non-Goals

**Goals:**

- Measure `weight_edit: 0.50` under the same balanced target sampling as E1.
- Identify over-editing caused by excessive edit pressure.

**Non-Goals:**

- No architecture change, class weighting, synthetic seed, soft LCB, or inference heuristic.

## Decisions

- Use the E1 sampler and the same initialization/evaluation protocol.
- Keep E2b as a candidate only when the helped-to-hurt balance and primary metric are acceptable.

## Risks / Trade-offs

- [Edit branch overwhelms recognition] → Check CTC and NERD separately and reject any run with material regression on correct CTC cases.
- [Large loss-scale change] → Record raw loss components and gradient or parameter-update diagnostics when available.

