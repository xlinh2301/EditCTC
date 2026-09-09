## 1. Frozen setup

- [ ] 1.1 Add a configuration flag that freezes the CTC encoder/backbone and CTC head.
- [ ] 1.2 Verify the optimizer contains only NERD parameters.
- [ ] 1.3 Verify CTC seed IDs remain stable across epochs.

## 2. Training and audit

- [ ] 2.1 Reuse E1 balanced edit sampling and diagnostics.
- [ ] 2.2 Train one diagnostic seed from the E0-compatible initialization.
- [ ] 2.3 Evaluate operation recall and final helpful/harmful changes.

## 3. Decision

- [ ] 3.1 If NERD remains all-KEEP, inspect target construction and decoder wiring before E2.
- [ ] 3.2 If NERD activates, use E1/E2 joint training for the next accuracy experiment.

