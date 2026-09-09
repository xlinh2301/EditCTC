## 1. Consistency loss

- [ ] 1.1 Compute expected insertion and deletion counts over valid seed positions.
- [ ] 1.2 Add the length residual and configurable robust penalty to the edit loss.
- [ ] 1.3 Set and record the initial E5 weight at 0.02 or 0.05.

## 2. Diagnostics

- [ ] 2.1 Log consistency loss and predicted length residual during training.
- [ ] 2.2 Add actual seed/refined length comparison to the evaluation report.
- [ ] 2.3 Preserve operation precision/recall and helped/hurt metrics.

## 3. Run and decide

- [ ] 3.1 Train one E5 seed from the selected E4 configuration.
- [ ] 3.2 Evaluate both datasets with natural CTC seeds.
- [ ] 3.3 Compare E5 against E4 for length residual, useful edits, CER, and exact match.
- [ ] 3.4 Reject the consistency term if it reduces helpful corrections or increases harmful edits without a compensating gain.

