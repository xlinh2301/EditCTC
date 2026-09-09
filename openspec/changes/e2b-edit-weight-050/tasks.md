## 1. Configuration

- [ ] 1.1 Copy the E1 configuration and set `weight_edit: 0.50`.
- [ ] 1.2 Verify the balanced sampler and all non-weight settings match E1.

## 2. Run and measure

- [ ] 2.1 Train one E2b seed from the E1-compatible initialization.
- [ ] 2.2 Run the standard evaluation on both datasets.
- [ ] 2.3 Record operation distributions, precision/recall, helped/hurt, CER, and exact match.

## 3. Decision

- [ ] 3.1 Compare E2b with E1 and E2a on the same primary and safety metrics.
- [ ] 3.2 Reject E2b if increased edits do not improve useful corrections or cause unacceptable regression.

