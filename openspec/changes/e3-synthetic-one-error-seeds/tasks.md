## 1. Corruption generator

- [ ] 1.1 Implement seeded substitution, deletion, and insertion corruption for encoded labels.
- [ ] 1.2 Enforce exactly one error and reject unchanged substitutions.
- [ ] 1.3 Add alignment checks for operation reconstruction and repeated-token edge cases.

## 2. Training plumbing

- [ ] 2.1 Add configurable natural/synthetic sampling probabilities.
- [ ] 2.2 Pass synthetic seed IDs and lengths into NERD and edit-loss target construction.
- [ ] 2.3 Log natural/synthetic sample counts and synthetic operation counts.

## 3. Evaluation

- [ ] 3.1 Keep validation and test on natural CTC seeds only.
- [ ] 3.2 Train one E3 seed from the best E1/E2 configuration.
- [ ] 3.3 Evaluate operation recall, final helpful/harmful edits, CER, and exact match.
- [ ] 3.4 Compare natural-seed performance against E0 and the selected E2 model.

