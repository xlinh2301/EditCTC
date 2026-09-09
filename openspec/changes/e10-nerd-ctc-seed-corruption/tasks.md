## 1. Seed corruption

- [ ] 1.1 Add a helper that corrupts a detached natural CTC seed once.
- [ ] 1.2 Validate substitution, deletion, and insertion edge cases against
  `levenshtein_ops` and `apply_edit_ops`.
- [ ] 1.3 Keep natural rows unchanged and return detached diagnostic metadata.

## 2. Training and logging

- [ ] 2.1 Add configuration for the natural/synthetic and operation ratios.
- [ ] 2.2 Pass the corrupted seed through the existing external seed path.
- [ ] 2.3 Log synthetic counts, operation recall, and correction diagnostics.

## 3. Evaluation

- [ ] 3.1 Add a controlled synthetic-seed evaluator separate from B_dev.
- [ ] 3.2 Run the natural B_dev protocol with one seed and save wrong cases.
- [ ] 3.3 Compare E10 with E0, E2a, and E3 using absolute accuracy and
  helped/hurt metrics.
