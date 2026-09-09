## 1. Balanced mask

- [ ] 1.1 Add a valid mask that excludes `IGNORE_INDEX` before selecting edit and KEEP positions.
- [ ] 1.2 Select all non-KEEP targets and sample KEEP targets at a 2:1 ratio.
- [ ] 1.3 Add configuration for the KEEP-to-edit sampling ratio.

## 2. Diagnostics

- [ ] 2.1 Log selected target counts and target/predicted operation distributions.
- [ ] 2.2 Compute REPLACE, DELETE, and INSERT_AFTER precision and recall.
- [ ] 2.3 Preserve final NERD helped/hurt and error-fix metrics.

## 3. Run and verify

- [ ] 3.1 Create the E1 config and Slurm script from the E0 protocol.
- [ ] 3.2 Train one seed from the same initialization as E0.
- [ ] 3.3 Evaluate natural CTC seeds on both datasets and record the E1 report.
- [ ] 3.4 Mark E1 as activation-pass only if non-KEEP predictions and changed cases are nonzero.

