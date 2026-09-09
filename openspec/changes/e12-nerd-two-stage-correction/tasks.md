## 1. Stage control

- [ ] 1.1 Add explicit freeze/unfreeze configuration for Stage A and Stage B.
- [ ] 1.2 Save and resume stage-specific checkpoints without optimizer-state
  ambiguity.
- [ ] 1.3 Log trainable parameter counts and stage identifiers.

## 2. Correction metrics

- [ ] 2.1 Add synthetic corruption validation with exact operation metrics.
- [ ] 2.2 Add natural CTC wrong-to-NERD-correct error-fix rate and edit
  precision.
- [ ] 2.3 Add checkpoint selection that records both natural and synthetic
  objectives.

## 3. Isolated run

- [ ] 3.1 Create an isolated config/worktree based on the best E2a recipe.
- [ ] 3.2 Run Stage A and Stage B on the light Slurm GPU allocation.
- [ ] 3.3 Save B_dev wrong cases and compare against E0, E3, and E4.
