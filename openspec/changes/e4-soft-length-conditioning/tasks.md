## 1. Soft conditioning

- [ ] 1.1 Replace hard length argmax with a valid-class probability distribution.
- [ ] 1.2 Implement probability-weighted length embedding in NERD.
- [ ] 1.3 Remove only the length-conditioning stop-gradient; preserve unrelated stop-gradient behavior.

## 2. Verification

- [ ] 2.1 Add tests for class-zero exclusion, probability normalization, and embedding shape.
- [ ] 2.2 Verify LCB receives gradient from the NERD edit objective.
- [ ] 2.3 Verify natural-seed inference still produces valid outputs.

## 3. Run and decide

- [ ] 3.1 Train one E4 seed from the selected E3 configuration.
- [ ] 3.2 Compare hard versus soft LCB on length accuracy, operation metrics, helped/hurt, CER, and exact match.
- [ ] 3.3 Continue to E5 only if the soft path is numerically stable and diagnostically useful.

