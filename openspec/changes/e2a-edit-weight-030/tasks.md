## 1. Configuration

- [ ] 1.1 Copy the E1 configuration and set `weight_edit: 0.30`.
- [ ] 1.2 Verify no sampler, architecture, data, or inference settings changed.

## 2. Run and measure

- [ ] 2.1 Train one E2a seed from the E1-compatible initialization.
- [ ] 2.2 Evaluate both datasets with the shared audit script.
- [ ] 2.3 Compute exact match, CER, operation precision/recall, helped, hurt, and error-fix rate.

## 3. Decision

- [ ] 3.1 Compare E2a against E1 using paired image-level results.
- [ ] 3.2 Select E2a for further work only if the additional edit pressure is useful.

