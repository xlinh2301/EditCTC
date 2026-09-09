## 1. Loss configuration

- [ ] 1.1 Add configurable operation class weights to the uncertainty edit loss.
- [ ] 1.2 Set E2c weights to `[1.0, 2.0, 2.0, 2.0]`.
- [ ] 1.3 Verify E1 sampler and selected edit-loss weight remain unchanged.

## 2. Metrics

- [ ] 2.1 Add operation target/prediction confusion counts with raw support.
- [ ] 2.2 Add operation precision, recall, and predicted-change rate to the report.
- [ ] 2.3 Preserve final helped/hurt and error-fix metrics.

## 3. Run and decide

- [ ] 3.1 Train and evaluate one E2c seed on both datasets.
- [ ] 3.2 Compare against the selected E1/E2 setting.
- [ ] 3.3 Continue only if rare-operation recall improves without harmful edit domination.

