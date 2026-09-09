## ADDED Requirements

### Requirement: Complete edit-target coverage
The edit loss SHALL include every valid target position whose operation is `REPLACE`, `DELETE`, or `INSERT_AFTER` in every training batch.

#### Scenario: Rare edit target is selected
- **WHEN** a valid non-KEEP operation occurs in a batch
- **THEN** that position is included in the edit-loss mask regardless of its random selection state

#### Scenario: Padding is excluded
- **WHEN** a position has `IGNORE_INDEX`
- **THEN** it is excluded from both edit and KEEP masks

### Requirement: Controlled KEEP subsampling
The edit loss SHALL randomly sample only valid KEEP positions and SHALL target an initial ratio of two selected KEEP positions per selected edit position, subject to available positions.

#### Scenario: Batch has edit targets
- **WHEN** a batch has `n_edit` valid non-KEEP positions
- **THEN** the loss selects all `n_edit` positions and at most `2 * n_edit` valid KEEP positions

### Requirement: Edit-supervision diagnostics
Training and evaluation artifacts SHALL report target operation counts, predicted operation counts, per-operation recall, predicted change rate, and final helped/hurt counts.

#### Scenario: Balanced run is inspected
- **WHEN** an E1 run finishes
- **THEN** its report makes it possible to determine whether NERD predicts and correctly applies each edit operation

