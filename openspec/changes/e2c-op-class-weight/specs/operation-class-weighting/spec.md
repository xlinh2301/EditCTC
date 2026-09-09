## ADDED Requirements

### Requirement: Moderate operation class weighting
The E2c edit operation loss SHALL use weights `[1.0, 2.0, 2.0, 2.0]` for `[KEEP, REPLACE, DELETE, INSERT_AFTER]` while retaining the E1 balanced sampler and the selected E2 edit-loss weight.

#### Scenario: Rare operation receives additional loss weight
- **WHEN** a valid REPLACE, DELETE, or INSERT_AFTER target contributes to operation loss
- **THEN** its contribution is weighted twice the contribution of a KEEP target before the selected edit-loss multiplier

### Requirement: Operation-specific reporting
The E2c report SHALL include a target-versus-prediction confusion matrix and precision/recall for every operation.

#### Scenario: Operation class weighting is evaluated
- **WHEN** E2c evaluation completes
- **THEN** the report distinguishes activation from correct operation prediction and from useful final corrections

