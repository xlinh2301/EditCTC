## ADDED Requirements

### Requirement: Moderate edit-loss weight
The E2a training configuration SHALL use the E1 balanced edit sampler and set `weight_edit` to `0.30`, with all other architecture and loss settings held constant.

#### Scenario: E2a configuration is isolated
- **WHEN** E2a is launched
- **THEN** the only intended model-training difference from E1 is `weight_edit: 0.30`

### Requirement: Paired edit evaluation
E2a SHALL be evaluated with the same natural-seed protocol and datasets as E0 and E1.

#### Scenario: E2a results are comparable
- **WHEN** E2a evaluation completes
- **THEN** its summary reports CTC, NERD, CER, operation metrics, helped cases, and hurt cases using the same image-label pairs

