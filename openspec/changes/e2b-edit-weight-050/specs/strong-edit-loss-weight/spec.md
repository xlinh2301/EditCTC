## ADDED Requirements

### Requirement: Strong edit-loss weight
The E2b training configuration SHALL use the E1 balanced edit sampler and set `weight_edit` to `0.50`, with all other architecture and loss settings held constant.

#### Scenario: E2b configuration is isolated
- **WHEN** E2b is launched
- **THEN** the only intended model-training difference from E1 is `weight_edit: 0.50`

### Requirement: Harm-aware evaluation
E2b SHALL report whether stronger edit supervision increases useful corrections more than harmful edits.

#### Scenario: Strong weighting is assessed
- **WHEN** E2b evaluation completes
- **THEN** the report includes `nerd_helped`, `nerd_hurt`, edit precision, error-fix rate, CER, and exact-match accuracy

