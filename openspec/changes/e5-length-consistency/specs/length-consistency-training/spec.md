## ADDED Requirements

### Requirement: Differentiable expected-length consistency
The training loss SHALL include an optional small consistency term comparing target length with seed length plus expected INSERT_AFTER count minus expected DELETE count.

#### Scenario: Expected edit length is computed
- **WHEN** NERD produces operation probabilities for a seed
- **THEN** the consistency term uses expected insertion and deletion probabilities over valid seed positions

### Requirement: Configurable low consistency weight
The consistency term SHALL be disabled by default outside E5 and SHALL support an initial weight in the range 0.02 to 0.05.

#### Scenario: E5 weight is selected
- **WHEN** E5 is configured
- **THEN** the consistency weight is explicitly recorded and is no larger than 0.05 for the first run

### Requirement: Consistency diagnostics
The report SHALL record the consistency loss, predicted length residual, and final helpful/harmful edit counts.

#### Scenario: Length coupling is evaluated
- **WHEN** E5 evaluation completes
- **THEN** the report shows whether improved length agreement corresponds to useful NERD corrections rather than over-editing

