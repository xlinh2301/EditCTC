## ADDED Requirements

### Requirement: Controlled synthetic seed generation
The training pipeline SHALL generate one-error synthetic seeds from ground truth using substitution, deletion, or insertion, with the changed token guaranteed to differ from the ground-truth token where applicable.

#### Scenario: Substitution corruption
- **WHEN** a substitution sample is generated for ground truth `018527`
- **THEN** the seed differs at exactly one position and the replacement token is not the original token

#### Scenario: Deletion corruption
- **WHEN** a deletion sample is generated
- **THEN** exactly one ground-truth token is removed from the seed

#### Scenario: Insertion corruption
- **WHEN** an insertion sample is generated
- **THEN** exactly one token is inserted at a valid anchor and the resulting alignment is unambiguous

### Requirement: Mixed natural and synthetic training
The training pipeline SHALL support a configurable mixture of natural CTC seeds and synthetic one-error seeds, initially using 50% natural and 50% synthetic samples with equal synthetic operation types.

#### Scenario: Mixed batch is created
- **WHEN** a training batch is assembled
- **THEN** each sample follows the configured natural or synthetic seed path and carries matching seed IDs, seed length, and edit targets

### Requirement: Natural-seed evaluation
Validation and test evaluation SHALL use natural CTC seeds and SHALL not use ground-truth-derived synthetic seeds.

#### Scenario: Generalization is measured
- **WHEN** E3 is evaluated
- **THEN** NERD corrections are measured from the model's natural CTC output against the ground truth

