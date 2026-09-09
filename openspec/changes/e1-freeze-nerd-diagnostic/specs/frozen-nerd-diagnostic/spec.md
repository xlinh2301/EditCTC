## ADDED Requirements

### Requirement: Frozen CTC representation
The diagnostic run SHALL freeze the CTC encoder/backbone and CTC head from the E0 initialization while training the NERD edit head with E1 sampling.

#### Scenario: Parameters remain frozen
- **WHEN** a training step completes
- **THEN** CTC encoder and CTC head parameters have no optimizer updates

### Requirement: Stable natural seed
The diagnostic SHALL use a fixed CTC seed for each training image across epochs.

#### Scenario: Same image is revisited
- **WHEN** an image is loaded in two different epochs
- **THEN** its seed IDs and seed length are unchanged unless the diagnostic explicitly regenerates the frozen baseline

### Requirement: Diagnostic interpretation
The run SHALL report NERD operation recall and final changed/helped/hurt counts separately from CTC accuracy.

#### Scenario: NERD remains inactive
- **WHEN** the frozen run predicts only KEEP
- **THEN** the result identifies edit-head supervision or target construction as the remaining suspect rather than moving-backbone instability

