## ADDED Requirements

### Requirement: Reproducible s1024 baseline
The evaluation workflow SHALL run checkpoint `s1024` on Cross-data and Indomain test data with fixed dataset paths, configuration, and checkpoint paths.

#### Scenario: Baseline run completes
- **WHEN** the baseline Slurm job is submitted with the recorded configuration
- **THEN** it produces one summary and one prediction file for each dataset with the number of evaluated and skipped samples

### Requirement: Branch audit baseline
The workflow SHALL record CTC, NRTR, LCB length, NERD seed, NERD refined output, and operation decisions for every evaluated image.

#### Scenario: Every evaluated image is auditable
- **WHEN** an image is processed successfully
- **THEN** its branch outputs and operation details are present in the append-only audit file

### Requirement: Wrong-case baseline
The workflow SHALL record every image whose final NERD output differs from ground truth, including the full branch audit for that image.

#### Scenario: Wrong output is retained
- **WHEN** final decoded text does not equal its label
- **THEN** the image filename, ground truth, branch correctness flags, and audit payload are written to the wrong-case log

