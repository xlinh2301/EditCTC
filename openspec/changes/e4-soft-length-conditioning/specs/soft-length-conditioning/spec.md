## ADDED Requirements

### Requirement: Soft LCB length embedding
NERD SHALL condition each seed position on the expected LCB length embedding under the normalized probability distribution over valid lengths, rather than on the embedding of a single argmax class.

#### Scenario: Ambiguous length prediction
- **WHEN** LCB assigns probability to multiple valid lengths
- **THEN** NERD receives their probability-weighted embedding rather than only the most probable length embedding

### Requirement: Unused length class exclusion
The soft length distribution SHALL exclude class zero before normalization because class zero is reserved and is not a valid target length.

#### Scenario: Length distribution is normalized
- **WHEN** length logits are converted to probabilities
- **THEN** class zero has zero probability and valid classes sum to one

### Requirement: Differentiable LCB-to-NERD path
The soft length path SHALL preserve gradients from the NERD edit objective into the LCB length distribution.

#### Scenario: Edit loss backpropagates
- **WHEN** NERD edit loss is backpropagated through soft length conditioning
- **THEN** LCB length-conditioning parameters receive a nonzero gradient when the path is active

