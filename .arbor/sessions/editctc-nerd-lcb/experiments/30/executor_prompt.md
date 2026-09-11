Mechanism: Extend branch diagnostics with full frame Top-K ids/probabilities, emitted-token frame spans, and aligned feature references.
Hypothesis: Explicit CTC candidate coverage will identify whether future correction failure is proposal-limited or verifier-limited.
Observable: Top-3/5 token recall, fully-fixable sequence coverage, length/substitution taxonomy, and leakage checks on Indomain/Cross.
Conflicts: Cross is evaluation-only; do not train, alter labels, or run B_test.
