# Cross-data E37 sequence audit

Read-only audit of persisted CTC/NRTR/NERD proposals. No model was fitted and no cross-data row entered training.

- Evaluated rows: `562`; wrong CTC seeds: `24`.
- Candidate exact coverage on wrong rows: `0.2083`.
- Beneficial candidate oracle coverage on wrong rows: `0.2917`.
- Candidate groups with any beneficial proposal: `7`.

## Transition taxonomy

```json
{
  "correct_to_correct": 538,
  "correct_to_worse": 2196,
  "wrong_to_better_wrong": 2,
  "wrong_to_correct": 5,
  "wrong_to_same": 24,
  "wrong_to_worse": 82,
  "wrong_to_wrong_same": 18
}
```

The persisted cross audit has no encoder spans or token Top-K probabilities; this limits E37 to the real proposals saved in branch_audit (NRTR full string and NERD token proposals).
