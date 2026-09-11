# Cross-data E37 sequence audit

Read-only audit of persisted CTC/NRTR/NERD proposals. No model was fitted and no cross-data row entered training.

- Evaluated rows: `1145`; wrong CTC seeds: `94`.
- Candidate exact coverage on wrong rows: `0.1383`.
- Beneficial candidate oracle coverage on wrong rows: `0.2021`.
- Candidate groups with any beneficial proposal: `19`.

## Transition taxonomy

```json
{
  "correct_to_correct": 1051,
  "correct_to_worse": 4884,
  "wrong_to_better_wrong": 6,
  "wrong_to_correct": 13,
  "wrong_to_same": 94,
  "wrong_to_worse": 376,
  "wrong_to_wrong_same": 77
}
```

The persisted cross audit has no encoder spans or token Top-K probabilities; this limits E37 to the real proposals saved in branch_audit (NRTR full string and NERD token proposals).
