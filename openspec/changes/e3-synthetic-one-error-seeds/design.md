## Context

The natural CTC seed is correct for most training samples, making edit targets overwhelmingly KEEP. E3 supplies direct one-operation examples while retaining natural seeds so that the edit decoder learns a correction task without abandoning deployment conditions.

## Goals / Non-Goals

**Goals:**

- Teach NERD to perform each supported operation from a known one-error seed.
- Keep the evaluation protocol natural and comparable to E0–E2.
- Make corruption deterministic under the experiment seed.

**Non-Goals:**

- No multi-error corruption in the first E3 run.
- No inference-time threshold or beam-search changes.

## Decisions

- Generate synthetic seeds from encoded ground truth only in training.
- Use 50% natural and 50% synthetic samples initially; split synthetic samples evenly across substitution, deletion, and insertion.
- Prefer corruption locations and tokens that avoid ambiguous Levenshtein alignments.
- Pass synthetic seed IDs and lengths through the existing externally supplied seed path, while preserving natural CTC seed generation.

## Risks / Trade-offs

- [Synthetic distribution differs from real CTC errors] → Retain half natural samples and evaluate only natural seeds.
- [Teacher corruption can overfit] → Limit each string to one error and later add confusion-conditioned corruptions only if needed.
- [Insertion/deletion target ambiguity] → Validate reconstructed operations before training.

