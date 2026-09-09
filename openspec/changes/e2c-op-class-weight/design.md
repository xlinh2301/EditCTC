## Context

E1 guarantees that edit positions are selected, but edit types remain uneven: REPLACE is more common than DELETE and INSERT_AFTER, and all are rare relative to KEEP. E2c applies a deliberately mild class-weighting control before trying focal loss or inverse-frequency weights.

## Goals / Non-Goals

**Goals:**

- Improve recall of rare edit operations.
- Attribute any change to moderate operation weighting.

**Non-Goals:**

- No focal loss, raw inverse-frequency weighting, sampler change, synthetic seed, or inference change.

## Decisions

- Reuse the best edit-loss weight from E2a/E2b rather than sweep it again.
- Apply weights only to operation classification; token loss remains unchanged.
- Measure operation precision and recall before judging final exact-match gain.

## Risks / Trade-offs

- [Overprediction of edits] → Track operation precision and `nerd_hurt` separately from recall.
- [Rare-class variance] → Report raw support counts and repeat promising settings across seeds.

