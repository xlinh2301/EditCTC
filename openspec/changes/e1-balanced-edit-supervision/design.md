## Context

The uncertainty-gated loss currently selects a random 10% subset of valid seed positions. On the E0 evaluation audit, non-KEEP targets are only 1.87% of Cross-data seed positions and 1.04% of Indomain seed positions. The current `KEEP` policy therefore receives much more reliable supervision than the correction operations.

## Goals / Non-Goals

**Goals:**

- Remove random omission of edit targets.
- Keep the architecture, `weight_edit: 0.15`, LCB, and inference decoder unchanged.
- Make operation activation and recall observable.

**Non-Goals:**

- No argmax, beam-search, threshold, or operation-application change.
- No synthetic seed generation or class weighting in E1.

## Decisions

- Construct `valid_mask` before deriving `edit_mask` and `keep_mask` so `IGNORE_INDEX` cannot become a false edit.
- Select all edit positions and sample only KEEP positions with `keep_to_edit_ratio: 2.0`.
- Normalize operation loss over the selected mask and keep token loss ignore behavior unchanged.
- Add E1-specific config and Slurm entry point while preserving E0 initialization.

## Risks / Trade-offs

- [Small selected set] → Log selected counts and use a minimum one KEEP position only for batches with no edit target.
- [More edit pressure] → Keep the same model and loss weight so any change can be attributed to sampling.
- [Random variance] → Run one diagnostic seed first, then repeat promising settings across multiple seeds.

