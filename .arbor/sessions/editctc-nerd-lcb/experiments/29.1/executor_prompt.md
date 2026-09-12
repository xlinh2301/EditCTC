## Codebase

Working directory: /datastore/cndt_thangcpd/linhtruong/workspace5/release_EditCTC/code

## Research Idea

**ID**: 29.1
**Hypothesis**:

Mechanism: Use frozen CTC collapsed spans to blur one character online before EditCTC encoding.
Hypothesis: Targeted blur creates authentic one-token CTC errors and reduces edit-head identity overfit while preserving clean examples.
Observable: increased natural wrong-seed rate, nonzero correction activation, helped>hurt on held-out Indomain; CTC baseline should not collapse.
Conflicts: blur may fail to change CTC seed, span index may misalign after augmentation, and too much blur may teach denoising rather than correction.

## Execution

Run `scripts/e41_ctc_span_index_job.sh` first, then `scripts/e41_nrtr_blur_online_train.sh`. Keep Cross-data evaluation-only. Report wrong-seed rate, edit operation rates, validation accuracy, helped/hurt, and compare with E41 30-epoch baseline.
