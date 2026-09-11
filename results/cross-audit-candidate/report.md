# Cross-data candidate/error audit

Read-only audit of precomputed `branch_audit.jsonl`; no training or checkpoint changes.

| Metric | Cross-data | Indomain test |
|---|---:|---:|
| Rows | 1145 | 562 |
| CTC exact | 1051 | 538 |
| CTC accuracy | 0.9179 | 0.9573 |
| NRTR accuracy | 0.8891 | 0.9520 |
| NERD final accuracy | 0.9179 | 0.9573 |
| NERD changed | 0 | 0 |
| NERD helped | 0 | 0 |
| NERD hurt | 0 | 0 |
| LCB length accuracy | 0.8646 | 0.8808 |

## Natural error taxonomy

| Category | Cross-data | Indomain test |
|---|---:|---:|
| correct | 1051 | 538 |
| deletion_only | 12 | 4 |
| insertion_only | 52 | 7 |
| mixed_length | 4 | 0 |
| substitution_only | 26 | 13 |

## Aligned character operations

| Operation | Cross-data | Indomain test |
|---|---:|---:|
| D | 18 | 5 |
| I | 56 | 7 |
| M | 5785 | 2564 |
| S | 35 | 15 |

## Interpretation

- The cross split has substantially more natural CTC errors than the indomain split, so it is a stronger stress test for correction.
- The precomputed NERD branch is KEEP-only on both splits; therefore its zero changed/helped/hurt is an inference behavior, not evidence that cross-data has no recoverable errors.
- Top-3/Top-5 candidate coverage cannot be recomputed from these files because only frame top-1 identities are persisted. A fresh inference dump with full top-k is required for that specific E34/E35 oracle.
- Confidence and margin summaries are provided to determine whether cross errors are low-confidence/ambiguous; these are the available posterior diagnostics without retraining.
