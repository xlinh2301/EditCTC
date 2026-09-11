# Cross-data E36 visual-oracle audit

The existing Cross-data `branch_audit.jsonl` was inspected without training or
checkpoint changes. It contains CTC frame top-1 token, probability, and margin,
but does not contain encoder span features or frame-level Top-3/Top-5 token
identities. Therefore E36-E/F/G's actual visual oracle cannot be recomputed
from this persisted artifact.

The available checkpoint-only proxy is reported in
`results/cross-audit-candidate/report.md`: wrong CTC rows have lower mean
confidence (`0.976`) and minimum emitted margin (`0.793`) than correct rows
(`0.996` and `0.965`), but many wrong rows remain high-confidence. This does
not provide visual discrimination and is not an E36 pass.

Compared with Indomain, the data needed for a valid E36 cross test is missing
from both persisted dumps. A fresh inference dump with `encoder_spans` and
full candidate identities would be required; no such dump was generated in
this evaluation-only loop.
