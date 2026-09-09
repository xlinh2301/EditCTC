## 1. Decoder support

- [ ] 1.1 Expose normalized CTC candidate scores and decoded lengths.
- [ ] 1.2 Add optional LCB log-probability reranking with `beta`.
- [ ] 1.3 Preserve exact legacy behavior when `beta=0` or diagnostics are off.

## 2. Audit

- [ ] 2.1 Add per-file CTC versus reranked decision logging.
- [ ] 2.2 Add LCB-helped, LCB-hurt, rerank-rate, and confidence summaries.
- [ ] 2.3 Run beta-zero regression and the beta-0.10 isolated experiment.
