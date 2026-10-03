# Wiki Evolution Logs: EditCTC

Continuous audit trail of model development, multi-seed training iterations, and empirical discoveries.

---

## [2026-10-02 18:30:00Z] ARCH-1 to ARCH-4 Exploration & Discovery
- **Author**: Autonomous Research Agent / Engineering Lead
- **Key Insight**: Standard cross-attention without spatial priors suffers from attention dispersion on repetitive digits (e.g., `00000`).
- **Breakthrough**: ARCH-4 Alignment-Guided Gaussian Bias ($\lambda=4.0, \sigma=0.08$) boosts cross-data accuracy from $88.27\%$ to **$91.35\%$**, reducing cross-data CER to $1.92\%$.

---

## [2026-10-02 21:00:00Z] ARCH-4C: 4D Continuous Uncertainty Integration
- **Author**: Autonomous Research Agent
- **Key Insight**: Adding 4D continuous uncertainty $[p_1, p_2, p_1-p_2, \text{entropy}]$ to decoder queries drastically reduces seed variance across 5 seeds from $\pm 0.81\%$ down to **$\pm 0.37\%$**.
- **Decision**: Promoted ARCH-4 and ARCH-4C as the primary recommendation for production deployment.

---

## [2026-10-03 00:30:00Z] Model Zoo & Checkpoint Archival
- **Author**: System Engineer
- **Action**: Published top-10 checkpoints across all architecture variants to Google Drive and generated `MODEL_ZOO_AND_DATA.md`.
- **Status**: Ready for production deployment and paper submission.
