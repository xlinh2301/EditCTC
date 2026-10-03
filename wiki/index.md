# Project Wiki: EditCTC

> **Persistent Knowledge Base** compiled via **WikiSkill** ([arXiv:2608.27454](https://arxiv.org/html/2608.27454)).
> Systematically compiling architectural innovations, failure patterns, continuous CTC confidence mechanics, and 52-seed empirical benchmarks.

---

## 📚 Table of Contents

### 1. Project Foundations & Theoretical Scope
- [Project Overview](overview.md) — Problem statement, industrial motivation, and OCR pipeline context.
- [System Architecture (ARCH-1 to ARCH-8)](architecture.md) — Alignment-guided cross-attention, Continuous 4D Uncertainty, and decoupled prediction heads.
- [Related Work & Comparative Taxonomy](related_work.md) — Comprehensive comparison against CRNN, SVTRv2, PerturbCTC, LevOCR, and CRAFT heatmaps.
- [Deep Research Synthesis](deep_research_insights.md) — Theoretical landscape, CTC conditional independence, and loss formulations.

### 2. Knowledge & Design Patterns (`wiki/patterns/`)
- [Spatial Alignment & Gaussian Bias](patterns/pattern_spatial_alignment_gaussian_bias.md) — Formulations mapping 1D temporal CTC peaks to 2D spatial coordinates ($\lambda \exp(-\frac{(u_k - c_i)^2}{2\sigma^2})$).
- [Continuous CTC Uncertainty Injection](patterns/pattern_continuous_ctc_uncertainty.md) — 4D confidence vectors ($[p_1, p_2, \text{margin}, \text{entropy}]$) projected into decoder queries.
- [Feature Perturbation vs. Gaussian Bias](patterns/pattern_feature_perturbation_vs_spatial_bias.md) — Comparing PerturbCTC implicit regularization against EditCTC analytical spatial priors.
- [Height-Progressive Merging vs. Visual Memory](patterns/pattern_height_progressive_visual_memory.md) — Resolution trade-offs and micro-stroke topology preservation.
- [Half-Digit Transition Dynamics](patterns/pattern_half_digit_transition_dynamics.md) — Resolving rotating mechanical drum transitions (3-4, 8-9) and carry-over state verification.
- [Decoupled Change vs. Token Heads](patterns/pattern_decoupled_change_token_heads.md) — Binary edit classification vs 97-way token prediction to prevent over-correction.
- [Character Confusion & Glare Mitigation](patterns/pattern_character_confusion_mitigation.md) — Resolving fine stroke ambiguities (8 vs 9, 3 vs 8, 0 vs 6) under outdoor glare/moisture.
- [Non-Autoregressive Gated Decoding](patterns/pattern_non_autoregressive_gated_decoding.md) — Dual threshold inference policy ($P_{change} \ge \tau_{change} \land \Delta P \ge \Delta$).
- [Multi-Seed Evaluation & Gating Protocol](patterns/pattern_multiseed_evaluation_protocol.md) — 52-seed benchmark protocol, in-domain vs. cross-data evaluation methodology.

### 3. Historical Logs & Validation Audit
- [Evolution Logs](logs.md) — Chronological history of ARCH-1 through ARCH-8 iterations, seed runs, and trajectory findings.
- [Skill & Architecture Impact Audit](skill-impact.md) — Quantitative validation scores (In-domain & Cross-data Accuracy/CER) and gating outcomes across 52 seeds.

---
*Compiled via WikiSkill standard | ARIA 6-Layer Compliant*
