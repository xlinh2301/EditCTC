# Skill & Architecture Impact Audit Tracker

Audit record of all 52 multi-seed runs and architecture gating decisions for EditCTC.

| Proposal / Model | Variant Description | In-Domain Acc (%) | In-Domain CER (%) | Cross-Data Acc (%) | Cross-Data CER (%) | Gating Outcome | Primary Rationale / Finding |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| `EXP-18B` | Canonical Baseline (No spatial bias) | 93.30 ± 0.32 | 2.32 | 88.27 ± 0.70 | 2.66 | **Baseline** | Baseline reference point (6 seeds) |
| `ARCH-1` | Decoupled Change Head | 92.85 ± 0.23 | 2.39 | 89.22 ± 1.23 | 2.44 | **Accepted** | Drops false edit rate to 0.08% |
| `ARCH-2A` | CTC Confidence 2D ($[p_1, p_2]$) | 92.82 ± 0.36 | 2.41 | 89.00 ± 0.71 | 2.52 | **Accepted** | +0.73% cross-data improvement |
| `ARCH-2B` | Full Confidence 4D | 93.03 ± 0.25 | 2.33 | 89.48 ± 0.92 | 2.39 | **Accepted** | +1.21% cross-data gain |
| `ARCH-3` | Temporal Alignment Embedding | 92.58 ± 0.59 | 2.45 | 89.61 ± 1.20 | 2.35 | **Accepted** | Handles character spacing variations |
| `ARCH-4` 🏆 | Align-Guided Cross-Attention | 93.09 ± 0.30 | **2.31** | **90.22 ± 0.81** | **2.22** | **Accepted (SOTA)** | **Peak 91.35% Cross / 93.33% In-domain** |
| `ARCH-4C` 🛡️ | Align-Guided + 4D Confidence | 93.06 ± 0.26 | 2.33 | **90.27 ± 0.37** | **2.18** | **Accepted (SOTA)** | **Highest stability ($\sigma = \pm 0.37\%$)** |
| `ARCH-5` | Local Visual Refinement Block | 92.89 ± 0.17 | 2.38 | 89.66 ± 0.82 | 2.33 | **Accepted** | High-precision visual features |
| `ARCH-6` | Backbone S5 Unfreeze | 92.65 ± 0.78 | **2.21** | 84.10 ± 0.89 | 3.28 | *Rejected (Cross)* | Overfits to In-domain, collapses Cross |
| `ARCH-7` | Gated Memory Fusion | 92.85 ± 0.23 | 2.39 | 88.68 ± 1.99 | 2.56 | **Accepted** | Multi-modal cross-attention fusion |
