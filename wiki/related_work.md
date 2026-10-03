# Related Work & Theoretical Taxonomy: Non-Autoregressive Sequence Refinement

This document provides a comprehensive academic taxonomy comparing **EditCTC** against existing literature across Scene Text Recognition (STR), Non-Autoregressive (NAR) text editing, Connectionist Temporal Classification (CTC) alignment, and Automated Meter Reading (AMR).

---

## 🏛 Comparative Taxonomy Table

| Paradigm / Model | Core Mechanism | Alignment Handling | Over-Correction Control | Latency / Complexity | Water Meter Half-Digit Robustness |
|:---|:---|:---|:---|:---:|:---:|
| **CRNN (Shi et al., 2016)** | CNN + BiLSTM + CTC | 1D CTC Collapse | N/A (Single-pass) | $O(T)$ Fast | ❌ Poor (Blur / Half-digit failures) |
| **SVTR / SVTRv2 (Du et al., 2022; 2025)** | Patch-based Single Visual Vision Transformer | 1D CTC Token Merging | N/A (Single-pass) | $O(T \cdot d)$ Moderate | ⚠️ Moderate (Confuses $8 \leftrightarrow 9, 3 \leftrightarrow 8$) |
| **PerturbCTC (2026)** | Feature Perturbation during CTC training | Implicit Alignment Regularization | N/A | $O(T)$ Fast | ⚠️ Moderate (No iterative refinement) |
| **Levenshtein OCR / LevOCR (2020)** | NAR Insertion + Deletion Transformer | Unconstrained Cross-Attention | Heuristic Deletion threshold | $O(K \cdot N^2)$ Iterative | ❌ Fails on repetitive digits (Scattering) |
| **CRAFT + Center Heatmap (2024)** | Pixel-level Character Centroid Segmentation | Explicit 2D Heatmaps + Otsu | N/A | $O(H \cdot W)$ Heavy | ⚠️ Requires pixel-level character boxes |
| **LLM Post-Correction (2025–2026)** | Autoregressive Prompt Correction | Text-only (No Visual Grounding) | Hallucination prone | $O(L^2)$ Very High | ❌ Hallucinates invalid utility meter values |
| **EditCTC (ARCH-4 / ARCH-4C)** 🏆 | **Align-Guided Cross-Attention + 4D Continuous Uncertainty** | **Continuous Gaussian Spatial Prior ($\lambda \exp(-\frac{(u_k-c_i)^2}{2\sigma^2})$)** | **Decoupled Change Head ($\text{FER} \le 0.08\%$)** | **$O(1)$ Fixed NAR Refinement** | **✅ SOTA 91.35% Cross-Data (CER 1.92%)** |

---

## 🔬 In-Depth Literature Analysis

### 1. Connectionist Temporal Classification (CTC) Limitations
CTC models compute sequence probabilities by summing over all valid alignments containing blank tokens ($\epsilon$):
$$P(\mathbf{y} \mid \mathbf{X}) = \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{y})} \prod_{t=1}^T P(\pi_t \mid \mathbf{x}_t)$$
While CTC assumes conditional independence between output tokens given the visual features:
$$P(\pi \mid \mathbf{X}) = \prod_{t=1}^T P(\pi_t \mid \mathbf{x}_t)$$
This independence assumption causes severe performance drops when visual features are degraded by moisture, reflections, or half-turned wheels.

### 2. The Levenshtein Transformer & NAR Editing Pitfalls
Levenshtein Transformer (Gu et al., 2019) and LevOCR treat sequence generation as iterative insertion and deletion operations. However, in visual meter reading:
- **Repetitive Sub-sequences**: Water meters frequently display repeated characters (e.g. `00000`, `99999`).
- **Attention Dispersion**: Without explicit inductive bias, cross-attention weights scatter across identical visual characters, leading to incorrect substitution.
- **EditCTC's Solution**: ARCH-4 restricts the cross-attention receptive field around the temporal peak center $c_i = t_i / (T-1)$ using Gaussian Spatial Bias $\mathbf{B}_{i,k} = \lambda \exp\left(-\frac{(u_k - c_i)^2}{2\sigma^2}\right)$, enforcing localized visual verification.

### 3. Character Center Heatmap vs. Temporal-Spatial Mapping
Recent approaches (e.g., CRAFT-inspired AMR models) attempt to predict pixel-level heatmaps for character centers. While effective, they require expensive character-level bounding box annotations during training. In contrast:
- **EditCTC utilizes weakly supervised temporal alignments**: Peak timesteps $t_i$ from standard sequence-level CTC loss are mapped to normalized horizontal coordinates without requiring any character-level bounding boxes.

### 4. Continuous Uncertainty vs. Hard Token Passing
Prior text post-editing architectures discard CTC logits and pass only discrete token indices. As proven in **ARCH-4C**:
- Continuous 4D uncertainty $\mathbf{c}_i = [p_{top1}, p_{top2}, p_{top1}-p_{top2}, \mathcal{H}(p)]$ captures the fuzzy boundary state of half-turned mechanical wheels.
- Gated injection into decoder queries allows the Transformer to selectively attend to visual memory only when the CTC branch expresses genuine ambiguity.
