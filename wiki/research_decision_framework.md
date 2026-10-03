# Research Decision Framework & New Experiment Guidance Engine

This document defines the **Decision Matrix & Hypothesis Engine** for formulating and evaluating new experimental trajectories in OCR and Automatic Meter Reading (AMR). When a new experiment, ablation study, or failure mode is encountered, the agent uses this framework to infer the highest-ROI research direction.

---

## 🧭 1. Research Decision Workflow

```
[ New Experiment / Failure Mode Encountered ]
                      │
                      ▼
[ Step 1: Failure Signature Diagnostic ]
  ├── Symptom A: Repeating digit hallucination (e.g. `00000` -> `0000`)
  ├── Symptom B: Half-digit boundary flip (e.g. `00742.8` read as `00743` before rollover)
  ├── Symptom C: Character stroke ambiguity under glare ($8 \leftrightarrow 9, 3 \leftrightarrow 8, 0 \leftrightarrow 6$)
  └── Symptom D: Over-correction on clean baseline tokens (False Edit Rate > 0.10%)
                      │
                      ▼
[ Step 2: Knowledge Base & Pattern Cross-Referencing ]
  ├── Cross-reference `wiki/patterns/` & `wiki/deep_dive_cnn_transformer_blocks.md`
  └── Query NotebookLM (`nlm query notebook`) or Academic Literature for prior art
                      │
                      ▼
[ Step 3: Architectural Block Selection & Tensor Formulation ]
  ├── 2D Path: PPLCNet / Rep-Conv for zero-latency mobile edge deployment
  ├── 3D Path: Axial Height-Width Factorization for vertical drum roll tracking
  └── 4D Path: Gaussian Spatial Bias ($\lambda, \sigma$) + 4D Continuous Uncertainty Injection
                      │
                      ▼
[ Step 4: Multi-Seed Gating & Verification Protocol ]
  ├── Run 5 seeds (`s1024`, `s2024`, `s3024`, `s4096`, `s8192`)
  └── Verification Gates: Cross-Data Acc ≥ 90.0%, CER ≤ 2.30%, Stability σ ≤ 0.40%, FER ≤ 0.10%
```

---

## 📊 2. Diagnostic & Remediation Decision Matrix

| Observed Failure Signature | Root Cause Analysis | Ground-Truth Knowledge Pattern | Prescribed Architectural Intervention | Target Metric Threshold |
| :--- | :--- | :--- | :--- | :--- |
| **Attention Dispersion on Identical Digits** (`00000`) | Unconstrained cross-attention in Transformer decoder scatters uniformly across identical visual tokens. | [`pattern_spatial_alignment_gaussian_bias.md`](patterns/pattern_spatial_alignment_gaussian_bias.md) | Inject 4D Gaussian Spatial Bias tensor $\mathbf{B}_{i, k} = \lambda \exp\left(-\frac{(u_k - c_i)^2}{2\sigma^2}\right)$ with $\lambda=4.0, \sigma=0.08$. | Zero digit omissions / insertions on repetitive test subsets. |
| **Half-Digit Transition Flip** (Bánh xe quay dở dang) | Discrete 10-class / 20-class classification forces brittle boundaries; high CTC entropy. | [`pattern_half_digit_transition_dynamics.md`](patterns/pattern_half_digit_transition_dynamics.md) | Inject continuous 4D uncertainty $[p_1, p_2, p_1-p_2, \mathcal{H}(p)]$ into decoder queries; verify right-adjacent wheel carry-over. | Reduce carry-over cascade errors by $>80\%$. |
| **Stroke Confusion Under Droplets/Glare** ($8 \leftrightarrow 9, 3 \leftrightarrow 8$) | Downsampling backbone to $1/8$ collapses vertical stroke micro-topology ($2 \times 48 = 96$ tokens). | [`pattern_height_progressive_visual_memory.md`](patterns/pattern_height_progressive_visual_memory.md) | Switch to `SharedHighResVisualMemory` ($1/4$ resolution: $4 \times 96 = 384$ tokens). | Cross-Data CER $\le 2.20\%$. |
| **Over-Correction on Clean Tokens** (Degrading In-Domain Acc) | Single coupled token head modifies correct CTC predictions when visual features are slightly noisy. | [`pattern_decoupled_change_token_heads.md`](patterns/pattern_decoupled_change_token_heads.md) | Decouple into Binary Change Head ($\mathcal{L}_{change}$) and Edit Token Head ($\mathcal{L}_{token}$). | False Edit Rate (FER) $\le 0.08\%$. |
| **Multi-Seed Instability** ($\sigma > 1.0\%$) | Discrete token embedding inputs without soft confidence gradients cause high sensitivity to weight initialization. | [`pattern_continuous_ctc_uncertainty.md`](patterns/pattern_continuous_ctc_uncertainty.md) | Soft continuous confidence gating: $\mathbf{q}_i = \mathbf{E}_{\text{tok}}(s_i) + \tanh(\alpha) \cdot \mathbf{W}_{\text{conf}}\mathbf{c}_i$. | Multi-Seed Standard Deviation $\sigma \le 0.40\%$. |

---

## 🔬 3. Deep Research & Knowledge Update Protocol

When entering a new sub-domain or exploring an uncharted optimization:
1. **Fast-Search Protocol**:
   - Run targeted search queries for recent preprints (arXiv, CVPR, ICCV, ECCV, IEEE TIM, IJDAR).
2. **Deep-Research Protocol (NotebookLM CLI)**:
   - Use `.venv/bin/nlm query notebook <notebook-id> "<in-depth query>"` or `mimi-sdlc` to extract mathematical formulas, source citations, benchmark figures, and dataset breakdowns.
3. **Wiki Knowledge Compilation**:
   - Translate all empirical findings and theoretical formulations into persistent Markdown documents in `wiki/patterns/` or `wiki/`.
   - Compile via `/mnt/d/workspace/sdd/.venv/bin/python -m ai_sdlc.cli wiki compile`.
   - Update Sub-Specs in `.agents/specs/` and commit to Git.
