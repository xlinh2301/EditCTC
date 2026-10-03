# Academic Survey: Scene Text Recognition (STR) Paradigms & Spatial Alignment (2016 – 2026)

This survey examines the structural evolution of Scene Text Recognition (STR) architectures, comparing Connectionist Temporal Classification (CTC), Autoregressive Sequence-to-Sequence (Seq2Seq), and Non-Autoregressive (NAR) refinement models.

---

## 🏛 1. STR Structural Taxonomy & Timeline

```
                                      ┌──────────────────────────────────────────────┐
                                      │        SCENE TEXT RECOGNITION (STR)          │
                                      └──────────────────────┬───────────────────────┘
                                                             │
             ┌───────────────────────────────────────────────┼───────────────────────────────────────────────┐
             ▼                                               ▼                                               ▼
┌─────────────────────────────┐                 ┌─────────────────────────────┐                 ┌─────────────────────────────┐
│  PARADIGM 1: CTC-BASED      │                 │ PARADIGM 2: AUTOREGRESSIVE  │                 │ PARADIGM 3: NON-AUTOREGRESS │
│  (Single-Pass Alignment)    │                 │ (Iterative Step-by-Step)    │                 │ (Parallel Refinement / Edit)│
├─────────────────────────────┤                 ├─────────────────────────────┤                 ├─────────────────────────────┤
│ • CRNN (Shi et al., 2016)   │                 │ • RARE (Shi et al., 2016)   │                 │ • LevOCR (Gu / Da et al.)   │
│ • Rosetta (Borisyuk, 2018)  │                 │ • SAR (Li et al., 2019)     │                 │ • ABINet (Fang et al., 2021)│
│ • SVTR (Du et al., 2022)    │                 │ • RobustScanner (Wang, 2020)│                 │ • VisionLAN (Wang et al.)   │
│ • SVTRv2 (Du et al., 2025)  │                 │ • TrOCR (Li et al., 2023)   │                 │ • EditCTC (2026)            │
│ • PerturbCTC (2026)         │                 │ • ParseQ (Bautista, 2022)   │                 │ • SMART (2025/2026)         │
└─────────────────────────────┘                 └─────────────────────────────┘                 └─────────────────────────────┘
```

---

## 🔬 2. Paradigm Breakdown & Mathematical Mechanics

### Paradigm 1: Connectionist Temporal Classification (CTC)
CTC maps a sequence of frame representations $\mathbf{X} = (\mathbf{x}_1, \dots, \mathbf{x}_T)$ to a sequence of label tokens $\mathbf{y} = (y_1, \dots, y_U)$ ($U \le T$) without requiring explicit character alignment.

1. **Alignment Probability Summation**:
   \[
   P(\mathbf{y} \mid \mathbf{X}) = \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{y})} P(\pi \mid \mathbf{X})
   \]
   where $\mathcal{B}$ is the collapse operator removing consecutive duplicates and blank tokens $\epsilon$.
2. **Conditional Independence Assumption**:
   \[
   P(\pi \mid \mathbf{X}) = \prod_{t=1}^T P(\pi_t \mid \mathbf{x}_t)
   \]
3. **Core Strengths**: Extremely fast inference ($O(T)$ single forward pass), highly parallelizable, no autoregressive error accumulation during deployment.
4. **Core Weakness**: The conditional independence assumption prevents CTC heads from modeling character co-occurrence or resolving fine-grained visual ambiguities ($8 \leftrightarrow 9, 3 \leftrightarrow 8$) when visual feature quality degrades.
5. **Modern Evolution (SVTR $\to$ SVTRv2 $\to$ PerturbCTC)**:
   - *SVTR (2022)*: Replaces CNN-BiLSTM with patch-based Vision Transformer using Local-Global Mixers and height-progressive token merging.
   - *SVTRv2 (2025)*: Optimizes token merging and reduces computational complexity for lightweight mobile inference.
   - *PerturbCTC (2026)*: Injects convolutional feature perturbation during training to regularize CTC alignment, but lacks explicit spatial priors at inference time.

---

### Paradigm 2: Autoregressive Sequence-to-Sequence (Seq2Seq)
Autoregressive models predict tokens sequentially conditioned on all previously generated tokens and the visual feature context $\mathbf{H}$:

1. **Probability Factorization**:
   \[
   P(\mathbf{y} \mid \mathbf{X}) = \prod_{i=1}^U P(y_i \mid y_{<i}, \mathbf{H})
   \]
2. **Attention Mechanism**:
   \[
   \alpha_{i, j} = \frac{\exp(e_{i, j})}{\sum_{k=1}^T \exp(e_{i, k})}, \quad e_{i, j} = \mathbf{v}^T \tanh(\mathbf{W}_s \mathbf{s}_{i-1} + \mathbf{W}_h \mathbf{h}_j + \mathbf{b})
   \]
3. **Core Strengths**: Strong implicit language modeling capability, robust handling of irregular text curvature and arbitrary orientations.
4. **Core Failures in Utility & Meter Reading**:
   - *Attention Drift & Repetition Hallucination*: On repetitive digit sequences (e.g. `00000`), autoregressive attention wanders between identical visual tokens, omitting or duplicating digits.
   - *Inference Latency*: Step-by-step sequential decoding incurs $O(U)$ latency, exceeding edge device power/time budgets.
   - *Error Propagation*: If an early character is mispredicted, subsequent tokens are corrupted.

---

### Paradigm 3: Non-Autoregressive Sequence Refinement (NAR)
Non-Autoregressive models decouple token dependencies from step-by-step sequential generation, predicting all positions in parallel and applying iterative or single-pass edit refinements.

1. **Levenshtein OCR (LevOCR)**:
   - Employs insertion and deletion operations iteratively:
     \[
     \mathbf{y}^{(0)} \xrightarrow{\text{Delete}} \mathbf{y}^{(0.5)} \xrightarrow{\text{Insert}} \mathbf{y}^{(1)}
     \]
   - *Failure*: Unconstrained cross-attention in deletion/insertion heads scatters across identical repetitive digits in meter readings.
2. **Autonomous Bidirectional Alignment (ABINet)**:
   - Enforces bidirectional language model (BCN) refinement on parallel visual predictions.
   - *Failure*: In meter reading (where digits are numerical and lack natural language semantic constraints), pure language models hallucinate incorrect numbers.
3. **Alignment-Guided Refinement (EditCTC)**:
   - **Analytical Gaussian Spatial Prior**: Couples CTC temporal alignment with 2D spatial memory:
     \[
     \mathbf{B}_{i, k} = \lambda \exp\left(-\frac{(u_k - c_i)^2}{2\sigma^2}\right)
     \]
   - **Continuous 4D CTC Uncertainty**: Injects $[p_1, p_2, p_1-p_2, \mathcal{H}(p)]$ into queries.
   - **Decoupled Dual Heads**: Prevents false edits on clean tokens while selectively correcting degraded tokens in a single parallel pass ($O(1)$ latency).

---

## 📊 3. Comparative Spatial Alignment Mechanisms

| Alignment Mechanism | Source Architecture | Mathematical Formulation | Character Box Labels Needed? | Behavior on Repetitive Digits (`00000`) | Receptive Field Focus |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **1D Temporal CTC Peak** | CRNN, SVTR | $c_i = \operatorname{argmax}_t P(y_i \mid \mathbf{x}_t) / T$ | ❌ No | Merged duplicate blanks; struggles on blur | Global 1D stripe |
| **2D Unconstrained Cross-Attn** | TrOCR, LevOCR | $\mathbf{A} = \operatorname{Softmax}(\mathbf{Q}\mathbf{K}^T / \sqrt{d})$ | ❌ No | ❌ Severe scattering across identical zeros | Global unconstrained |
| **2D Gaussian Spatial Bias** | **EditCTC (ARCH-4C)** | $\mathbf{A} = \operatorname{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d}} + \mathbf{B}\right)$ | ❌ **No (Weak CTC)** | **✅ Strict localized visual focus ($3\sigma$)** | **Local $\pm 3\sigma$ coordinate** |
| **Explicit Centroid Heatmap** | CRAFT + AMR | $\mathcal{L}_{\text{hm}} = \text{MSE}(\hat{\mathbf{H}}, \mathbf{H}^*)$ | ⚠️ **Yes (Expensive)** | Moderate; requires segmentation masks | Pixel-level bounding box |
| **Implicit Feature Perturbation**| PerturbCTC | $\mathbf{z}' = \mathbf{z} + \gamma \cdot \operatorname{Conv}(\mathbf{z})$ | ❌ No | Implicit regularizer; no explicit coordinates | Global feature map |
