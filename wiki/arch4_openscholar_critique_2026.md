# ARCH-4 Critique & 2026 Literature Gap Analysis (via OpenScholar)

> Compiled via `brain scholar query` (OpenScholar: Semantic Scholar + arXiv retrieval, Agent-as-LLM synthesis).
> Grounded in direct source-code audit of `ppocr/modeling/heads/rec_edit_refine_nrtr_head.py`.
> Raw synthesis outputs: `scratch/openscholar_arch4_query{1..5}.md`.

---

## 1. ARCH-4 Architecture Recap

**Pipeline**: `PPLCNetV4 (frozen) → lightSVTR+CTCHead → Greedy CTC peak extraction → Gaussian Spatial Bias → 4-layer NRTR cross-attention decoder → Decoupled Change/Token heads → Gated substitution`

**Core mechanism** (`_build_spatial_cross_bias`, L548-633): maps each CTC-emitted peak timestep $t_j \in [0, 39]$ to a normalized horizontal coordinate $c_j = t_j / 39$, and injects an additive Gaussian bias into cross-attention logits:

$$\mathcal{B}_{j,k} = \lambda \cdot \exp\left(-\frac{(u_k - c_j)^2}{2\sigma^2}\right), \quad \lambda=1.0,\ \sigma=0.15$$

This restricts each refined character query to attend mostly to visual memory tokens physically near its CTC-predicted horizontal position, preventing attention dispersion across repeated/confusable digits — the key reason ARCH-4 hits **91.35% cross-data accuracy**.

---

## 2. Weaknesses (Grounded in Code Audit)

| # | Weakness | Evidence (file:line) | Why it matters |
|---|---|---|---|
| 1 | **CPU-bound per-sample Python loop** building the bias matrix | `_build_spatial_cross_bias` L603-631: `for b in range(bsz): ... for j in range(1, n+1): ...` with `.numpy()` round-trips | Breaks graph/AMP fusion, forces GPU↔CPU sync every forward pass, blocks `paddle.jit`/`torch.compile`-style tracing, scales linearly with batch×seq-len in pure Python |
| 2 | **Fixed, hand-tuned scalars** $\lambda=1.0$, $\sigma=0.15$ | Same function, config keys `align_spatial_weight`, `align_spatial_sigma` | Not learned; identical sharpness regardless of per-character ambiguity (a confidently-localized CTC peak and a blurry one get the same window width) |
| 3 | **1D horizontal-only prior — no vertical/row awareness** | L569-573: `u_col` is tiled `vis_h` (3) times identically — every row gets the *same* horizontal coordinate | The high-res visual memory is a genuine 2D grid (4×96), but the bias collapses all 3 rows to 1D, discarding vertical information useful against glare bands / dial-needle occlusion that is row-localized |
| 4 | **Inherits CTC's localization errors, not just its classification errors** | Bias is centered strictly at the greedy CTC peak `t_j`; there's no fallback if the peak itself is spatially wrong | ARCH-4 is designed to fix *character identity* errors, but if CTC's temporal alignment drifts (severe blur), the Gaussian bias actively *reinforces* attention to the wrong image region — a failure mode the architecture cannot self-correct |
| 5 | **Hyperparameter sprawl, inconsistent treatment of tail/append tokens** | 3 separate Gaussian regimes: `align_spatial_*` (seed tokens), `tail_spatial_*`, `tail_prog_*` (append slot) — L594-613 | 6 hand-tuned constants instead of one unified, learnable spatial-prior module; append-token logic duplicates the seed-token logic with different math |
| 6 | **Static, hand-set gating thresholds** | `edit_gate_threshold`, `edit_delta_threshold`, `edit_op_threshold`, `append_gate_threshold` — all fixed scalars in `forward()` L959-1052 | No calibration against domain shift (indomain vs cross-domain have different error/confidence distributions); thresholds tuned once, not adapted per-domain or per-sample |
| 7 | **Frozen backbone ⇒ representation ceiling** | `PPLCNetV4 (Frozen)` in README; ARCH-6 (unfreeze Stage-5) trades −5.9pp cross-domain for +0.1pp in-domain CER | Visual features are never jointly optimized for the refinement task; the in-domain/cross-domain tension (ARCH-6: 93.? in-domain but only 84.10% cross) suggests an unresolved domain-generalization gap in the frozen backbone |
| 8 | **Single-token append limits multi-character correction** | `_insert_tail_slot` only ever inserts *one* dummy slot; only ARCH-8 extends to full DELETE/INSERT_AFTER | ARCH-4 itself cannot correct CTC's merged/dropped-duplicate errors in the *middle* of a sequence (e.g., a missed repeated digit), only at the tail |
| 9 | **No uncertainty-aware $\sigma$** | $\sigma_{\text{align}}$ constant across all samples/positions | A confidently-localized peak (high CTC margin) should get a *narrow* window; an ambiguous peak should get a *wider* search window — ARCH-4C adds 4D confidence to the query embedding but never couples it back into the spatial bias width itself |
| 10 | **Fixed $T=40$ timestep normalization** | `max_ctc_timesteps` hardcoded to 40 (tied to 320px/stride-8 input) | Normalization constant `t_j / 39` breaks if input resolution/aspect ratio changes without re-deriving the backbone stride ratio |

---

## 3. OpenScholar Findings (5 Queries, Semantic Scholar + arXiv, 2025–2026 bias)

**Honest-gap queries** (1, 2 — direct Gaussian-bias-in-NAR-OCR, and 2D/row-aware positional priors for STR): OpenScholar found **no literature directly matching** these niche combinations — confirming ARCH-4's exact formulation (temporal-peak-to-spatial-coordinate Gaussian injection) is itself a fairly novel, under-explored contribution worth emphasizing in the paper's novelty claims.

**Productive queries** (3, 4, 5) surfaced real, citable related work:

| Paper | Year | Relevance to ARCH-4 weakness |
|---|---|---|
| **Bayes Risk CTC** (Tian et al.) [arXiv:2210.07499] | 2022 | Directly addresses weakness #4 — controllable/customizable CTC alignment paths instead of trusting raw greedy peaks. Could replace greedy `ctc_seed_and_conf` peak extraction with a risk-calibrated alignment that is more robust before the Gaussian bias is even computed. |
| **PC-MLM / Deletable PC-MLM** (Futami et al.) [arXiv:2209.04062] | 2022 | Near-exact ASR analog of ARCH-4's philosophy: mask low-confidence greedy CTC outputs, predict corrections conditioned on unmasked tokens + auxiliary phone (≈ confidence) signal. Validates the core NAR-refinement-over-CTC design pattern from an independent domain. |
| **IPAD** (Yang et al.) [arXiv:2312.11923] | 2023 | Iterative, parallel, **discrete-diffusion**-based decoding with an "easy-first" strategy for STR. Suggests ARCH-4's single-pass refinement could become multi-step iterative refinement (re-run the decoder 2-3 times, committing high-confidence edits first) — directly targets weakness #8 (multi-character correction) without going fully autoregressive. |
| **PIMNet** (Qiao et al.) [arXiv:2109.04145] | 2021 | **Mimicking learning**: train the NAR decoder to mimic an auxiliary autoregressive teacher, then detach the AR branch at inference. Could calibrate ARCH-4's decoder without inference-time cost — addresses weakness #6/#9 (better-calibrated confidence via distillation). |
| **CAM: Class-Aware Mask-Guided Feature Refinement** (Yang et al.) [arXiv:2402.13643] | 2024 | Canonical glyph masks from standard fonts fused with visual features to disambiguate confusable glyphs. Directly applicable to EditCTC's own documented pain point (`pattern_character_confusion_mitigation.md`: 8↔9, 3↔8) — an explicit glyph-shape prior as a *second* bias term alongside the spatial Gaussian bias. |
| **ESTR-CoT** (Wang et al.) [arXiv:2507.02200] | 2025 | Event-stream + chain-of-thought reasoning for low-illumination/high-motion STR — a heavier-weight direction, useful mainly as a "what large multimodal models are doing" contrast point for the paper's related-work section (ARCH-4 stays lightweight/edge-deployable by comparison). |

**Confirmed gap** across all 5 queries: **no existing work couples a *learnable, confidence-adaptive* Gaussian/spatial bias with CTC-guided non-autoregressive refinement specifically for digit/meter OCR.** This is good news for the paper's novelty section — the gap search itself is evidence to cite.

---

## 4. Concrete Recommendations for 2026 (mapped to weaknesses)

1. **Vectorize `_build_spatial_cross_bias`** — replace the Python double-loop with batched `paddle` broadcasting ops (no `.numpy()` round-trip). Pure engineering fix, zero accuracy risk, likely a real throughput win (addresses #1).
2. **Learnable, confidence-adaptive $\sigma$** — make $\sigma_j$ a function of the existing 4D CTC confidence vector (`margin`, `entropy`) already computed in ARCH-2B/4C: $\sigma_j = \sigma_0 \cdot f(\text{margin}_j, \mathcal{H}_j)$, e.g. wider window when entropy is high. This is the single highest-leverage addition — fuses ARCH-4 + ARCH-4C into one principled mechanism instead of two parallel variants (addresses #2, #9).
3. **Add a row/vertical coordinate term** — extend $u_k$ to a true 2D $(u_k, v_k)$ and use an anisotropic or 2D Gaussian bias, especially since the high-res visual memory already has 3 rows available but currently ignores row identity (addresses #3).
4. **Alignment-robustness fallback** — borrow Bayes-Risk-CTC-style controllable alignment, or simply widen $\sigma$ dynamically when the top-2 CTC margin is low (same signal already computed), so a wrong peak gets a softer, more forgiving prior rather than a hard wrong-anchor (addresses #4).
5. **Glyph-shape prior as a second bias term** (à la CAM) — directly targets the project's own documented 8↔9/3↔8 confusion pattern; orthogonal to the spatial bias, so it composes additively on the attention logits.
6. **Collapse the 3 separate hand-tuned Gaussian regimes (align/tail/prog) into one unified, position-conditioned bias generator** — reduces hyperparameter count from 6+ to a small learned module, simplifying ablations for the paper (addresses #5).
7. **Mimicking-learning calibration** (PIMNet-style) — train with an AR teacher branch during training only, distill into the NAR decoder's confidence outputs to make gating thresholds more reliable without inference cost (addresses #6, #9).
8. **Lightweight 2-pass iterative refinement** (IPAD-style easy-first) as an optional "ARCH-4-iter" variant — re-feed the once-refined sequence through the same decoder once more, committing only the highest-confidence edits first, to reach middle-of-sequence multi-character fixes without a full insert/delete op head (addresses #8, complements existing ARCH-8).

These map cleanly onto a natural **ARCH-9 (Adaptive Spatial Bias)** and **ARCH-10 (Iterative Glyph-Aware Refinement)** follow-up pair for the paper's "future work" or a final ablation round before submission.
