# EditCTC ARCH-4: 2020-2026 Trend Analysis & Experiment Proposals

> Compiled via **OpenScholar** (`brain scholar query` — Semantic Scholar + arXiv synthesis) combined with direct **arXiv/web search** cross-checks. NotebookLM integration attempted but session auth could not be completed interactively this round (see §4 note) — findings below rely on OpenScholar + web search only, which already surfaced several papers from **Aug–Sep 2026**.
>
> Companion document: [`arch4_openscholar_critique_2026.md`](arch4_openscholar_critique_2026.md) (architecture + code-level weaknesses). This document focuses on **year-by-year field trends** and **concrete experiment proposals**.

---

## 1. Year-by-Year Trend Timeline (Non-AR Sequence Refinement / CTC-based OCR)

| Year | Trend | Representative Work | Relevance to EditCTC |
|---|---|---|---|
| 2019-2020 | NAR editing via insert/delete ops | Levenshtein Transformer (Gu 2019), LevOCR [arXiv:2209.03594] | Direct ancestor of ARCH-8's 4-way op head; LevOCR already identified as a comparison baseline in `related_work.md` |
| 2021 | Mimicking/distillation NAR training | PIMNet [arXiv:2109.04145] | AR-teacher → NAR-student distillation, unused in EditCTC so far |
| 2022 | Parallel + iterative decoding; controllable CTC alignment; ASR-side masked correction | IPAD [2312.11923] (announced 2022, published 2023), Bayes Risk CTC [2210.07499], PC-MLM [2209.04062] | All three validate EditCTC's "refine a greedy CTC hypothesis" philosophy from independent angles (vision + speech) |
| 2024 | Glyph-shape priors; decoder-scaling > encoder-scaling | CAM [2402.13643], CSD/Samsung CVPR'25 (submitted 2024) | CAM directly targets EditCTC's own documented 8↔9/3↔8 confusion; CSD's "decoder scaling matters more" finding questions ARCH-4's frozen-backbone design |
| 2025 | VLM/LLM + physical reasoning for meters; event-stream CoT STR; multilingual MoE scaling | DialBench/MRLM [2511.21982], ESTR-CoT [2507.02200], ScriptMoE [2609.24058] | MRLM shows the *meter-reading* field is moving toward large VLMs with explicit geometric/causal cross-attention fusion — a heavier-weight contrast point for EditCTC's lightweight edge positioning |
| **2026** | **Diffusion-based NAR decoding solves "overconfidence"; CTC validated as best fix for rare/compositional failures; CTC-seeded edit-refinement reaches speech domain; confidence-routed ensembles** | **MDiff4STR (AAAI'26)**, **"Can STR Read Rare Compositions?" [2609.00816]**, **CTC-Seeded Token Edit Refinement for NAR ASR [2606.28732]**, NAR-MBR Decoding [2606.17537], Confidence-Aware Ensemble (WordArt-V1.5, SIBGRAPI'26) [2608.29970] | **See §2 — these are the highest-value 2026 findings for EditCTC specifically.** |

**Macro-trend read-out**: the field is converging on exactly EditCTC's bet — *keep CTC as the fast, parallel backbone and spend the extra compute budget on a confidence-aware non-autoregressive refinement stage* — rather than moving to full autoregressive or giant VLM decoders. The 2026 rare-composition stress-test paper [2609.00816] is notable because it independently concludes, across 9 recognizers and 4 writing systems, that **switching an AR decoder to CTC decoding is the only architectural change that reliably helps** on the hardest failure mode (rare/compositional inputs) — i.e., direct external validation of EditCTC's core premise.

---

## 2. Three 2026 Papers Worth Direct Engagement

### 2.1 MDiff4STR (AAAI 2026) — Mask Diffusion for STR
- **What**: First application of Mask Diffusion Models to STR. Identifies two failure modes of vanilla diffusion-NAR: (a) train/inference noising mismatch, (b) **overconfident but incorrect predictions**. Fixes #2 with a "token-replacement noise" mechanism that forces the model to reconsider confident-but-wrong tokens during training.
- **Relevance**: ARCH-4's `edit_gate_threshold` / `edit_delta_threshold` are static, and the change head is known to need careful focal-loss tuning ($\alpha_t=0.75$) specifically to avoid over/under-firing — this is the *same* overconfidence problem MDiff4STR targets with a training-time mechanism instead of a tuned inference threshold.

### 2.2 CTC-Seeded Token Edit Refinement for NAR ASR (2026)
- **What**: Formulates ASR decoding as "variable-length edit refinement of a greedy CTC hypothesis" via an **Edit Flow decoder** predicting insert/delete/substitute in parallel, trained with a continuous-time discrete diffusion loss, and **constrains edit proposals using CTC confidence**. Reports that just **2 edit steps** yield large WER drops.
- **Relevance**: This is close to a line-for-line description of EditCTC's own design philosophy (`seed → edit ops → refined sequence`), independently arrived at in the speech domain in 2026. It validates the architecture family and offers two concrete, citable techniques EditCTC doesn't yet use: (a) a diffusion-style training loss instead of pure cross-entropy, (b) an explicit **CTC-confidence-constrained edit proposal** rule (ARCH-4 uses confidence only in the query embedding, not to gate *which positions* are even eligible for edits).

### 2.3 "Can Scene Text Recognition Read Rare Compositions?" (Sep 2026)
- **What**: Large stress-test (9 recognizers × 4 scripts) showing accuracy gains from 6× model scaling do **not** fix the rare-composition failure corner, but switching AR→CTC decoding (SVTRv2) does (+2.5pp, p=0.02).
- **Relevance**: External, large-scale evidence that EditCTC's core bet (CTC backbone, not AR) is the right one for robustness — strong citation for the paper's introduction/motivation section. Also flags that **scaling alone won't fix water-meter half-digit/rare-transition cases** — reinforces that EditCTC should keep investing in the refinement mechanism rather than just scaling PPLCNetV4/NRTR.

---

## 3. Confirmed Field Gap (Novelty Anchor)

Across all 8 OpenScholar queries run (this session + previous), **no retrieved paper combines**: (a) a *learnable, confidence-adaptive* spatial/Gaussian bias, (b) CTC-peak-to-pixel-coordinate mapping, and (c) application to **mechanical digit-wheel / seven-segment water-meter OCR** specifically. The AMR-focused query explicitly flagged this as an open gap ("current meter reading literature primarily targets pointer and dial meters... leaving a gap... tailored to mechanical rolling counter wheels... under harsh environmental degradation"). This triple combination is EditCTC's clean novelty claim.

---

## 4. NotebookLM Status Note

`nlm login --wsl --storage file` was re-run this session; `login --check` reports valid credentials, but `nlm notebook list` still returns an auth-expired error (cookie/token refresh race), and a fresh interactive `--force` login opened a real Chrome window on the Windows host awaiting manual Google sign-in that timed out before completion. **Action needed from you**: finish the Google sign-in in the Chrome window (or run `cd /mnt/d/workspace/sdd && .venv/bin/nlm login --wsl --storage file --force` yourself from a terminal where you can interact with the popped-up browser), then ping me to re-run the NotebookLM deep-research pass as a follow-up — it wasn't a blocker for the findings above, but it could add Google's web-grounded synthesis on top for the final paper draft.

---

## 5. Proposed Experiments (Ranked by Expected ROI / Effort)

Each experiment is grounded in a specific weakness from `arch4_openscholar_critique_2026.md` and a specific 2026 finding above. All should run on the existing **5-6 seed protocol** against the standardized `Indomain_curated.tar.gz` (585) / `Cross-data_curated.tar.gz` (1145) sets, consistent with `EXPERIMENTS_MASTER_LEADERBOARD.md`.

### Exp-1 (Low effort, high ROI): Vectorize `_build_spatial_cross_bias`
- **Hypothesis**: Pure engineering fix — replacing the Python/numpy double-loop (L603-631) with batched Paddle tensor ops — changes **zero** accuracy but measurably cuts latency, enabling AMP/jit.
- **Method**: Rewrite using broadcasting: compute `c_j` for all `(b, j)` as a tensor, `u_k` as a static buffer, then a single broadcasted `exp(-((u[None,None,:] - c[:,:,None])**2) / (2*sigma**2))`.
- **Metric**: Latency (ms/img) on T4, throughput (FPS); accuracy must be bit-identical (sanity regression test) on a fixed seed.
- **Risk**: Very low. Good first PR before touching any modeling logic.

### Exp-2: Confidence-Adaptive σ (merges ARCH-4 + ARCH-4C into one mechanism)
- **Hypothesis**: Coupling $\sigma_j$ to the existing 4D CTC confidence (margin, entropy) — e.g. $\sigma_j = \sigma_0 \cdot (1 + \beta \cdot \mathcal{H}_j)$ — recovers ARCH-4C's stability gain ($\pm0.37\%$) *and* ARCH-4's peak accuracy (91.35%) in a single variant, instead of needing two separate trained checkpoints.
- **Method**: Modify `_build_spatial_cross_bias` to accept the existing `confs` tensor already computed upstream; make $\beta$ a learned scalar (or small MLP) rather than hand-tuned.
- **Ablations**: (a) $\beta$ fixed vs. learned, (b) coupling to margin only vs. full entropy vs. full 4D vector.
- **Metric**: Cross-data accuracy/CER + std-dev across 5 seeds — target: match or beat ARCH-4C's $\pm0.37\%$ stability while reaching ARCH-4's 91.35% peak.

### Exp-3: CTC-Confidence-Gated Edit Eligibility (inspired by §2.2)
- **Hypothesis**: Restricting which seed positions are even eligible for `op != KEEP` based on a CTC-confidence threshold (not just the existing post-hoc `best_prob - seed_prob >= delta` rule) reduces false-edit rate further below the current 0.08% FER without hurting recall on true errors.
- **Method**: Add a pre-filter in `forward()` (near L957-993): positions where CTC top-1 margin exceeds a high-confidence threshold are forced to `op_ids=KEEP` regardless of decoder output, freeing the decoder's error budget for genuinely ambiguous positions.
- **Metric**: False Edit Rate (currently 0.08%), overall accuracy, and specifically error-correction recall on the known half-digit transition failure pattern (`pattern_half_digit_transition_dynamics.md`).

### Exp-4: 2-Pass Iterative Refinement (inspired by IPAD / CTC-Seeded Edit Flow §2.2)
- **Hypothesis**: Re-feeding the once-refined sequence through the same decoder a second time, committing only the highest-confidence edits on pass 1 and resolving remaining ambiguous positions on pass 2, fixes **middle-of-sequence multi-character errors** that today require the full ARCH-8 op head.
- **Method**: Add an "easy-first" commit rule: after pass 1, lock positions where `best_prob - seed_prob` exceeds a high threshold; re-run cross-attention + heads only on unlocked positions for pass 2.
- **Metric**: Accuracy/CER delta vs. single-pass ARCH-4, plus added latency (must stay under ~20ms/img T4 budget per `EXPERIMENTS_MASTER_LEADERBOARD.md`'s existing latency column).
- **Risk**: Medium — changes inference control flow; needs careful masking to avoid breaking the existing `branch_debug` introspection payload used by `infer_rec.py`.

### Exp-5: Glyph-Shape Canonical Mask Prior (inspired by CAM, §1)
- **Hypothesis**: Adding a second, additive bias term derived from canonical font-glyph masks for the top-2 CTC alternatives (not just spatial position) directly reduces the project's own documented 8↔9/3↔8 confusion pattern.
- **Method**: Precompute a small canonical glyph-mask embedding per vocab digit (10 classes is cheap — can even be a fixed, non-learned lookup to start), fuse with the existing token embedding via a gated sum (mirrors ARCH-7's gated fusion pattern already in the codebase).
- **Metric**: Confusion-matrix-specific metrics — accuracy restricted to samples whose ground truth or CTC alternative is in `{8,9,3}` (cross-reference `visual_error_audit.md` for the exact failure subset).

### Exp-6: Row-Aware (2D) Spatial Bias
- **Hypothesis**: Extending $u_k$ to a true 2D $(u_k, v_k)$ coordinate (the high-res visual memory already has 3 rows, currently collapsed to identical horizontal coordinates — see weakness #3) helps specifically on glare/occlusion bands that are row-localized.
- **Method**: Anisotropic Gaussian $\mathcal{B}_{j,k} = \lambda \exp(-\frac{(u_k-c_j)^2}{2\sigma_u^2} - \frac{(v_k - v_j^{\text{target}})^2}{2\sigma_v^2})$, with $v_j^{\text{target}}$ either fixed (center row) or predicted by a tiny auxiliary head.
- **Metric**: Accuracy stratified by the existing `results/cross-audit-visual` glare/occlusion subset if available, else full cross-data accuracy.
- **Risk**: Medium-high — requires deciding how to supervise/predict $v_j^{\text{target}}$ since CTC alone gives no vertical signal; likely the weakest-ROI experiment of the six unless a cheap heuristic (e.g. always center-row) already helps.

### Suggested Execution Order
1. **Exp-1** (free latency win, do regardless) →
2. **Exp-2** (highest expected accuracy/stability ROI, directly unifies two existing checkpoints) →
3. **Exp-3** (cheap, directly targets the paper's headline FER metric) →
4. **Exp-5** (targets a named, already-documented failure pattern) →
5. **Exp-4** and **Exp-6** as stretch goals / ARCH-9+ARCH-10 candidates if time budget allows before the Q1-target submission.
