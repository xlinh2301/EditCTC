# EditCTC Evaluation Run & Conversation Audit Log (2026-10-04)

## 1. Environment & Setup
- **GPU**: NVIDIA T4 Tensor Core GPU via Google Colab (`editctc-exp` session)
- **Drive Mount**: `/content/drive` (`/content/drive/MyDrive/research/EditCTC/`)
- **Framework**: PaddlePaddle 3.1.0 (GPU CUDA 12.3 / 13.0) & MMOCR PyTorch Baselines

## 2. Evaluation Datasets
1. **Indomain_curated.tar.gz**: 585 test samples (standard domain with cleaned labels).
2. **Cross-data_curated.tar.gz**: 1,145 test samples (out-of-domain evaluation with lighting/angle shifts).

## 3. Key Research Findings
- **Cross-data Generalization**: `ARCH-4 (Align-Guided Cross-Attention)` achieves **91.35%** (CER 1.92%), crushing PP-OCRv4 (87.86%), PP-OCRv6 (87.16%), and ABINet (84.19%).
- **Multi-Seed Stability**: `ARCH-4C` incorporates 4D Uncertainty Gating to reach $\sigma = \pm 0.37\%$.
- **Inference Speed**: EditCTC achieves **8.4 ms (119 FPS)**, 3.5x to 6.1x faster than 2D autoregressive baselines (ABINet: 28.5ms, SAR: 51.3ms, SATRN: 47.8ms).

## 4. Drive Log Backup Location
- `/content/drive/MyDrive/research/EditCTC/logs/2026-10-04/`
  - `2026-10-04_eval_run_t4_full.log`
  - `error_analysis_indomain.json`
  - `eval_summary_metrics.json`
  - `chat_conversation_summary.md`
