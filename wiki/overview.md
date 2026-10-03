# Project Overview: EditCTC

## 🎯 Industrial Problem Statement & Motivation
In industrial automated meter reading (AMR) and smart utility monitoring, optical character recognition (OCR) of mechanical water meters, gas counters, and electric dials faces severe challenges:
1. **Out-of-Domain Degradation**: Models trained on clean indoor or synthetic data collapse when deployed on dusty, humid, or degraded outdoor utility meters.
2. **CTC Alignment Drift & Attention Scattering**: Standard Connectionist Temporal Classification (CTC) decoders emit peak activations without spatial bounds, causing downstream autoregressive/NAR decoders to attend to incorrect adjacent characters.
3. **Over-Correction Pathology**: Naive error-correction modules attempt to "correct" already accurate seed tokens, replacing correct digits with spurious predictions.
4. **Fine-Stroke Character Confusion**: Dial digits with subtle topological differences (e.g. `8` vs `9`, `3` vs `8`, `0` vs `6`) fail under specular highlights and water droplet distortions.

**EditCTC** resolves these core issues via **Alignment-Guided Non-Autoregressive Sequence Refinement** (ARCH-4 / ARCH-4C), delivering **91.35% Cross-Data Accuracy** (CER 1.92%) and **93.30% In-Domain Accuracy** across 52 multi-seed benchmark evaluations.

---

## 🛠 Technical Stack & Dependencies
- **Deep Learning Framework**: PaddlePaddle 3.0.0 / PyTorch compatible
- **Backbone Architecture**: PPLCNetV4 (Lightweight visual feature extractor with Stage 1–5 hierarchy)
- **Neck Module**: lightSVTR (Depth=2, Dims=120) + SharedHighResVisualMemory (Conv2D + LayerNorm, 1/4 scale)
- **Primary CTC Branch**: Linear CTCHead (vocab_size=97) emitting 40 timesteps
- **Refinement Decoder**: 4-Layer Non-Autoregressive NRTR Transformer Decoder with Alignment-Guided Cross-Attention
- **Evaluation Benchmark**: In-domain (585 meter displays) & Cross-data (1,145 challenging displays)

---

## 📂 Repository & Knowledge Architecture
```text
EditCTC/
├── wiki/                          # 3-Layer Persistent Knowledge Base (WikiSkill, arXiv:2608.27454)
│   ├── patterns/                  # Mathematical formulations, spatial bias, uncertainty priors
│   ├── raw/                       # Immutable multi-seed benchmark JSONs, confusion matrices
│   ├── logs.md                    # Chronological evolution logs of ARCH-1 to ARCH-8
│   └── skill-impact.md            # Validation gating tracker across 52 seed evaluations
├── .agents/specs/                 # ARIA Sub-Specs (arXiv:2510.11143)
│   └── 0001-editctc-alignment-guided-cross-attention.md
├── config/                        # Multi-seed experiment YAML configurations
├── openspec/                      # Specification evolution changes
└── paper/                         # Publication-ready LaTeX manuscript (PaperOrchestra, arXiv:2604.05018)
    ├── main.tex
    ├── references.bib
    └── sections/
```
