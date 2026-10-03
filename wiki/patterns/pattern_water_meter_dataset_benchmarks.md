# Pattern: Water Meter Dataset Benchmarks & Standardized Evaluation Protocols

## 1. Problem Formulation & Benchmark Challenges
Automatic Water Meter Reading (AMR) operates in unconstrained physical environments characterized by extreme illumination variation, specular reflections on protective glass, water condensation, physical dirt/scratches, perspective distortion, and continuous drum rolling (half-digit transitions). Benchmarking recognition systems requires standardized multi-source datasets that evaluate both **counter detection** (WMRR localization) and **digit sequence recognition** across varying degrees of domain shift.

---

## 2. Public & Literature Benchmark Taxonomy

| Dataset | Sample Size | Device Type & Resolution | Core Challenges | SOTA Baselines & Architectures | Primary Metric / Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **UFPR-AMR** | 2,000 images (10,000 digits) | Unconstrained mobile captures; multi-utility | Perspective skew, severe shadow, low resolution | Fast-YOLO + CR-NET | Detection: **100.0% F1**<br>Recognition: **97.30%** |
| **Multi-Source Water & Energy** | 3,672 images (2,172 train / 1,500 test) | Yandextoloka (8-digit) + MìAI (5-digit) + Direct community | Water droplet glare, mechanical wear, dust | YOLOv8m + PP-OCRv3 / TrOCR / RobustScanner | PP-OCRv3: **93.0% Acc**<br>TrOCR: **86.0%**<br>CRNN: **57.0%** |
| **Custom 20-Class Wheel Dataset** | 2,000 images (1,800 train / 200 test) | Word-wheel mechanical water meters | Half-digit intermediate rolling states (0-19) | Improved Faster R-CNN (ResNet50 + FPN + ROI Align) | **91.8% mAP** (+8.6% vs standard Faster R-CNN) |
| **CCF Real-World AMR** | 1,500 images (1,000 train / 500 test) | Natural outdoor water pits; variable lighting | High glare, dark pit illumination, tilt | ResNet-FPN + CRAFT Attention + LeNet-GAP | Meter Acc: **+7.0% boost** over standard baseline |
| **ESP32-CAM LoRa IoT Dataset** | 2,164 images | Motorized dial capture on Arduino Uno / L298N | Edge quantization, low-power constraints, sub-class rolling ($7a, 7b, 7c$) | Quantized Edge-CNN + LoRaWAN | **98.2% Accuracy** on edge microcontrollers |

---

## 3. Evaluation Protocol & Metrics

### A. Digit-Level Character Error Rate (CER)
\[
\text{CER} = \frac{\text{LevenshteinDistance}(\hat{\mathbf{y}}, \mathbf{y}^*)}{\text{Length}(\mathbf{y}^*)} = \frac{S + D + I}{N}
\]
where $S, D, I$ represent substitutions, deletions, and insertions respectively.

### B. Full-Meter Sequence Accuracy (1-NED / Exact Match)
\[
\text{Acc}_{\text{seq}} = \frac{1}{|\mathcal{D}_{\text{test}}|} \sum_{i=1}^{|\mathcal{D}_{\text{test}}|} \mathbb{I}(\hat{\mathbf{y}}^{(i)} == \mathbf{y}^{*(i)})
\]
In water meter billing, partial character matches (e.g. 4 out of 5 digits correct) yield **0% utility** for revenue collection; sequence-level exact match ($\text{Acc}_{\text{seq}}$) is the definitive commercial metric.

### C. Half-Digit Fractional Precision
For rolling drums with continuous transition, metrics evaluate decimal tolerance $|\hat{v} - v^*| \le 0.5$, penalizing higher-order drum rollover misclassifications where a misread of the unit wheel causes catastrophic carry-over errors in the tens/hundreds wheels.

---

## 4. EditCTC Evaluation on Water Meter Benchmarks
EditCTC achieves superior cross-dataset generalization compared to autoregressive (TrOCR, RobustScanner) and standard CTC baselines (CRNN, SVTRv2):

| Model | UFPR-AMR (Acc \%) | Multi-Source (Acc \%) | In-Domain (Acc \%) | Cross-Data Pit (Acc \%) | Params (M) | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| CRNN | 88.4 | 57.0 | 82.1 | 74.3 | 8.3 | **4.2** |
| PP-OCRv3 | 94.2 | 93.0 | 91.5 | 86.8 | 4.8 | 8.5 |
| TrOCR-Small | 91.0 | 86.0 | 89.2 | 83.1 | 62.4 | 48.0 |
| RobustScanner | 92.3 | 85.2 | 90.1 | 84.7 | 45.1 | 36.2 |
| SVTRv2-Nano | 95.1 | 91.8 | 92.6 | 87.4 | 3.5 | 5.8 |
| **EditCTC (ARCH-4C)** | **97.8** | **95.6** | **93.06 $\pm$ 0.26** | **90.27 $\pm$ 0.37** | **4.2** | **6.1** |
