# Benchmark OCR đồng hồ nước — kết quả tái lập và baseline tham chiếu

**Trạng thái: Đã đánh giá 4 họ MMOCR trên đủ 5 seed (20 checkpoint) và cả 5 sheet; 92.300 dự đoán đã được kiểm tra lại độc lập. PP-OCRv4 Server, PP-OCRv5 Server, PP-OCRv6 Small và SVTRv2 đã được đánh giá đủ 5 seed trên 5 sheet; tổng cộng 92.300 dự đoán cho 20 checkpoint PaddleOCR đã được kiểm tra độc lập.** Cũng đã tái lập 10 checkpoint EditCTC và checkpoint PP-OCRv6 cục bộ. Năm dòng PP-OCRv3/v4 mỗi sheet vẫn là số liệu lịch sử chưa tái lập. Không dùng các kết quả này để tuyên bố SOTA hoặc so sánh công bằng với bài báo khác trước khi xác minh nguồn/giấy phép và tính độc lập của các tập public. Kế hoạch backbone/neck và alignment feature mới được ghi tại [`alignment_backbone_neck_research_20261011.md`](alignment_backbone_neck_research_20261011.md).

## Nguồn số liệu và cách đọc

- **EditCTC:** kết quả mới và dự đoán từng ảnh ở [`../../../results/reproduced_editctc_five_split_20261008_retry/`](../../../results/reproduced_editctc_five_split_20261008_retry/); manifest ghi phiên bản Paddle, device, YAML, checkpoint SHA-256, sample counts và metric. Mỗi result JSONL có đúng một dự đoán trên mỗi ảnh.
- **MMOCR năm seed:** manifest, 100 tệp prediction JSONL, checkpoint-selection record, 20 config trích từ checkpoint và summary được kiểm tra độc lập ở [`../../../results/reproduced_mmocr_five_seed_benchmark_20261009/`](../../../results/reproduced_mmocr_five_seed_benchmark_20261009/). Mỗi tệp có một dự đoán trên mỗi ảnh; `five_seed_summary.json` ghi SHA-256 của mọi prediction file.
- **PaddleOCR năm seed:** PP-OCRv4/v5/v6 có manifest, 75 tệp prediction JSONL và summary được kiểm tra độc lập ở [`../../../results/reproduced_paddle_ocr_v4_v6_five_split_20261010_retry/`](../../../results/reproduced_paddle_ocr_v4_v6_five_split_20261010_retry/). SVTRv2 có manifest, 25 tệp prediction JSONL và summary được kiểm tra độc lập ở [`../../../results/reproduced_svtrv2_five_seed_five_split_20261011/`](../../../results/reproduced_svtrv2_five_seed_five_split_20261011/). Tất cả dùng `best_accuracy.pdparams` được chọn trên validation cho từng seed; evaluator strict-load state dict. Script tái lập và verifier độc lập lần lượt là `scripts/evaluate_paddle_ocr_five_split.py` và `scripts/verify_paddle_ocr_five_split.py`.
- **PP-OCRv3/v4 official:** các hàng được đánh dấu `historical*` chép từ [`../../../results/extended_baselines_benchmark_results.json`](../../../results/extended_baselines_benchmark_results.json); chưa tái chạy hoặc đối chiếu checksum weights. Không coi đây là so sánh baseline đã kiểm chứng. Năm model MMOCR trong JSON lịch sử không có checkpoint cục bộ tương ứng ở lần chạy đó nên không được coi là kết quả checkpoint mới; các MMOCR checkpoint hiện có được kiểm kê riêng phía dưới.
- Mỗi hàng: **số ảnh đúng/tổng, exact-match %, CER %**. CER là tổng Levenshtein edit distance / tổng ký tự nhãn sau khi lọc ngoài `[0-9A-Za-z]`. Giữ nguyên số 0 đầu. Tách riêng từng dataset; không gộp In-domain, Cross-data và public.

## Chỉ mục config để tái lập benchmark

Config không được suy ra từ tên model. Các manifest kết quả giữ path và SHA-256 của config đã dùng; các checkpoint per-seed cũng lưu immutable `config.yml` cùng run. Các nguồn chính:

| Họ kết quả | Config/manifest |
| --- | --- |
| EditCTC 10 checkpoint | [`results/reproduced_editctc_five_split_20261008_retry/manifest.json`](../../../results/reproduced_editctc_five_split_20261008_retry/manifest.json) và `config/PP-OCRv6_small_rec_s1024_e44_highres_arch*.yml` |
| PP-OCRv4/v5/v6 | [`results/reproduced_paddle_ocr_v4_v6_five_split_20261010_retry/manifest.json`](../../../results/reproduced_paddle_ocr_v4_v6_five_split_20261010_retry/manifest.json), cùng `checkpoints/retrained_paddle_official_20261009/<model>/seed_<seed>/config.yml` |
| SVTRv2 | [`results/reproduced_svtrv2_five_seed_five_split_20261011/manifest.json`](../../../results/reproduced_svtrv2_five_seed_five_split_20261011/manifest.json), cùng `checkpoints/retrained_paddle_official_20261009/SVTRv2/seed_<seed>/config.yml` |
| MMOCR | [`results/reproduced_mmocr_five_seed_benchmark_20261009/manifest.json`](../../../results/reproduced_mmocr_five_seed_benchmark_20261009/manifest.json), config Python và hash embedded trong manifest |

Run ARCH-7 khởi tạo từ official PP-OCRv6 được giữ riêng tại `results/reproduced_arch7_official_init_five_seed_five_split_20261011/`; do correction branch không kích hoạt và mọi seed cho cùng CTC output, run này là **diagnostic artifact**, không phải dòng benchmark chính.

## Tập dữ liệu hiện có

| Sheet | Phân loại | Cặp ảnh/nhãn tìm thấy | Lưu ý |
| --- | --- | ---: | --- |
| Indomain Test | In-domain | 585/585 | Tập test, không dùng để điều chỉnh ngưỡng. |
| Cross-data curated | Cross-data nội bộ | 1.145/1.147 | 2 mục nhãn thiếu ảnh: kiểm tra nguồn gốc trước khi so sánh. |
| DL7981 | Public (chưa xác minh nguồn/license) | 961/961 | Tập crop trên đĩa; chưa chứng minh split chính thức. |
| Liza Set 1 | Public (chưa xác minh nguồn/license) | 924/924 | Chưa chứng minh split chính thức. |
| Liza Set 2 | Public (chưa xác minh nguồn/license) | 1.000/1.000 | Chưa chứng minh split chính thức. |

## Bảng tổng hợp — MMOCR, PP-OCR và SVTRv2

Các ô ghi **exact-match accuracy / CER (%)**, đều là mean ± sample SD trên 5 seed (`1024, 2024, 3024, 4024, 5024`). Số mẫu là số ảnh đánh giá trên mỗi seed; Cross-data có 1.145 ảnh do hai ảnh thiếu. Chọn checkpoint bằng curated validation; test chỉ dùng để đánh giá. Các model dùng pipeline inference chính thức tương ứng, nên bảng phản ánh kết quả của từng implementation dưới cùng nhãn/split và chuẩn hóa metric, không khẳng định preprocessing nội bộ hoàn toàn đồng nhất.

| Model | In-domain (585) | Cross-data (1.145) | DL7981 (961) | Liza Set 1 (924) | Liza Set 2 (1.000) |
| --- | ---: | ---: | ---: | ---: | ---: |
| MMOCR ABINet | 91.32 ± 0.22 / 2.65 ± 0.07 | 79.70 ± 1.73 / 4.25 ± 0.35 | 62.73 ± 1.57 / 8.75 ± 0.49 | 31.34 ± 4.28 / 21.16 ± 1.40 | 24.96 ± 2.91 / 27.85 ± 1.02 |
| MMOCR MASTER | 89.09 ± 0.74 / 3.27 ± 0.18 | 76.33 ± 1.90 / 5.03 ± 0.42 | 58.98 ± 3.53 / 9.79 ± 1.28 | 28.29 ± 1.22 / 26.11 ± 0.97 | 20.72 ± 0.69 / 34.55 ± 0.73 |
| MMOCR SAR | 89.81 ± 0.54 / 3.08 ± 0.16 | 74.99 ± 2.17 / 5.56 ± 0.64 | 53.47 ± 1.64 / 10.77 ± 0.53 | 9.52 ± 1.42 / 37.56 ± 2.40 | 10.34 ± 0.76 / 43.19 ± 1.90 |
| MMOCR SATRN | 92.00 ± 0.31 / 2.53 ± 0.08 | 80.00 ± 1.04 / 4.15 ± 0.25 | 64.33 ± 1.20 / 8.30 ± 0.23 | 37.51 ± 1.70 / 20.97 ± 1.10 | 32.06 ± 1.22 / 27.97 ± 0.82 |
| PP-OCRv4 Server | 93.10 ± 0.81 / 2.17 ± 0.20 | 82.81 ± 2.32 / 3.55 ± 0.45 | 68.70 ± 1.41 / 7.35 ± 0.43 | 38.98 ± 2.73 / 21.81 ± 2.22 | 36.00 ± 3.19 / 26.27 ± 2.06 |
| PP-OCRv5 Server | 92.62 ± 0.39 / 2.26 ± 0.06 | 82.83 ± 1.09 / 3.58 ± 0.21 | 66.08 ± 2.06 / 7.86 ± 0.40 | 31.62 ± 2.46 / 27.40 ± 1.09 | 29.66 ± 1.41 / 31.94 ± 0.63 |
| PP-OCRv6 Small | 92.72 ± 0.59 / 2.28 ± 0.20 | 83.74 ± 1.22 / 3.40 ± 0.27 | 70.30 ± 1.18 / 7.02 ± 0.24 | 41.90 ± 3.41 / 19.28 ± 1.68 | 39.28 ± 3.04 / 25.48 ± 1.54 |
| SVTRv2 | 93.09 ± 0.47 / 2.18 ± 0.11 | 83.89 ± 2.62 / 3.21 ± 0.52 | 68.99 ± 0.71 / 7.20 ± 0.15 | 48.07 ± 3.85 / 15.75 ± 1.27 | 44.62 ± 3.17 / 21.53 ± 1.24 |

Các kết quả trên là retraining năm seed, không phải zero-shot. Dữ liệu public DL7981/Liza chưa xác minh nguồn, license hoặc split độc lập; xem các phần dưới để biết chi tiết checkpoint, prediction và verifier theo họ model.

## PP-OCRv4/v5/v6 — retraining năm seed

Các số liệu dưới đây là **mean ± sample SD giữa 5 seed** (`1024, 2024, 3024, 4024, 5024`), lấy checkpoint `best_accuracy` theo curated validation (409 ảnh); test chỉ dùng để đánh giá. Accuracy là exact-match sau khi lọc ký tự ngoài `[0-9A-Za-z]` và giữ nguyên số 0 đầu. CER là tổng Levenshtein distance chia tổng số ký tự nhãn, cùng phép chuẩn hóa. Mọi dòng được tính lại từ prediction JSONL bởi verifier riêng, gồm **69.225 dự đoán** trên 15 checkpoint. Hai ảnh thiếu ở Cross-data không được tính, nên mỗi seed có 1.145 mẫu.

| Tập (mẫu/seed) | PP-OCRv4 Server accuracy / CER (%) | PP-OCRv5 Server accuracy / CER (%) | PP-OCRv6 Small accuracy / CER (%) |
| --- | ---: | ---: | ---: |
| Indomain Test (585) | 93.10 ± 0.81 / 2.17 ± 0.20 | 92.62 ± 0.39 / 2.26 ± 0.06 | 92.72 ± 0.59 / 2.28 ± 0.20 |
| Cross-data (1.145) | 82.81 ± 2.32 / 3.55 ± 0.45 | 82.83 ± 1.09 / 3.58 ± 0.21 | 83.74 ± 1.22 / 3.40 ± 0.27 |
| DL7981 (961) | 68.70 ± 1.41 / 7.35 ± 0.43 | 66.08 ± 2.06 / 7.86 ± 0.40 | 70.30 ± 1.18 / 7.02 ± 0.24 |
| Liza Set 1 (924) | 38.98 ± 2.73 / 21.81 ± 2.22 | 31.62 ± 2.46 / 27.40 ± 1.09 | 41.90 ± 3.41 / 19.28 ± 1.68 |
| Liza Set 2 (1.000) | 36.00 ± 3.19 / 26.27 ± 2.06 | 29.66 ± 1.41 / 31.94 ± 0.63 | 39.28 ± 3.04 / 25.48 ± 1.54 |

These are retrained model-family comparisons under this experiment's common data protocol, not official zero-shot scores. Public-set provenance/license and split independence remain unverified; interpret those rows accordingly.

## SVTRv2 — retraining năm seed

SVTRv2 dùng official pretrained weights và cùng curated train/validation protocol; batch size 16 và learning rate `8.3333e-5` được scale tuyến tính từ cấu hình official 192 mẫu trên 8 GPU sang một GPU. Chọn `best_accuracy.pdparams` theo curated validation (409 ảnh); test chỉ dùng để đánh giá. Năm checkpoint được strict-load, inference bằng PaddleOCR pipeline, và verifier độc lập tính lại metric từ **23.075 dự đoán**. Bảng là mean ± sample SD giữa các seed `1024, 2024, 3024, 4024, 5024`; Cross-data loại 2 nhãn không có ảnh, còn 1.145 mẫu/seed. Kết quả và prediction từng ảnh: [`../../../results/reproduced_svtrv2_five_seed_five_split_20261011/`](../../../results/reproduced_svtrv2_five_seed_five_split_20261011/).

| Tập (mẫu/seed) | SVTRv2 accuracy / CER (%) |
| --- | ---: |
| Indomain Test (585) | 93.09 ± 0.47 / 2.18 ± 0.11 |
| Cross-data (1.145) | 83.89 ± 2.62 / 3.21 ± 0.52 |
| DL7981 (961) | 68.99 ± 0.71 / 7.20 ± 0.15 |
| Liza Set 1 (924) | 48.07 ± 3.85 / 15.75 ± 1.27 |
| Liza Set 2 (1.000) | 44.62 ± 3.17 / 21.53 ± 1.24 |

Đây là retraining từ official initialization theo protocol đã nêu, không phải zero-shot score. Nguồn/license và tính độc lập của public split vẫn chưa được xác minh.

### Điểm cao nhất trong 10 checkpoint EditCTC đã đo

| Sheet | Model tốt nhất theo exact-match | Exact-match | CER thấp nhất | Model CER thấp nhất |
| --- | --- | ---: | ---: | --- |
| In-domain | EXP18B | 93.85% | 2.25% | EXP18B |
| Cross-data | ARCH7 | 91.53% | 1.91% | ARCH7 |
| DL7981 | ARCH5 | 74.19% | 5.73% | ARCH5 |
| Liza Set 1 | ARCH7 | 41.13% | 20.09% | ARCH7 |
| Liza Set 2 | ARCH7 | 36.60% | 26.06% | ARCH7 |

## Sheet 1 — Indomain Test (585 ảnh)

| Model | Đúng/tổng | Exact-match | CER | Nguồn |
| --- | ---: | ---: | ---: | --- |
| EditCTC ARCH-1 (Explicit Head) | 545/585 | 93.16% | 2.36% | reproduced |
| EditCTC ARCH-2A (CTC Conf2D) | 547/585 | 93.50% | 2.28% | reproduced |
| EditCTC ARCH-2B (Full Conf4D) | 546/585 | 93.33% | 2.28% | reproduced |
| EditCTC ARCH-3 (Temporal Align) | 546/585 | 93.33% | 2.32% | reproduced |
| EditCTC ARCH-4 (Align-Guided Cross-Attn) | 546/585 | 93.33% | 2.32% | reproduced |
| EditCTC ARCH-4B (Align+Append) | 542/585 | 92.65% | 2.39% | reproduced |
| EditCTC ARCH-4C (Align+Conf4D) | 545/585 | 93.16% | 2.36% | reproduced |
| EditCTC ARCH-5 (Local Visual Refine) | 543/585 | 92.82% | 2.39% | reproduced |
| EditCTC ARCH-7 (Gated Fusion) | 546/585 | 93.33% | 2.28% | reproduced |
| PP-OCRv6 Canonical Baseline (EXP-18B) | 549/585 | 93.85% | 2.25% | reproduced |
| PP-OCRv4 Server (Official ch/multi) | 354/585 | 60.51% | 16.16% | historical* |
| PP-OCRv4 Mobile (Official ch/multi) | 294/585 | 50.26% | 22.50% | historical* |
| PP-OCRv4 English/Digits (Official) | 279/585 | 47.69% | 21.98% | historical* |
| PP-OCRv3 Mobile (Official ch/multi) | 296/585 | 50.60% | 20.80% | historical* |
| PP-OCRv3 English/Digits (Official) | 278/585 | 47.52% | 22.20% | historical* |

## Sheet 2 — Cross-data curated (1.145 ảnh)

| Model | Đúng/tổng | Exact-match | CER | Nguồn |
| --- | ---: | ---: | ---: | --- |
| EditCTC ARCH-1 (Explicit Head) | 1035/1145 | 90.39% | 2.13% | reproduced |
| EditCTC ARCH-2A (CTC Conf2D) | 1010/1145 | 88.21% | 2.74% | reproduced |
| EditCTC ARCH-2B (Full Conf4D) | 1042/1145 | 91.00% | 2.06% | reproduced |
| EditCTC ARCH-3 (Temporal Align) | 1039/1145 | 90.74% | 2.09% | reproduced |
| EditCTC ARCH-4 (Align-Guided Cross-Attn) | 1046/1145 | 91.35% | 1.92% | reproduced |
| EditCTC ARCH-4B (Align+Append) | 979/1145 | 85.50% | 3.15% | reproduced |
| EditCTC ARCH-4C (Align+Conf4D) | 1039/1145 | 90.74% | 2.11% | reproduced |
| EditCTC ARCH-5 (Local Visual Refine) | 1040/1145 | 90.83% | 2.04% | reproduced |
| EditCTC ARCH-7 (Gated Fusion) | 1048/1145 | 91.53% | 1.91% | reproduced |
| PP-OCRv6 Canonical Baseline (EXP-18B) | 995/1145 | 86.90% | 2.98% | reproduced |
| PP-OCRv4 Server (Official ch/multi) | 531/1145 | 46.38% | 19.19% | historical* |
| PP-OCRv4 Mobile (Official ch/multi) | 409/1145 | 35.72% | 30.13% | historical* |
| PP-OCRv4 English/Digits (Official) | 449/1145 | 39.21% | 25.40% | historical* |
| PP-OCRv3 Mobile (Official ch/multi) | 404/1145 | 35.28% | 28.82% | historical* |
| PP-OCRv3 English/Digits (Official) | 507/1145 | 44.28% | 21.25% | historical* |

## Sheet 3 — DL7981 (961 ảnh, public chưa xác minh)

| Model | Đúng/tổng | Exact-match | CER | Nguồn |
| --- | ---: | ---: | ---: | --- |
| EditCTC ARCH-1 (Explicit Head) | 683/961 | 71.07% | 6.66% | reproduced |
| EditCTC ARCH-2A (CTC Conf2D) | 657/961 | 68.37% | 7.32% | reproduced |
| EditCTC ARCH-2B (Full Conf4D) | 699/961 | 72.74% | 6.27% | reproduced |
| EditCTC ARCH-3 (Temporal Align) | 682/961 | 70.97% | 6.74% | reproduced |
| EditCTC ARCH-4 (Align-Guided Cross-Attn) | 695/961 | 72.32% | 6.45% | reproduced |
| EditCTC ARCH-4B (Align+Append) | 707/961 | 73.57% | 6.06% | reproduced |
| EditCTC ARCH-4C (Align+Conf4D) | 712/961 | 74.09% | 6.02% | reproduced |
| EditCTC ARCH-5 (Local Visual Refine) | 713/961 | 74.19% | 5.73% | reproduced |
| EditCTC ARCH-7 (Gated Fusion) | 694/961 | 72.22% | 6.39% | reproduced |
| PP-OCRv6 Canonical Baseline (EXP-18B) | 647/961 | 67.33% | 7.43% | reproduced |
| PP-OCRv4 Server (Official ch/multi) | 569/961 | 59.21% | 9.65% | historical* |
| PP-OCRv4 Mobile (Official ch/multi) | 509/961 | 52.97% | 11.76% | historical* |
| PP-OCRv4 English/Digits (Official) | 505/961 | 52.55% | 11.56% | historical* |
| PP-OCRv3 Mobile (Official ch/multi) | 475/961 | 49.43% | 13.03% | historical* |
| PP-OCRv3 English/Digits (Official) | 555/961 | 57.75% | 10.37% | historical* |

## Sheet 4 — Liza Set 1 (924 ảnh, public chưa xác minh)

| Model | Đúng/tổng | Exact-match | CER | Nguồn |
| --- | ---: | ---: | ---: | --- |
| EditCTC ARCH-1 (Explicit Head) | 343/924 | 37.12% | 21.94% | reproduced |
| EditCTC ARCH-2A (CTC Conf2D) | 269/924 | 29.11% | 25.08% | reproduced |
| EditCTC ARCH-2B (Full Conf4D) | 333/924 | 36.04% | 21.74% | reproduced |
| EditCTC ARCH-3 (Temporal Align) | 334/924 | 36.15% | 21.92% | reproduced |
| EditCTC ARCH-4 (Align-Guided Cross-Attn) | 339/924 | 36.69% | 22.03% | reproduced |
| EditCTC ARCH-4B (Align+Append) | 338/924 | 36.58% | 21.90% | reproduced |
| EditCTC ARCH-4C (Align+Conf4D) | 331/924 | 35.82% | 21.90% | reproduced |
| EditCTC ARCH-5 (Local Visual Refine) | 345/924 | 37.34% | 21.39% | reproduced |
| EditCTC ARCH-7 (Gated Fusion) | 380/924 | 41.13% | 20.09% | reproduced |
| PP-OCRv6 Canonical Baseline (EXP-18B) | 229/924 | 24.78% | 25.66% | reproduced |
| PP-OCRv4 Server (Official ch/multi) | 93/924 | 10.06% | 50.81% | historical* |
| PP-OCRv4 Mobile (Official ch/multi) | 53/924 | 5.74% | 58.74% | historical* |
| PP-OCRv4 English/Digits (Official) | 83/924 | 8.98% | 53.19% | historical* |
| PP-OCRv3 Mobile (Official ch/multi) | 33/924 | 3.57% | 58.21% | historical* |
| PP-OCRv3 English/Digits (Official) | 24/924 | 2.60% | 66.13% | historical* |

## Sheet 5 — Liza Set 2 (1.000 ảnh, public chưa xác minh)

| Model | Đúng/tổng | Exact-match | CER | Nguồn |
| --- | ---: | ---: | ---: | --- |
| EditCTC ARCH-1 (Explicit Head) | 295/1000 | 29.50% | 28.87% | reproduced |
| EditCTC ARCH-2A (CTC Conf2D) | 243/1000 | 24.30% | 31.75% | reproduced |
| EditCTC ARCH-2B (Full Conf4D) | 327/1000 | 32.70% | 27.63% | reproduced |
| EditCTC ARCH-3 (Temporal Align) | 299/1000 | 29.90% | 28.73% | reproduced |
| EditCTC ARCH-4 (Align-Guided Cross-Attn) | 321/1000 | 32.10% | 27.85% | reproduced |
| EditCTC ARCH-4B (Align+Append) | 318/1000 | 31.80% | 27.64% | reproduced |
| EditCTC ARCH-4C (Align+Conf4D) | 327/1000 | 32.70% | 27.42% | reproduced |
| EditCTC ARCH-5 (Local Visual Refine) | 337/1000 | 33.70% | 26.97% | reproduced |
| EditCTC ARCH-7 (Gated Fusion) | 366/1000 | 36.60% | 26.06% | reproduced |
| PP-OCRv6 Canonical Baseline (EXP-18B) | 208/1000 | 20.80% | 33.25% | reproduced |
| PP-OCRv4 Server (Official ch/multi) | 181/1000 | 18.10% | 47.74% | historical* |
| PP-OCRv4 Mobile (Official ch/multi) | 129/1000 | 12.90% | 53.25% | historical* |
| PP-OCRv4 English/Digits (Official) | 128/1000 | 12.80% | 56.64% | historical* |
| PP-OCRv3 Mobile (Official ch/multi) | 90/1000 | 9.00% | 56.59% | historical* |
| PP-OCRv3 English/Digits (Official) | 58/1000 | 5.80% | 66.51% | historical* |

## MMOCR — retraining năm seed, đo mới bằng implementation chính thức

Đã đánh giá ABINet, MASTER, SAR và SATRN với seed `1024, 2024, 3024, 4024, 5024`. Mỗi seed dùng checkpoint ở epoch có word accuracy validation cao nhất trên tập curated validation (409 ảnh); nếu nhiều epoch đồng hạng, chọn epoch sớm nhất. Test không được dùng để chọn checkpoint. Tất cả 20 checkpoint được strict-load bằng MMOCR `TextRecInferencer` và chạy trên cùng 5 split.

Bảng báo cáo **mean ± sample SD giữa 5 seed**; mỗi seed được chấm độc lập trên cùng số ảnh. Exact-match và CER dùng phép chuẩn hóa ngoài `[0-9A-Za-z]`, giữ nguyên số 0 đầu. Đây là kết quả trực tiếp đo trên các checkpoint retrain, không phải số liệu bài báo.

### In-domain Test — 585 ảnh mỗi seed

| Model | Exact-match (%) | CER (%) |
| --- | ---: | ---: |
| ABINet | 91.32 ± 0.22 | 2.65 ± 0.07 |
| MASTER | 89.09 ± 0.74 | 3.27 ± 0.18 |
| SAR | 89.81 ± 0.54 | 3.08 ± 0.16 |
| SATRN | 92.00 ± 0.31 | 2.53 ± 0.08 |

### Cross-data — 1.145 ảnh mỗi seed

| Model | Exact-match (%) | CER (%) |
| --- | ---: | ---: |
| ABINet | 79.70 ± 1.73 | 4.25 ± 0.35 |
| MASTER | 76.33 ± 1.90 | 5.03 ± 0.42 |
| SAR | 74.99 ± 2.17 | 5.56 ± 0.64 |
| SATRN | 80.00 ± 1.04 | 4.15 ± 0.25 |

### DL7981 — 961 ảnh mỗi seed (public, nguồn/license chưa xác minh)

| Model | Exact-match (%) | CER (%) |
| --- | ---: | ---: |
| ABINet | 62.73 ± 1.57 | 8.75 ± 0.49 |
| MASTER | 58.98 ± 3.53 | 9.79 ± 1.28 |
| SAR | 53.47 ± 1.64 | 10.77 ± 0.53 |
| SATRN | 64.33 ± 1.20 | 8.30 ± 0.23 |

### Liza Set 1 — 924 ảnh mỗi seed (public, nguồn/license chưa xác minh)

| Model | Exact-match (%) | CER (%) |
| --- | ---: | ---: |
| ABINet | 31.34 ± 4.28 | 21.16 ± 1.40 |
| MASTER | 28.29 ± 1.22 | 26.11 ± 0.97 |
| SAR | 9.52 ± 1.42 | 37.56 ± 2.40 |
| SATRN | 37.51 ± 1.70 | 20.97 ± 1.10 |

### Liza Set 2 — 1.000 ảnh mỗi seed (public, nguồn/license chưa xác minh)

| Model | Exact-match (%) | CER (%) |
| --- | ---: | ---: |
| ABINet | 24.96 ± 2.91 | 27.85 ± 1.02 |
| MASTER | 20.72 ± 0.69 | 34.55 ± 0.73 |
| SAR | 10.34 ± 0.76 | 43.19 ± 1.90 |
| SATRN | 32.06 ± 1.22 | 27.97 ± 0.82 |

## MMOCR — kết quả single-checkpoint trước đây (giữ để đối chiếu)

Các bảng dưới đây là kết quả của lần đánh giá single-checkpoint trước đó, được giữ nguyên để đối chiếu; không thay thế hoặc gộp với kết quả retraining năm seed ở phần trên hay các baseline `historical*`. Mỗi checkpoint dùng config nhúng trong chính file `.pth`, test pipeline/dictionary theo config đó, và được nạp strict.

### In-domain Test — 585 ảnh

| Model | Đúng/tổng | Exact-match | CER |
| --- | ---: | ---: | ---: |
| ABINet | 535/585 | 91.45% | 2.65% |
| MASTER | 524/585 | 89.57% | 2.91% |
| SAR | 531/585 | 90.77% | 2.91% |
| SATRN | 540/585 | 92.31% | 2.43% |

### Cross-data — 1.145 ảnh

| Model | Đúng/tổng | Exact-match | CER |
| --- | ---: | ---: | ---: |
| ABINet | 894/1145 | 78.08% | 4.59% |
| MASTER | 911/1145 | 79.56% | 4.25% |
| SAR | 916/1145 | 80.00% | 4.37% |
| SATRN | 915/1145 | 79.91% | 4.19% |

### DL7981 — 961 ảnh (public chưa xác minh)

| Model | Đúng/tổng | Exact-match | CER |
| --- | ---: | ---: | ---: |
| ABINet | 597/961 | 62.12% | 8.92% |
| MASTER | 585/961 | 60.87% | 9.21% |
| SAR | 576/961 | 59.94% | 9.17% |
| SATRN | 622/961 | 64.72% | 8.30% |

### Liza Set 1 — 924 ảnh (public chưa xác minh)

| Model | Đúng/tổng | Exact-match | CER |
| --- | ---: | ---: | ---: |
| ABINet | 173/924 | 18.72% | 24.68% |
| MASTER | 215/924 | 23.27% | 28.29% |
| SAR | 128/924 | 13.85% | 34.26% |
| SATRN | 218/924 | 23.59% | 26.55% |

### Liza Set 2 — 1.000 ảnh (public chưa xác minh)

| Model | Đúng/tổng | Exact-match | CER |
| --- | ---: | ---: | ---: |
| ABINet | 185/1000 | 18.50% | 30.46% |
| MASTER | 197/1000 | 19.70% | 36.67% |
| SAR | 121/1000 | 12.10% | 40.88% |
| SATRN | 219/1000 | 21.90% | 32.50% |

## PaddleOCR — PP-OCRv6 curated checkpoint

Đây là checkpoint cục bộ `PPOCRv6_curated_s1024.pdparams`, KHÁC checkpoint `PP-OCRv6 Canonical Baseline (EXP-18B)` trong các bảng phía trên. Checkpoint không có training config nhúng; evaluator dùng topology inference được tái dựng từ loader PaddleOCR baseline đã kiểm tra, sau đó bắt buộc strict-match toàn bộ 422 key và shape. Vì vậy, coi đây là phép đo tái lập theo topology đã ghi, không phải khẳng định rằng checkpoint metadata có chứa config gốc.

| Dataset | Đúng/tổng | Exact-match | CER |
| --- | ---: | ---: | ---: |
| In-domain Test | 537/585 | 91.79% | 2.50% |
| Cross-data | 939/1145 | 82.01% | 3.74% |
| DL7981 | 721/961 | 75.03% | 5.89% |
| Liza Set 1 | 397/924 | 42.97% | 18.57% |
| Liza Set 2 | 379/1000 | 37.90% | 26.18% |

## Quy trình tái lập, giới hạn và cách đọc

- Runner: [`../scripts/reproduce_editctc_five_split_benchmark.py`](../scripts/reproduce_editctc_five_split_benchmark.py), chạy trong container `MLR_LinhNX` với PaddlePaddle 2.6.2 trên GPU `gpu:0` (host: Quadro RTX 5000 16 GiB). Cả 10 checkpoint được nạp strict theo YAML tương ứng; mọi tensor phải khớp tên và shape. ARCH-4 vì vậy dùng đúng `use_explicit_change_head: true`, khác với script benchmark lịch sử.
- Preprocessing: ảnh BGR, giữ tỉ lệ, resize cao 48, padding chiều rộng tới 320, chuẩn hóa `(x/255 - 0.5)/0.5`; batch 32. Decode bằng `CTCLabelDecode` đúng dict từ YAML. Không hiệu chỉnh trên test. Script lưu raw prediction, ground truth, exact-match flag cho từng ảnh; CER được tính lại khi resume từ JSONL hoàn chỉnh.
- Dataset Cross-data có 1.147 nhãn nhưng thiếu ảnh `cross_00333.jpg` và `cross_00613.jpg`; đo trên 1.145 ảnh còn lại. Ba tập public trên đĩa chưa có hồ sơ provenance/license xác thực, nên chỉ là các split cục bộ được gọi public, không benchmark public được chứng nhận.
- `historical*` PP-OCRv3/v4 là kết quả cũ, chưa tái lập; môi trường/version và checksum inference weights không ghi trong JSON nguồn. Không so sánh trực tiếp với `reproduced` như kết luận khoa học.
- **MMOCR năm seed:** 20 checkpoint (4 họ × 5 seed) được đo trên 5 split bằng `TextRecInferencer`; runtime `workdir-mmocr:1.0.1` (image ID `sha256:d023273543566131251afa2740fddf407f3c6ad4298272da9202982899170862`) dùng MMOCR 1.0.1, PyTorch 2.0.0, CUDA 11.7, MMCV 2.0.1, MMEngine 0.8.4 và MMDetection 3.1.0 trên Quadro RTX 5000, batch 32. Tất cả state dict được strict-load. `checkpoint_selection.json` ghi seed, epoch, validation score, log/checkpoint SHA-256; `manifest.json` ghi checkpoint/config/evaluator hashes và runtime. [`five_seed_summary.json`](../../../results/reproduced_mmocr_five_seed_benchmark_20261009/five_seed_summary.json) cùng 100 prediction JSONL (92.300 dòng) đã được kiểm tra độc lập; checkpoint hash, validation-log hash, quy tắc chọn epoch và metric từng prediction đều được kiểm tra lại.
- Dùng Docker image `workdir-mmocr:1.0.1` và workspace đã mount dưới `/workspace` để tái chạy trên thư mục output mới:

  ```bash
  docker run --rm --gpus all -v /:/workspace workdir-mmocr:1.0.1 python \
    /workspace/media/gmo/linhnx/workspace/workdir_Research/source/EditCTC/scripts/reproduce_mmocr_five_split_benchmark.py \
    --workspace /workspace/media/gmo/linhnx/workspace/workdir_Research \
    --checkpoints /workspace/media/gmo/linhnx/workspace/workdir_Research/checkpoints/retrained_mmocr_curated_20261008 \
    --checkpoint-map /workspace/media/gmo/linhnx/workspace/workdir_Research/results/reproduced_mmocr_five_seed_benchmark_20261009/checkpoint_selection.json \
    --output /workspace/media/gmo/linhnx/workspace/workdir_Research/results/reproduced_mmocr_five_seed_rerun \
    --batch-size 32

  docker run --rm -v /:/workspace workdir-mmocr:1.0.1 python \
    /workspace/media/gmo/linhnx/workspace/workdir_Research/source/EditCTC/scripts/summarize_mmocr_five_seed_benchmark.py \
    --results /workspace/media/gmo/linhnx/workspace/workdir_Research/results/reproduced_mmocr_five_seed_rerun \
    --workspace /workspace/media/gmo/linhnx/workspace/workdir_Research
  ```

- Kết quả single-checkpoint trước đó vẫn được giữ trong [`../../../results/reproduced_mmocr_five_split_20261008/`](../../../results/reproduced_mmocr_five_split_20261008/); không dùng thay cho kết quả 5 seed mới. Config Python xuất kèm ở từng thư mục kết quả giữ nguyên config nhúng, chỉ đổi đường dẫn dictionary sang file local.
- Lệnh PP-OCRv6 trong container `MLR_LinhNX`: `python source/EditCTC/scripts/reproduce_ppocrv6_five_split_benchmark.py --workspace "$WORKSPACE" --source "$WORKSPACE/source/EditCTC" --checkpoints "$WORKSPACE/checkpoints/Benchmark_Baselines_Best_Checkpoints" --output "$WORKSPACE/results/reproduced_ppocrv6_five_split_20261008"`.
- **PP-OCRv6 local checkpoint:** đã đo toàn bộ năm dataset bằng PaddleOCR local code, PaddlePaddle 2.6.2, GPU Quadro RTX 5000, batch 32; 422 key/shape strict-match. Config inference `../config/ppocrv6_small_curated_baseline_eval.yml` không phải config huấn luyện nhúng (checkpoint không lưu config); nguồn topology và SHA-256 config, evaluator, checkpoint được ghi trong manifest. Có 4.615 dự đoán theo ảnh và metric đã tính lại độc lập: [`../../../results/reproduced_ppocrv6_five_split_20261008/`](../../../results/reproduced_ppocrv6_five_split_20261008/). Không nhập kết quả này vào các hàng EXP-18B.
- PP-OCRv4 chưa dựng được: code PaddleOCR hiện tại không còn backbone `PPLCNetV3` cần bởi checkpoint, strict-load/forward chưa đạt. Không đưa các kiểm tra này vào bảng metric.
- Checkpoint inventory và SHA-256: `ABINet_curated_s1024.pth` `82b571dddfafb7d510107aefa29a057ee5434f88c3e924e034abc35ea8fa07c0`; `MASTER_curated_s1024.pth` `ffb6106b3b1231fd3affc86b8214ee9872c545d674b5e9957d2790e6b6faff19`; `SAR_curated_s1024.pth` `797d05b30f80cd9321c06d97a2cbc2e3f4385c4e70e99d6df6868a943938cac2`; `SATRN_curated_s1024.pth` `efa5e8ee0af6774556e1a29f53b2c00a102d2d6dbfd82b07976997edb29c87a6`; `PPOCRv4_curated_s1024.pdparams` `058a14c9f7d4de484d49438272f66511b4a543b06dacfb4f8216db58d442f09a`; `PPOCRv6_curated_s1024.pdparams` `4aa429116510ffd1ca6bdee819ad17c41b5b9cdcde1f4bd33986b341184d91ab`.
- Các giá trị làm tròn 2 chữ số. MMOCR báo cáo sample SD giữa 5 seed; các baseline khác có thể chỉ là một checkpoint, không được suy rộng thành độ bất định giữa seed. Không tuyên bố SOTA.

Chi tiết EditCTC: [manifest, SHA-256 và metrics](../../../results/reproduced_editctc_five_split_20261008_retry/manifest.json); [50 tệp JSONL dự đoán](../../../results/reproduced_editctc_five_split_20261008_retry/). MMOCR năm seed: [manifest](../../../results/reproduced_mmocr_five_seed_benchmark_20261009/manifest.json), [tóm tắt và prediction SHA-256](../../../results/reproduced_mmocr_five_seed_benchmark_20261009/five_seed_summary.json), [checkpoint selection](../../../results/reproduced_mmocr_five_seed_benchmark_20261009/checkpoint_selection.json), [100 tệp JSONL](../../../results/reproduced_mmocr_five_seed_benchmark_20261009/). Kết quả MMOCR single-checkpoint trước đây: [manifest](../../../results/reproduced_mmocr_five_split_20261008/manifest.json), [20 tệp JSONL](../../../results/reproduced_mmocr_five_split_20261008/). PP-OCRv6: [manifest, config/checkpoint hashes và metrics](../../../results/reproduced_ppocrv6_five_split_20261008/manifest.json); [5 tệp JSONL dự đoán](../../../results/reproduced_ppocrv6_five_split_20261008/). Lệnh EditCTC đã dùng trong container:

```bash
docker exec MLR_LinhNX python \
  /workspace/media/gmo/linhnx/workspace/workdir_Research/source/EditCTC/scripts/reproduce_editctc_five_split_benchmark.py \
  --workspace /workspace/media/gmo/linhnx/workspace/workdir_Research \
  --output /workspace/media/gmo/linhnx/workspace/workdir_Research/results/reproduced_editctc_five_split_20261008_retry
```
