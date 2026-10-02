# Workspace 5 Transfer Manifest & Hướng Dẫn Sao Lưu Về Máy Cá Nhân

Tài liệu này tổng hợp toàn bộ các file, thư mục, kết quả thực nghiệm, mã nguồn Git và session chat của Antigravity (AGY) từ server DGX-A100 (`workspace5`) để bạn thuận tiện tải về máy cá nhân và tiếp tục phát triển.

---

## 1. Trạng Thái Mã Nguồn & Git (Đã Push Thành Công)

Toàn bộ code mới nhất của EditCTC (bao gồm toàn bộ 13 biến thể kiến trúc ARCH-0 đến ARCH-8, loss functions, modules, scripts đánh giá đa seed) đã được đồng bộ và push lên GitHub cá nhân của bạn:

- **Repository**: [`git@github.com:xlinh2301/EditCTC.git`](https://github.com/xlinh2301/EditCTC)
- **Branch chính**: `main` (đã commit và push toàn bộ 120 commits)
- **Branch tính năng**: `feat/arch-innovations-multiseed` (nhánh chuyên biệt cho chuỗi thực nghiệm kiến trúc)
- **Các thành phần cốt lõi đã được lưu trữ trên Git**:
  - `ppocr/modeling/heads/rec_edit_refine_nrtr_head.py`: Chứa toàn bộ các khối kiến trúc (Change Head, Conf 2D/4D, Temporal Align, Align-Guided Cross-Attention, Append Head, Dual Cross-Attention Gated Fusion, Local Refinement DWConv).
  - `ppocr/losses/rec_edit_loss_token_refine.py`: Hàm loss tinh chỉnh token và change gate BCE.
  - `ppocr/losses/rec_multi_loss_editrefine_token.py`: Đa loss kết hợp (CTC + Length + EditTok + ChangeGate).
  - `config/PP-OCRv6_small_rec_s1024_e44_highres_arch*.yml`: Toàn bộ 12 file cấu hình tương ứng từng bài thử nghiệm kiến trúc.
  - `tools/eval.py`, `tools/infer_rec.py`, `tools/train.py`, `tools/sweep_pareto_curve.py`: Công cụ huấn luyện, suy luận và sweep threshold.

---

## 2. Toàn Bộ Session Chat AGY & Trực Quan Hóa Lỗi (Đã Lưu Trữ)

Toàn bộ dữ liệu session chat AGY từ bộ nhớ tạm `/dev/shm` đã được sao chép an toàn vào workspace tại thư mục:  
📂 `/datastore/cndt_thangcpd/linhtruong/workspace5/agy_session_backup/`

Các file bao gồm:
1. `transcript_session.json`: Toàn bộ **2.404 bước hội thoại** chi tiết, định dạng JSON chuẩn (có thể mở bằng bất kỳ trình đọc JSON nào).
2. `transcript_full.jsonl` & `transcript.jsonl`: Bản ghi log hội thoại gốc của Antigravity CLI.
3. `visual_error_audit.md`: Báo cáo audit trực quan toàn bộ ~36 ca lỗi trên tập Indomain.
4. `images/grid_1_proven_mislabeled.png`: Lưới ảnh trực quan 4 mẫu dán sai nhãn 100%.
5. `images/grid_2_crop_truncation.png`: Lưới ảnh trực quan 16 mẫu bị crop cụt mất đuôi (lỗi dữ liệu vật lý).
6. `images/grid_3_real_model_errors.png`: Lưới ảnh 16 mẫu lỗi thực sự của mô hình (bị kẹt số ở công tơ, dính viền).

---

## 3. Kết Quả Thực Nghiệm Đa Seed (100% Hoàn Tất - 52/52 Seeds)

Tất cả 52 lượt huấn luyện bổ sung (4 seed mới cho mỗi kiến trúc) **đã hoàn thành 100%** và được tự động tổng hợp đầy đủ số liệu Mean ± Std vào:  
📄 `/datastore/cndt_thangcpd/linhtruong/workspace5/OPENSPEC_MULTISEED_RESULTS.md`

### Bảng Thống Kê Tổng Hợp (Statistical Summary):

| Kiến Trúc | Số Seed | Indomain Acc (Mean ± Std) | Indomain CER | Crossdata Acc (Mean ± Std) | Crossdata CER | Điểm Nổi Bật |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **PP-OCRv4 Baseline (Official)** | *Ref* | *93.68% (548/585)* | *2.36%* | *87.86% (1006/1145)* | *2.74%* | Mốc tham chiếu |
| **EXP-18B (Canonical Baseline)** | **6/6** | **93.30 ± 0.32%** | **2.32%** | **88.27 ± 0.70%** | **2.66%** | Đỉnh cao: **93.85%** (549/585) 👑 |
| **ARCH-4 (Align-Guided Cross-Attn)**| **5/5** | **93.09 ± 0.30%** | **2.31%** | **90.22 ± 0.81%** | **2.22%** | Kỷ lục Crossdata: **91.35%** 🥇 |
| **ARCH-4C (Align + Conf 4D)** | **5/5** | **93.06 ± 0.26%** | **2.33%** | **90.27 ± 0.37%** | **2.18%** | Tổng quát hóa cực vững (Std 0.37%) |
| **ARCH-2B (Full Conf 4D)** | **5/5** | **93.03 ± 0.25%** | **2.33%** | **89.48 ± 0.92%** | **2.39%** | Bảo toàn seed tốt |
| **ARCH-1 (Explicit Change Head)** | **5/5** | **92.85 ± 0.23%** | **2.39%** | **89.22 ± 1.23%** | **2.44%** | Over-Correction thấp nhất (0.08%) |
| **ARCH-5 (Local Visual Refine)** | **5/5** | **92.89 ± 0.17%** | **2.38%** | **89.66 ± 0.82%** | **2.33%** | Độ lệch Indomain thấp nhất (0.17%) |
| **ARCH-7 (Gated Memory Fusion)** | **5/5** | **92.85 ± 0.23%** | **2.39%** | **88.68 ± 1.99%** | **2.56%** | Cân bằng CTC và Visual |
| **ARCH-6 (Backbone S5 Unfreeze)** | **5/5** | **92.65 ± 0.78%** | **2.34%** | **84.10 ± 0.89%** | **3.28%** | CER Indomain chạm mốc 2.21% |
| **ARCH-3 (Temporal Align Emb)** | **5/5** | **92.58 ± 0.59%** | **2.45%** | **89.61 ± 1.20%** | **2.35%** | Neo vị trí CTC 40 steps |
| **ARCH-2A (CTC Conf 2D)** | **5/5** | **92.82 ± 0.36%** | **2.41%** | **89.00 ± 0.71%** | **2.52%** | Top1 + Margin gating |
| **ARCH-4B (Align-Guided + Append)** | **5/5** | **91.56 ± 0.70%** | **2.62%** | **79.34 ± 4.57%** | **4.50%** | Bắt được 2 lỗi rớt đuôi khó |
| **ARCH-8 (4-Way Op + Gap Insert)** | **5/5** | **91.76 ± 1.10%** | **2.58%** | **81.40 ± 3.79%** | **3.88%** | 4 thao tác Levenshtein |
| **ARCH-4D (Grand Combo)** | **5/5** | **91.42 ± 0.54%** | **2.65%** | **78.57 ± 2.98%** | **4.62%** | Combo 3 module |

---

## 4. Danh Mục Các Thư Mục & File Cần Tải Về Máy

Dưới đây là các nhóm file ưu tiên theo mức độ quan trọng:

### Nhóm 1: Báo Cáo, Metrics & Session Chat (Dung lượng: ~25 MB - Bắt buộc tải)
- `OPENSPEC_MULTISEED_RESULTS.md`: Báo cáo đa seed chi tiết từng seed và review code chuyên sâu.
- `openspec/specs/OPENSPEC_EDITCTC_ARCH_EXPERIMENTS.md`: Master spec kiến trúc.
- `agy_session_backup/`: Toàn bộ session JSON/JSONL và 3 ảnh error grid.
- `workdir_eval/recognition/curated_results/`: Toàn bộ file JSON metrics và `predictions.txt` của tất cả các kiến trúc.
- `workdir_eval/recognition/curated_eval_labels/`: Nhãn ground truth chuẩn hóa (`indomain_test_label.present.txt`, `crossdata_label.present.txt`).

### Nhóm 2: Dữ Liệu Đã Chuẩn Hóa (Curated Datasets) (Dung lượng: ~50 MB)
- `Data/Indomain_curated/`: Toàn bộ ảnh crops valid/test (585 ảnh), file nhãn và file nhật ký `curation_log.jsonl` (ghi nhận 20 lỗi crop vật lý).
- `Data/Cross-data_curated/`: Toàn bộ 1.145 ảnh test cross-domain và file nhãn.

### Nhóm 3: Các Checkpoint Mô Hình Tốt Nhất (Trọng số Inference `.pdparams`) (Dung lượng: ~162 MB mỗi checkpoint)
> **Mẹo tiết kiệm băng thông:** Chỉ cần tải file `best_accuracy.pdparams` (trọng số mô hình), không cần tải file `.pdopt` (trạng thái optimizer 207 MB) trừ khi muốn resume training optimizer.

1. **Quán quân Indomain (EXP-18B, 93.85%)**:
   - `Data/EditCTC_arbor_runs/curated_runs/e44_highres_exp18b_canonical_g4.0_s3024/best_accuracy.pdparams`
2. **Quán quân Crossdata (ARCH-4, 91.35%)**:
   - `Data/EditCTC_arbor_runs/curated_runs/e44_highres_arch4_align_guided_cross_attn/best_accuracy.pdparams`
3. **Quán quân Cân Bằng (ARCH-4C, 90.27% Cx / 93.06% In)**:
   - `Data/EditCTC_arbor_runs/curated_runs/e44_highres_arch4c_align_conf4d/best_accuracy.pdparams`
4. **Mô hình Khởi Tạo Pretrained**:
   - `Data/EditCTC_arbor_runs/curated_runs/e44_highres_ctc_aware_confusion/best_accuracy.pdparams`

---

## 5. Lệnh Tải Dữ Liệu Nhanh Từ Máy Cá Nhân (Rsync / SCP)

Bạn mở terminal trên **máy cá nhân (Local Machine)** và chạy các lệnh sau:

### Cách 1: Tải nhanh Toàn Bộ Báo Cáo, Metrics, Labels & Session Chat (Khuyên dùng - Rất nhẹ, ~25MB)
```bash
# Thay <SERVER_IP_OR_HOST> và <PORT> bằng thông tin SSH của bạn
rsync -avz -e "ssh -p <PORT>" \
  cndt_thangcpd@<SERVER_IP_OR_HOST>:/datastore/cndt_thangcpd/linhtruong/workspace5/OPENSPEC_MULTISEED_RESULTS.md \
  cndt_thangcpd@<SERVER_IP_OR_HOST>:/datastore/cndt_thangcpd/linhtruong/workspace5/agy_session_backup \
  cndt_thangcpd@<SERVER_IP_OR_HOST>:/datastore/cndt_thangcpd/linhtruong/workspace5/workdir_eval/recognition/curated_eval_labels \
  cndt_thangcpd@<SERVER_IP_OR_HOST>:/datastore/cndt_thangcpd/linhtruong/workspace5/workdir_eval/recognition/curated_results \
  ./workspace5_results_and_session/
```

### Cách 2: Tải Dữ Liệu Curated (Indomain & Cross-data)
```bash
rsync -avz -e "ssh -p <PORT>" \
  cndt_thangcpd@<SERVER_IP_OR_HOST>:/datastore/cndt_thangcpd/linhtruong/workspace5/Data/Indomain_curated \
  cndt_thangcpd@<SERVER_IP_OR_HOST>:/datastore/cndt_thangcpd/linhtruong/workspace5/Data/Cross-data_curated \
  ./workspace5_datasets/
```

### Cách 3: Tải 3 Checkpoint Xuất Sắc Nhất (.pdparams)
```bash
mkdir -p ./workspace5_checkpoints

# 1. EXP-18B Baseline Quán quân Indomain (93.85%)
scp -P <PORT> cndt_thangcpd@<SERVER_IP_OR_HOST>:/datastore/cndt_thangcpd/linhtruong/workspace5/Data/EditCTC_arbor_runs/curated_runs/e44_highres_exp18b_canonical_g4.0_s3024/best_accuracy.pdparams ./workspace5_checkpoints/exp18b_canonical_best.pdparams

# 2. ARCH-4 Quán quân Crossdata (91.35%)
scp -P <PORT> cndt_thangcpd@<SERVER_IP_OR_HOST>:/datastore/cndt_thangcpd/linhtruong/workspace5/Data/EditCTC_arbor_runs/curated_runs/e44_highres_arch4_align_guided_cross_attn/best_accuracy.pdparams ./workspace5_checkpoints/arch4_align_guided_best.pdparams

# 3. ARCH-4C Quán quân Cân Bằng (90.27%)
scp -P <PORT> cndt_thangcpd@<SERVER_IP_OR_HOST>:/datastore/cndt_thangcpd/linhtruong/workspace5/Data/EditCTC_arbor_runs/curated_runs/e44_highres_arch4c_align_conf4d/best_accuracy.pdparams ./workspace5_checkpoints/arch4c_conf4d_best.pdparams
```

### Cách 4: Tạo Gói Nén Tổng Hợp Ngay Trên Server Để Tải 1 Lần Duy Nhất
Trên server, bạn có thể chạy file đóng gói sẵn:
```bash
tar -czvf /datastore/cndt_thangcpd/linhtruong/workspace5/workspace5_bundle_essentials.tar.gz \
  /datastore/cndt_thangcpd/linhtruong/workspace5/OPENSPEC_MULTISEED_RESULTS.md \
  /datastore/cndt_thangcpd/linhtruong/workspace5/WORKSPACE5_TRANSFER_MANIFEST.md \
  /datastore/cndt_thangcpd/linhtruong/workspace5/agy_session_backup \
  /datastore/cndt_thangcpd/linhtruong/workspace5/workdir_eval/recognition/curated_eval_labels \
  /datastore/cndt_thangcpd/linhtruong/workspace5/Data/Indomain_curated \
  /datastore/cndt_thangcpd/linhtruong/workspace5/Data/Cross-data_curated
```
Sau đó ở máy cá nhân chỉ cần kéo 1 file duy nhất:
```bash
scp -P <PORT> cndt_thangcpd@<SERVER_IP_OR_HOST>:/datastore/cndt_thangcpd/linhtruong/workspace5/workspace5_bundle_essentials.tar.gz .
```
