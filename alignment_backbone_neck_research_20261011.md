# Kế hoạch backbone, neck và alignment feature cho CTC OCR

**Ngày khảo sát:** 2026-10-11
**Phạm vi:** recognition trên ảnh crop đồng hồ nước; đầu ra hiện tại là **CTC head**, chưa đưa EditCTC correction head vào.
**Mục tiêu:** kế thừa backbone/neck từ pretrained PaddleOCR, sau đó thử một module alignment feature độc lập và có thể kiểm toán.

## Quyết định ngắn gọn

1. **Điểm xuất phát được khuyến nghị:** `PPLCNetV4 + light-SVTR neck + CTCHead`, kế thừa PP-OCRv6 Small. Đây là topology đã có checkpoint local/official và tương thích trực tiếp với pipeline hiện tại.
2. **Hai baseline đối chứng bắt buộc:** `PPHGNet_small + light-SVTR + CTCHead` (PP-OCRv4 Server) và `PPHGNetV2_B4 + light-SVTR + CTCHead` (PP-OCRv5 Server).
3. **Alignment v1 nên là residual, zero-init, không thay đổi kích thước tensor:** alignment nhận feature 2-D trước khi collapse thành chuỗi, học offset/gate nhỏ rồi cộng residual vào feature gốc. Nếu module chưa học được, model vẫn suy biến an toàn về CTC baseline.
4. **Không dùng test để chọn module/ngưỡng.** Chọn bằng curated validation; báo cáo test theo năm seed, exact-match và CER.

## 1. Những backbone có thể thử

Danh sách dưới đây được kiểm tra từ `PaddleOCR local registry: ppocr/modeling/backbones/__init__.py` và các config recognition trong `PaddleOCR upstream configs`.

| Backbone | Config/checkpoint tham chiếu | Khả năng kế thừa pretrained | Vai trò đề xuất |
| --- | --- | --- | --- |
| `PPLCNetV4` | `configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml` | **Tốt nhất / trực tiếp**; PP-OCRv6 Small, Medium, Tiny có topology tương ứng | Baseline chính; nhẹ, đã có 2-D feature phù hợp alignment |
| `PPHGNet_small` | `configs/rec/PP-OCRv4/PP-OCRv4_server_rec.yml` | **Tốt** nếu dùng đúng PP-OCRv4 Server checkpoint và config | Baseline server mạnh, kiểm tra lợi ích alignment trên backbone lớn |
| `PPHGNetV2_B4` | `configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml` | **Tốt** nếu dùng đúng PP-OCRv5 Server checkpoint | Baseline hiện đại thứ hai |
| `PPLCNetV3` | `configs/rec/PP-OCRv4/PP-OCRv4_mobile_rec.yml`, PP-OCRv5 mobile | Có official mobile weights; cần khớp đúng phiên bản/config | Baseline nhẹ hơn, kiểm tra trade-off tốc độ |
| `RepSVTR` | `configs/rec/SVTRv2/ch_RepSVTR_rec.yml` | Có official SVTRv2-family weights khi topology khớp | Backbone recognition có re-parameterization; alignment sau feature map cần audit shape |
| `SVTRv2` | `configs/rec/SVTRv2/ch_SVTRv2_rec.yml` | **Đã retrain trong benchmark**, có config/checkpoint per seed | Đối chứng transformer; không tự động coi là checkpoint tương thích với PPLCNet |
| `SVTRNet` | `configs/rec/rec_svtrnet.yml` | Có config/implementation; checkpoint phải xác minh riêng | Đối chứng attention/sequence |
| `ResNet`, `ResNet31/32/45`, `ResNetFPN` | `configs/rec/rec_r*.yml` | Có implementation; official weight phụ thuộc từng recipe | Ablation cổ điển, không ưu tiên trước khi baseline PaddleOCR hoàn tất |
| `ViTSTR`, `ViTParseQ`, `HybridTransformer`, `DonutSwinModel`, `Vary_VIT_B` | Các config recognition tương ứng | **Không nên là vòng đầu**; checkpoint và input/topology không đồng nhất với PP-OCR CTC | Chỉ thử ở nhánh nghiên cứu riêng |

### Quy tắc load weight

- Không gọi “pretrained” nếu chỉ có class implementation. Mỗi run phải ghi `pretrained_path`, SHA-256, số tensor khớp, shape mismatch và source URL/license.
- Backbone khác nhau **không thể** dùng chung checkpoint chỉ vì cùng số output channels.
- Khi thêm alignment sau backbone, phải chứng minh load được toàn bộ backbone/neck cũ; alignment parameters phải được đánh dấu newly initialized.

## 2. Những neck có thể thử

PaddleOCR đăng ký các neck tổng quát trong `ppocr/modeling/necks/__init__.py`, nhưng không phải neck nào cũng dành cho recognition. Các lựa chọn khả dụng cho CTC recognition là:

| Neck/config name | Dạng feature | Tương thích với CTC | Khuyến nghị |
| --- | --- | --- | --- |
| `SequenceEncoder` với `encoder_type: reshape` | Collapse chiều cao thành chuỗi | **Cao**, ít tham số | Baseline tối giản để tách riêng tác động của alignment |
| `SequenceEncoder` với `encoder_type: rnn` | Chuỗi + BiLSTM/GRU | Cao nhưng tăng latency | Ablation về temporal context; không dùng trong baseline tốc độ đầu tiên |
| `SequenceEncoder` với `encoder_type: svtr` / `svtr` trong `MultiHead` | Light-SVTR sequence modeling | **Cao**, dùng trong PP-OCRv4/v5/v6 | **Khuyến nghị chính**; giữ nguyên khi thêm alignment |
| `reshape` trong PP-OCRv6 Tiny | Collapse nhẹ trước MultiHead | Cao cho Tiny nếu giữ đúng config | Baseline edge/mobile |
| `SVTRv2` backbone nội bộ | Sequence/2-D transformer đã tích hợp | CTC-compatible theo config SVTRv2 | Thử riêng, không trộn với PPLCNet neck |
| `RFAdaptor` | Receptive-field adaptor | Không phải neck CTC benchmark mặc định | Chỉ thử nếu cần mở rộng receptive field; phải kiểm tra output rank |
| `FPN`, `FCEFPN`, `DBFPN`, `RSEFPN`, `CSPPAN`, `FPN_UNet`, ... | Multi-scale detection feature | **Không phù hợp trực tiếp** với CTC recognition | Không đưa vào vòng đầu; đây là neck detection/table |

**Lưu ý:** với PP-OCRv4/v5/v6, “neck” thường là một phần của `CTCHead`/`MultiHead` (`CTCHead.Neck`), không phải luôn là top-level `Architecture.Neck`. Vì vậy phải giữ nguyên nesting và kiểm tra state-dict key; đổi vị trí neck sẽ làm mất khả năng strict-load.

## 3. Alignment feature module khả thi

Alignment ở đây có nghĩa là làm cho feature thị giác tại vị trí ngang của ký tự nhất quán hơn trước khi CTC collapse/decoding. Đây là **đề xuất thiết kế cho dự án**, không phải claim rằng các paper dưới đây đã giải quyết water-meter OCR.

### A. Learned deformable sampling / DCNv4-style residual alignment

**Ý tưởng:** dự đoán offset `(dx, dy)` và modulation/gate từ feature; sample lân cận thích nghi thay vì dùng kernel cố định:

```text
F_align = F + alpha * DeformSample(F, offset(F), gate(F))
```

DCNv4 (Xiong et al., 2024) bỏ softmax normalization trong spatial aggregation và tối ưu memory access; paper báo cáo operator nhanh hơn DCNv3 và có thể thay vào vision backbone. Nguồn: [arXiv:2401.06197](https://arxiv.org/abs/2401.06197), code [OpenGVLab/DCNv4](https://github.com/OpenGVLab/DCNv4).

**Phù hợp:** ảnh số bị nghiêng, crop lệch, stroke không nằm ổn định trên grid.
**Rủi ro:** custom CUDA/operator và checkpoint portability; implementation DCNv4 không có sẵn trong PaddleOCR tree. Nên bắt đầu bằng Paddle-native approximation, chưa port kernel CUDA.

### B. Offset-guided deformable cross-attention

**Ý tưởng:** dùng sequence query của CTC hoặc query cố định theo timestep để dự đoán một số điểm sample trên feature 2-D; cross-attention chỉ đọc vùng lân cận thay vì toàn bộ map:

```text
q_t -> offsets_t, weights_t -> sample(F, x_t + offsets_t) -> aligned_t
```

Đây là biến thể recognition-specific của deformable attention: dễ giữ output `(B, T, C)` và không cần EditCTC head. Có thể triển khai bằng `paddle.nn.functional.grid_sample`/bilinear sampling hoặc gather nội suy thuần Paddle, tùy operator có trong runtime.

**Phù hợp:** CTC đã cung cấp timestep/peak; alignment có inductive bias đúng với chuỗi ngang.
**Rủi ro:** offset collapse hoặc học shortcut từ CTC; cần log offset norm, entropy weights và ablation `offset=0`.

### C. Confidence-conditioned alignment gate

**Ý tưởng:** không chỉnh mọi vị trí như nhau. Dùng confidence CTC (`p_top1`, margin, entropy) để điều khiển một residual gate:

```text
g_t = sigmoid(MLP([feature_t, confidence_t]))
F_align_t = F_t + g_t * AlignResidual_t
```

Vị trí confidence cao gần identity; vị trí mơ hồ được phép lấy thêm visual context. Đây là lựa chọn **an toàn nhất cho v1** vì không cần thay đổi CTC head và có thể zero-init gate.

### D. Multi-scale cross-level alignment

**Ý tưởng:** căn chỉnh feature high-resolution với feature sequence low-resolution bằng một projection và cross-attention/gated fusion:

```text
F_low = Project(F_low)
F_high = Project(F_high)
F_out = F_low + sigmoid(G([F_low, Downsample(F_high)])) * F_high
```

**Phù hợp:** phân biệt stroke nhỏ (`3/8`, `8/9`) và xử lý crop/blur.
**Rủi ro:** lệch coordinate giữa hai stride; bắt buộc ghi rõ `(H,W)` và mapping `x_high -> x_low`, không chỉ resize bằng mắt.

### E. Coordinate-aware alignment bias

**Ý tưởng:** tạo positional bias theo tọa độ ngang và chiều cao, hoặc local window bias, trước CTC sequence projection. Đây là lựa chọn nhẹ hơn deformable sampling và tương đồng với prior spatial đã có trong các thí nghiệm EditCTC, nhưng **ở giai đoạn này chỉ dùng trên feature trước CTC**, không dùng correction decoder.

**Phù hợp:** baseline alignment rẻ, dễ kiểm toán.
**Rủi ro:** có thể chỉ học lại positional encoding mà không thật sự căn chỉnh; cần so với `SequenceEncoder` nguyên bản.

## 4. Thứ tự thí nghiệm đề xuất

| Phase | Backbone/neck | Alignment | Mục đích |
| --- | --- | --- | --- |
| 0 | PPLCNetV4 + light-SVTR | Không có | Reproduce CTC baseline; kiểm tra pretrained load và metric |
| 1 | PPLCNetV4 + light-SVTR | Confidence-conditioned residual gate | **Ứng viên đầu tiên**, ít rủi ro |
| 2 | PPLCNetV4 + light-SVTR | Offset-guided local sampling | Kiểm tra căn chỉnh không gian thực sự |
| 3 | PPHGNet_small + light-SVTR | Cùng module tốt nhất Phase 1/2 | Kiểm tra phụ thuộc backbone |
| 4 | PPHGNetV2_B4 + light-SVTR | Cùng module | Đối chứng PP-OCRv5 |
| 5 | SVTRv2 | Cùng module nếu shape phù hợp | Kiểm tra alignment trên transformer backbone |

Mỗi phase tối thiểu chạy seeds `1024, 2024, 3024, 4024, 5024`; smoke test trước khi chạy đủ seed. Không chọn module theo public test.

## 5. Yêu cầu config và kiểm toán

Mỗi run phải lưu immutable copy của YAML dưới `checkpoints/<run>/seed_<seed>/config.yml`, cùng:

- `config_sha256`, `pretrained_sha256`, code revision;
- backbone/neck/head name và mọi channel/stride/shape;
- alignment name, version, init strategy, parameter count;
- seed, batch size, learning rate, optimizer, epoch, augmentation;
- train/validation/test label paths và dataset counts;
- checkpoint selection rule: validation only;
- checkpoint SHA-256, log SHA-256, prediction JSONL SHA-256;
- exact-match, CER, sample count và sample SD giữa seeds.

Các config benchmark hiện có không được xóa hoặc sửa ngầm. Config gốc và config per-seed phải tiếp tục được tham chiếu từ manifest kết quả.

## 6. Giới hạn diễn giải

- Một module alignment có thể cải thiện metric nhưng chưa chứng minh được “alignment” nếu không log offset/gate hoặc có ablation identity.
- Official pretrained weights có license/provenance riêng; không gọi là reproduction nếu đổi backbone, neck, preprocessing hoặc checkpoint.
- DL7981 và Liza vẫn có hạn chế provenance/license/split-independence như ghi trong benchmark.
- Các kết quả ARCH-7 official-initialization bị collapse về CTC output và **không được dùng làm bằng chứng cho alignment module mới**.

## Tài liệu và nguồn code đã kiểm tra

- PaddleOCR recognition registry: [`ppocr/modeling/backbones/__init__.py`](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/modeling/backbones/__init__.py) và [`ppocr/modeling/necks/__init__.py`](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/modeling/necks/__init__.py) — local source snapshot.
- PP-OCRv6 Small config: [`source/PADDLEOCR/configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml`](https://github.com/PaddlePaddle/PaddleOCR/blob/main/configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml).
- PP-OCRv4 Server config: [`source/PADDLEOCR/configs/rec/PP-OCRv4/PP-OCRv4_server_rec.yml`](https://github.com/PaddlePaddle/PaddleOCR/blob/main/configs/rec/PP-OCRv4/PP-OCRv4_server_rec.yml).
- PP-OCRv5 Server config: [`source/PADDLEOCR/configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml`](https://github.com/PaddlePaddle/PaddleOCR/blob/main/configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml).
- SVTRv2 config: [`source/PADDLEOCR/configs/rec/SVTRv2/ch_SVTRv2_rec.yml`](https://github.com/PaddlePaddle/PaddleOCR/blob/main/configs/rec/SVTRv2/ch_SVTRv2_rec.yml).
- Xiong et al., *Efficient Deformable ConvNets: Rethinking Dynamic and Sparse Operator for Vision Applications*, 2024: [arXiv:2401.06197](https://arxiv.org/abs/2401.06197).
- Deformable attention background: Zhu et al., *Deformable DETR*, ICLR 2021: [arXiv:2010.04159](https://arxiv.org/abs/2010.04159).
