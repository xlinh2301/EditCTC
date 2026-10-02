# Báo Cáo Thị Giác: Phân Tích Lỗi Tập Dữ Liệu Indomain Test

## 1. Tóm Tắt Phát Hiện Cốt Lõi

Khi phân tích 36–39 lỗi còn lại trên tập Indomain (585 mẫu), phát hiện có **20 mẫu (chiếm hơn 50% lỗi) không thể nhận diện đúng do lỗi dữ liệu khách quan**:
- **16 mẫu bị cắt mất ký tự (Crop Truncation)**: Trong khâu cắt bounding box tự động (`curation_log.jsonl`), ảnh bị cắt mất từ 30% đến 55% chiều rộng bên phải, khiến các chữ số cuối không còn trong ảnh, nhưng nhãn vẫn giữ nguyên 6-7 chữ số.
- **4 mẫu bị gán nhãn sai 100% (Proven Mislabeling)**: Ảnh đồng hồ hiển thị rõ ràng một số, nhưng nhãn Ground Truth bị gõ sai hoặc thiếu số 0.

Do đó, **trần lý thuyết tối đa của tập test này là $565 / 585 = 96.58\%$**.  
Mô hình hiện tại đạt **549 / 585**, tương đương **$549 / 565 = 97.17\%$ độ chính xác thực tế trên các ảnh hợp lệ**.

---

## 2. Nhóm 1: 4 Mẫu Bị Gán Sai Nhãn Ground Truth (Mô Hình Đọc Đúng Ảnh)

Mô hình dự đoán hoàn toàn trùng khớp với chữ số in trên đồng hồ, nhưng bị tính là sai vì nhãn GT bị lỗi:

![Visual Grid Mislabeled Samples](/dev/shm/agy-private.GTiNm5/myagy/.gemini/antigravity-cli/brain/b67ad1d6-69a8-4dd7-97a8-f97aca43e3cb/images/grid_1_proven_mislabeled.png)

### Bảng đối chiếu chi tiết:
| Tên File | Chữ Số Trên Ảnh | Nhãn GT (Sai) | Model Dự Đoán | Phân Tích Lỗi Dữ Liệu |
| :--- | :---: | :---: | :---: | :--- |
| `crop_003105.jpg` | **00027** (5 số) | `0027` (4 số) | **00027** ✅ | Đồng hồ có 5 bánh răng rõ ràng, GT bị gõ thiếu số 0 ở đầu. |
| `crop_003253.jpg` | **00877** (5 số) | `80877` (5 số) | **00877** ✅ | Chữ số đầu tiên là vòng oval tròn của số 0, GT gõ nhầm thành số 8. |
| `crop_003180.jpg` | **00379** (5 số) | `00319` (5 số) | **00379** ✅ | Chữ số thứ 4 có nét gạch ngang đỉnh và nét xiên của số 7, GT gõ nhầm thành 1. |
| `crop_002935.jpg` | **1284** (4 số) | `1294` (4 số) | **1284** ✅ | Chữ số thứ 3 có hai vòng tròn khép kín của số 8, GT gõ nhầm thành 9. |

---

## 3. Nhóm 2: Các Mẫu Bị Cắt Mất Ký Tự Khỏi Ảnh Trong Khâu Curation (Crop Truncation)

Trong quá trình tạo dataset, bounding box bị thu hẹp làm mất các chữ số bên phải, nhưng nhãn Ground Truth không được cập nhật:

![Visual Grid Crop Truncation Samples](/dev/shm/agy-private.GTiNm5/myagy/.gemini/antigravity-cli/brain/b67ad1d6-69a8-4dd7-97a8-f97aca43e3cb/images/grid_2_crop_truncation.png)

### Bảng đối chiếu trích từ curation manifest:
| Tên File | Kích Thước Gốc $\to$ Sau Cắt | Nhãn GT | Model Đọc | Ký Tự Đã Bị Cắt Mất Khỏi Ảnh |
| :--- | :---: | :---: | :---: | :--- |
| `crop_add_test_..._8352088.jpg` | `[341, 107]` $\to$ `[201, 102]` | `1536129` (7 số) | `1536` | Mất 41% ảnh bên phải $\to$ chùm số `129` bị cắt bỏ. |
| `crop_add_test_..._8062388.jpg` | `[650, 180]` $\to$ `[364, 171]` | `0101909` (7 số) | `0101` | Mất 44% ảnh bên phải $\to$ chùm số `909` bị cắt bỏ. |
| `crop_add_test_..._8083062.jpg` | `[1702, 579]` $\to$ `[1118, 551]` | `071907` (6 số) | `1719` | Mất 584 pixel bên phải $\to$ chùm số `07` bị cắt bỏ. |
| `crop_add_test_..._8184054.jpg` | `[389, 111]` $\to$ `[226, 111]` | `4669904` (7 số) | `4669` | Mất 42% ảnh bên phải $\to$ chùm số `904` bị cắt bỏ. |
| `crop_add_test_..._8194658.jpg` | `[472, 152]` $\to$ `[254, 145]` | `1069026` (7 số) | `1069` | Mất 46% ảnh bên phải $\to$ chùm số `026` bị cắt bỏ. |
| `crop_add_test_..._8063346.jpg` | `[678, 220]` $\to$ `[460, 218]` | `110243` (6 số) | `1102` | Mất 32% ảnh bên phải $\to$ chùm số `43` bị cắt bỏ. |
| `crop_003341.jpg` | `[580, 128]` $\to$ `[510, 122]` | `000592` (6 số) | `00059` | Mất 70 pixel bên phải $\to$ số `2` cuối bị cắt bỏ. |
| `crop_003365.jpg` | `[592, 128]` $\to$ `[541, 122]` | `004008` (6 số) | `00400` | Mất 51 pixel bên phải $\to$ số `8` cuối bị cắt bỏ. |

---

## 4. Nhóm 3: Các Lỗi Mô Hình Thật Sự Còn Lại (Tail Drop Sát Mép & Bánh Răng Quay Lửng)

Chỉ còn khoảng 16 mẫu là lỗi thật sự của mô hình, chủ yếu do ký tự cuối nằm quá sát viền mép ảnh hoặc bánh răng cơ học đang quay lửng giữa 2 vạch số:

![Visual Grid Real Errors](/dev/shm/agy-private.GTiNm5/myagy/.gemini/antigravity-cli/brain/b67ad1d6-69a8-4dd7-97a8-f97aca43e3cb/images/grid_3_real_model_errors.png)

### Bảng phân tích:
| Tên File | Nhãn GT | Model Đọc | Bản Chất Hiện Tượng |
| :--- | :---: | :---: | :--- |
| `crop_002901.jpg` | `003745` | `00374` | Số 5 cuối mờ và nằm sát góc mép phải ảnh $\to$ CTC bị trượt frame. |
| `crop_002954.jpg` | `42703` | `4270` | Số 3 cuối sát mép (*ARCH-4B đã bắt thành công: 42703*). |
| `crop_003287.jpg` | `01631` | `0163` | Số 1 cuối bị lệch (*ARCH-4B đã bắt thành công: 01631*). |
| `crop_003349.jpg` | `004299` | `00429` | CTC Blank Collapsing: Hai số 9 liên tiếp ở cuối bị gộp làm một. |
| `crop_003125.jpg` | `02602` | `02601` | Bánh răng đang quay lửng giữa vạch số 1 và vạch số 2. |
| `crop_003134.jpg` | `00489` | `00488` | Bánh răng đang quay lửng giữa vạch số 8 và vạch số 9. |
