# Đã biết, còn mở, và ràng buộc cho phương pháp cải tiến

> Phần này là **tổng hợp bằng chứng**, không phải quyết định. Câu hỏi nghiên cứu và phương pháp do người thiết kế
> (Quân) chốt cùng nhóm.

## 1. Điều đã khá chắc (B3, 3 seed)

1. **SAGE không hơn VIS có ý nghĩa** (+0,82 [−0,86; +3,04]); bài tuyên bố +12,1 – không tái lập được.
2. **Trộn trung bình TB nhỉnh hơn SAGE** (+1,05\*, cả 3 seed). Nguyên nhân chưa biết.
3. Cái lợi ở **lớp người** (đêm, khói) đến từ **ảnh nhiệt**, không riêng SAGE (IR, TB ngang SAGE).
4. Chỉ số ghép SCD, Q_ABF, EN, SD, SF, AG **không** phản ánh hiểu cảnh; chỉ MS-SSIM (tổng) xếp hạng khớp mIoU (E14).

## 2. Giả thuyết về "vì sao TB ≥ SAGE" – đã loại / còn mở

| Giả thuyết | Kết quả | Thí nghiệm |
|---|---|---|
| SAGE làm mờ biên | Loại | E10 |
| Student kém vì train crop nhỏ | Loại | E9 |
| Ảnh SAGE thiếu độ sắc | Loại (làm sắc còn hại) | E1 |
| SAGE không tăng IR khi VIS tối | Loại | E15b |
| SAGE lấy IR ở nền quá ít | Không ủng hộ | E20 |
| Model phân đoạn lạ ảnh SAGE khi tối (lệch phân phối) | Loại | E15c |
| Nhiễu VIS bị SAGE giữ lại là nguyên nhân chính | Loại (nhiễu có thật nhưng khử đi vẫn thua) | E15d, E15e |
| Thích nghi IR/VIS theo **ảnh** sẽ giải quyết | Trần nhỏ (~+1 so với TB) | E5, E8, E12 |
| **Phần SAM/Teacher có thật sự giúp?** | **Đang chạy** | E17 |
| SAGE trộn **phi tuyến** làm mất thông tin VIS tần trung (kết cấu, chữ, màu xe lớn)? | **Chưa kiểm** – gợi ý từ E21 (thua ở xe buýt/xe tải ban ngày, TB cũng hơn SAGE ở ca thua) và E15c (vệt ở vùng phẳng) | – |
| SAGE tối ưu chỉ số ghép (loss max-gradient, tương phản) chứ không tối ưu phân đoạn | **Chưa kiểm trực tiếp** – nhất quán với E14 | – |

## 3. Ràng buộc cho phương pháp cải tiến

- **Mục tiêu đã chốt (Mân):** SAGE cải tiến phải **hơn cả TB và VIS** theo đúng thiết lập B3:
  - mốc: TB **64,42**, VIS **62,55**, SAGE gốc 63,37 (273 ảnh, TB 3 seed) → cần tăng ≥ ~1 điểm so với SAGE gốc;
  - thành công khi hơn ở cả 3 seed **và** CI gộp > 0 (cùng cách bootstrap như B3);
  - kèm: không tụt ở nhóm ảnh sáng so với SAGE gốc.
- **Dev vs test:** 12 model B3 train trên 1.220 ảnh = train + Dev → **không** dùng chúng để chấm prototype trên Dev.
  Mỗi phương pháp ghép mới phải train SegFormer **mới** trên ảnh của chính nó. Giai đoạn chọn phiên bản: train
  SegFormer trên 1.045 ảnh train, chấm Dev (có thể dùng B0/ít epoch để xếp hạng rẻ). Kết quả cuối: train 1.220 ảnh,
  chấm test **một lần**.
- **Không chỉnh gì theo kết quả test.** Các E trong repo dùng test để **phân tích**, không để chọn tham số.
- **Máy:** Kaggle T4 x2 (~2,6 giờ/run SegFormer-B3, giới hạn 12 giờ/phiên, quota GPU theo tuần); máy Mân RTX 3050 6 GB
  (train Student SAGE ~17 phút).
- **Hai môn:** cần phần **xử lý ảnh** (CS406) và phần **học sâu** (CS431).

## 4. Các hướng ứng viên (chưa chốt)

| Hướng | Ý | Căn cứ | Rủi ro |
|---|---|---|---|
| **Fine-tune Student bằng loss phân đoạn** | Gắn SegFormer (đóng băng hoặc cùng train) sau Student, thêm loss CE để ảnh ghép tối ưu cho hiểu cảnh | B3: SAGE ≈ VIS, TB > SAGE → còn chỗ; SegMiF (cùng nhóm tác giả) train chung với phân đoạn được +4,3 so với VIS; trùng việc của Lộc bé | Dễ "học mẹo" cho đúng một SegFormer; phải chấm bằng SegFormer train lại riêng như B3 |
| **Hiểu rồi sửa cách trộn của SAGE** | Kiểm giả thuyết "trộn phi tuyến mất tần trung của VIS", rồi sửa đích của loss gradient / thêm nhánh giữ kết cấu VIS | E21, E15c, E14 | Chưa có bằng chứng trực tiếp; cần 1 thí nghiệm phân tích trước |
| **SAM có cần không** | Nếu E17 cho thấy SAM không giúp: thay Teacher bằng nguồn dạy rẻ hơn (mô hình phân đoạn nhẹ) | Bài: −4,7 khi bỏ SAM; chờ E17 | Phụ thuộc kết quả E17 |
| Thích nghi theo **vùng** (không theo ảnh) | Cổng IR/VIS theo điểm ảnh/vùng | E12: theo ảnh trần nhỏ | E13: SAGE đã chọn IR theo vùng → dư địa có thể nhỏ |

Đã rớt: tăng độ sắc (E1), đổi kích thước train (E9), luật theo độ sáng cấp ảnh (E12), khử nhiễu trước khi ghép (E15e).
