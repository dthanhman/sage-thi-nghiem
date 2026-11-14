# E17 – SAM có làm ảnh ghép SAGE tốt hơn cho phân đoạn không?

Bài SAGE (Bảng 3): bỏ SAM thì mIoU FMB 61,2 → 56,5 (**−4,7**). Nhóm kiểm lại bằng đúng thiết lập B3.

## Cách làm

1. `train_student_nosam.py`: train lại Student SAGE (cùng kiến trúc, 0,136M tham số) trên 1.220 ảnh train FMB,
   **chỉ loss nội dung** (MSE + gradient, hệ số như code gốc), bỏ Teacher, SAM, loss tương phản, loss đặc trưng.
   Adam 2e-3, cosine, batch 4, crop 192×256, 100 epoch, chỉ cập nhật ở batch chẵn như code gốc. Seed 0. Train trên máy
   Mân (RTX 3050, ~22 phút) → `sage_student/nosam.pt`.
2. Kaggle notebook `thanhmn123/sage-e17-nosam-b3` (dataset `thanhmn123/fmb-ivif-sage` + `thanhmn123/sage-e17-nosam`):
   xuất ảnh `nosam` (Y = Student_nosam(Y_vis, IR), ghép CrCb của VIS) → SegFormer-B3 y hệt B3, seed 0, 1, 2.
3. So với 12 run B3 đã khóa bằng `so_sanh_b3.py` (thêm `nosam` vào danh sách biến thể), 273 ảnh, mIoU_co_mat.

## Tiêu chí đặt trước (ghi 14/11/2026, trước khi có kết quả)

Gọi d = mIoU(sage) − mIoU(nosam); CI95 bootstrap theo khối, gộp 3 seed (cùng bộ chỉ số bootstrap, như B3).

| Kết luận | Điều kiện |
|---|---|
| **SAM có ích** | d > 0 ở cả 3 seed **và** CI gộp > 0 |
| **Không thấy SAM có ích** | CI gộp chứa 0, hoặc d gộp ≤ 0 |
| **SAM làm hại** | d < 0 ở cả 3 seed **và** CI gộp < 0 |
| Lưng chừng | còn lại |

Kèm (không đổi kết luận trên):
- **Tái lập ablation của bài?** Có nếu d gộp ≥ +3 (bài +4,7); không nếu cận trên CI gộp < +4,7.
- nosam − tb và nosam − vis (Student không SAM có hơn trộn trung bình không).

Giới hạn: chỉ 1 lần train Student (seed 0) – SAGE cũng chỉ công bố 1 checkpoint; Student gốc còn được chưng cất từ Teacher
nên "bỏ SAM" ở đây = bỏ toàn bộ Teacher (SAM + SPA + X-Decoder), giống cách bài làm ablation "w/o SAM" hay không thì bài không ghi rõ.
