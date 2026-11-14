# Checkpoint SAGE công bố được train trên dữ liệu nào? – 08/11/2026

Checkpoint: `checkpoints/checkpoints.pt` của `RollingPlain/SAGE_IVIF` (Student, 135.617 tham số, md5
`241396a51a3c5e85e360a409d9e8fe4d`). Bài nói dùng 5 dataset (TNO, RoadScene, MFNet, FMB, M³FD) nhưng code chỉ có
loader FMB và không ghi checkpoint train trên gì.

Ý tưởng: lớp đầu của Student là conv tuyến tính trên (Y, IR) rồi BatchNorm. BN lưu trung bình và phương sai chạy của
dữ liệu lúc train → giải ngược ra độ sáng dữ liệu train, và so phương sai với từng dataset.

Script (chỉ cần numpy + OpenCV): `doc_ckpt.py` (đọc BN), `giai_nguoc.py` (giải ngược E[Y], E[IR]), `kiem_var.py` (so
phương sai trên crop 192×256).

| | E[Y] | E[IR] |
|---|---|---|
| Giải ngược từ checkpoint (R² = 0,9998) | 0,468 | 0,293 |
| FMB train (1.220 ảnh) | 0,465 | 0,309 |
| FMB test (280 ảnh) | 0,485 | 0,309 |
| M³FD (6 ảnh mẫu) | 0,443 | 0,334 |

| Dữ liệu | Lệch trung bình | Tỉ lệ phương sai (trung vị, gần 1 = khớp) |
|---|---|---|
| FMB train (200 ảnh) | 1,8% | **0,99** |
| FMB test (100 ảnh) | 2,8% | 1,19 |
| M³FD (6 ảnh mẫu) | 3,5% | 0,79 |

**Kết luận:** checkpoint **rất có thể được train trên FMB train** (cuối quá trình train).

**Giới hạn:** BN chỉ phản ánh vài chục batch cuối, không loại trừ trộn dataset khác trước đó; chỉ có 6 ảnh M³FD để
so; `num_batches_tracked` = 2.925 không chia tròn cho 305 (batch/epoch FMB với batch 4) – chưa giải thích được.

**Hệ quả:** M³FD là dữ liệu "mới" với checkpoint này (kiểm tra tổng quát hóa hợp lý). Kết quả trên FMB test vẫn là
test chính thức (7 ảnh rò đã gắn cờ). Thêm một kiểm tra liên quan (14/11): chạy checkpoint trên 6 ảnh mẫu của repo
SAGE ra gần trùng ảnh ghép tác giả để sẵn (lệch TB 0,2 mức xám, tương quan 0,9999) → bước tạo ảnh SAGE của nhóm đúng.
