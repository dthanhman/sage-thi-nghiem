# Bối cảnh

## 1. Đồ án

- Đề tài chung cho 2 môn: **CS406** (Xử lý ảnh) và **CS431** (Học sâu). Nhóm 4 người: Quân, Mân, Lộc bé, Lộc lớn.
- Bài gốc được chọn: **SAGE** – *Every SAM Drop Counts: Embracing Semantic Priors for Multi-Modality Image Fusion
  and Beyond* (CVPR 2025, arXiv 2503.01210, code `RollingPlain/SAGE_IVIF`). Tác giả: Guanyao Wu, Haoyu Liu,
  Hongming Fu, Yichuan Peng, Jinyuan Liu, Xin Fan, Risheng Liu (ĐH Công nghệ Đại Liên).
- Phân công hiện tại:

| Người | Việc |
|---|---|
| Mân | Hoàn thiện và khóa baseline SegFormer-B3 trên FMB (**đã xong**) + các thí nghiệm trong repo này |
| Quân | Nghiên cứu và thiết kế phương pháp cải tiến SAGE: giả thuyết, prototype, ablation trên Dev |
| Lộc bé | Code phương pháp mới, baseline fine-tuning SAGE |
| Lộc lớn | SAGE trên M³FD (300 ảnh ghép) + chuẩn hóa code chỉ số ghép (EN, SD, SCD, MS-SSIM) |

## 2. SAGE làm gì

- **Bài toán:** ghép ảnh hồng ngoại (IR, 1 kênh) và ảnh thường (VIS, RGB) đã căn chỉnh thành một ảnh. Không có ảnh
  ghép "chuẩn" làm nhãn.
- **Teacher** (chỉ lúc train): dùng SAM (ViT-B) + module SPA + X-Decoder để ghép "có ngữ nghĩa".
- **Student** (lúc chạy): mạng dense block 0,136M tham số, chưng cất từ Teacher, không cần SAM. Nhận Y của VIS + IR,
  ra Y ghép; màu lấy Cr/Cb của VIS. Tác giả chỉ công bố checkpoint Student (`checkpoints/checkpoints.pt`).
- **Loss Student** (code): MSE với VIS và IR + **loss gradient L1(|∇F|, max(|∇VIS|, |∇IR|))** (hệ số 9) +
  loss đặc trưng (cosine) + loss tương phản trong không gian SAM.
- **Bài tuyên bố** (Bảng 1, FMB, SegFormer-B3): VIS 49,1 / IR 42,7 / **SAGE 61,2 → SAGE hơn VIS +12,1**.
  Bảng 3 (ablation): bỏ SAM thì mIoU FMB 61,2 → 56,5.

## 3. Dữ liệu FMB

- 1.500 cặp IR–VIS 800×600 + nhãn phân đoạn: **1.220 train + 280 test** (split chính thức). 14 lớp + nhãn 0
  (unlabeled). Lớp bicycle có 0 ảnh test → mIoU tính trên 13 lớp có mặt.
- Ảnh cắt từ video → **chia theo khối id liền nhau**, không chia ngẫu nhiên. Dev = 175 ảnh tách từ train theo khối
  (train còn 1.045).
- **Rò dữ liệu:** 7/280 ảnh test là khung liền kề ảnh train (cột `leak` trong manifest) → mọi kết luận dùng
  **273 ảnh sạch**.
- Ảnh khó: đêm, sương, khói (vd test 00089: khói che người ở VIS, IR vẫn thấy), đèn chói.

## 4. Các loại ảnh đầu vào so sánh

| Tên | Là gì |
|---|---|
| `vis` | Chỉ ảnh VIS |
| `ir` | Chỉ ảnh IR (nhân 3 kênh) |
| `tb` | **Trộn trung bình**: Y = (Y_vis + IR)/2, màu của VIS. Không học, không SAM – đối chứng đơn giản (bài không có) |
| `sage` | Student SAGE pretrained (FP32), Y ghép + màu VIS |
| `clahe`, `ir30`, `ir70`, `sage_usm*` | Chỉ có ở B0: tăng tương phản VIS; trộn IR 30%/70%; làm sắc ảnh SAGE |

## 5. Cách chấm

- **Chấm bằng phân đoạn** như Bảng 1 của bài: mỗi loại ảnh train **một SegFormer riêng** (cùng cấu hình), chấm mIoU
  trên test. Chênh lệch giữa các dòng chỉ đến từ ảnh đầu vào.
- **mIoU chính:** một ma trận nhầm lẫn trên cả tập, bỏ pixel nhãn 0, trung bình 13 lớp có mặt. Kèm các quy ước khác
  (kiểu SegMiF, MRFS, 7 lớp bài in) – kết luận không đổi.
- **Khoảng tin cậy 95%:** bootstrap theo khối id liền nhau, B = 2000. Gộp seed: cùng bộ chỉ số bootstrap cho mọi seed,
  lấy TB chênh lệch.
- **Chỉ số ghép** (không cần nhãn): EN, SD, SF, AG, SCD, MS-SSIM (tổng với 2 ảnh nguồn, tối đa 2), Q_ABF, NIQE.
  E14 cho thấy chỉ MS-SSIM xếp hạng khớp mIoU.

## 6. Baseline B3 (đã khóa 14/11/2026 – mốc chính)

SegFormer-B3 (khởi tạo `nvidia/mit-b3`), 15 lớp khi train, 100 epoch (15.200 vòng lặp), batch 8, AdamW 6e-5,
crop 512, train trên đủ 1.220 ảnh, lấy checkpoint cuối, FP16 hỗn hợp, Kaggle T4. Seed 0, 1, 2.
Chi tiết: [`b3_baseline/README.md`](b3_baseline/README.md).

**mIoU, 273 ảnh, TB ± SD 3 seed:**

| VIS | IR | SAGE | **TB** |
|---|---|---|---|
| 62,55 ± 0,48 | 56,41 ± 1,64 | 63,37 ± 0,94 | **64,42 ± 0,60** |

**Chênh lệch gộp 3 seed:**

| | Hiệu | CI95 |
|---|---|---|
| SAGE − VIS | +0,82 | [−0,86; +3,04] |
| TB − VIS | +1,87 | [−0,01; +4,19] |
| **TB − SAGE** | **+1,05\*** | [+0,08; +2,16] – TB > SAGE ở cả 3 seed |
| SAGE − IR | +6,96\* | [+3,58; +10,46] |

- Ảnh tối (1/3 tối nhất, 91 ảnh): SAGE − VIS +2,5 [−2,9; +8,1]; TB − SAGE +3,4 [−1,6; +4,5] – lớn nhưng chưa
  có ý nghĩa.
- Lớp người: VIS 59,0 → SAGE 69,9 ≈ TB 69,7 ≈ IR 70,9 → cái lợi đến từ ảnh nhiệt, không riêng SAGE.
- Theo từng ảnh: SAGE thắng VIS 159 ảnh, thua 51, hòa 63; thắng gần như nhờ lớp người (đêm, khói, đèn chói); thua ở
  ban ngày, xe lớn (xe buýt/xe tải bị đoán thành xe con).
- **Không tái lập được +12,1 của bài.** Mọi bài chấm lại SAGE khác cũng chỉ được 0 đến +5,5 (xem `LIEN_QUAN.md`).

## 7. B0 (khám phá, trước B3)

SegFormer-B0, 2 seed, 273 ảnh: VIS 51,6 / CLAHE 51,4 / SAGE 53,3 / TB 54,1 (TB 2 seed). Ảnh tối: VIS 45,7 /
SAGE 50,3 / TB 51,5. Lớp người: VIS 50,6 / SAGE 63,6 / TB 65,4. Cùng chiều với B3, mức tăng của SAGE lớn hơn một
chút (+1,7) khi mạng chấm yếu hơn.
