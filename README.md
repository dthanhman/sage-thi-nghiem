# SAGE trên FMB – các thí nghiệm tìm khoảng trống (E0–E17)

Repo này gom **toàn bộ thí nghiệm phân tích** nhóm đã chạy quanh bài SAGE (CVPR 2025) trên dữ liệu FMB, để
người thiết kế phương pháp cải tiến (và trợ lý AI của họ) đọc một chỗ là hiểu đủ bối cảnh.

- Repo chính của nhóm (code, baseline đã khóa, manifest): `quan14333/xulianh`. Baseline B3 nằm ở
  `reports/experiments/b3_fmb_seg/` bên đó; bản sao báo cáo ở [`b3_baseline/`](b3_baseline/README.md).
- Repo này **chỉ để đọc và tra cứu**: script giữ nguyên như đã chạy (đường dẫn trỏ vào máy của Mân, dữ liệu
  FMB và checkpoint không có ở đây), kèm file kết quả dạng chữ và vài hình mẫu.

## Đọc theo thứ tự

1. [`BOI_CANH.md`](BOI_CANH.md) – SAGE là gì, FMB, cách chấm, kết quả baseline B3 (số đã khóa).
2. [`KET_QUA_E.md`](KET_QUA_E.md) – **từng thí nghiệm**: câu hỏi, cách làm, tiêu chí đặt trước, kết quả, kết luận,
   độ tin cậy.
3. [`CAU_HOI_MO.md`](CAU_HOI_MO.md) – điều đã loại trừ, điều còn mở, ràng buộc cho phương pháp cải tiến,
   các hướng ứng viên (**chưa chốt**).
4. [`LIEN_QUAN.md`](LIEN_QUAN.md) – bài liên quan: SegMiF (cùng nhóm tác giả, tạo ra FMB), các bài chấm lại SAGE.

## Nếu bạn là trợ lý AI đọc repo này

Nhiệm vụ dự kiến: đề xuất **câu hỏi nghiên cứu** và **hướng cải tiến SAGE** dựa trên bằng chứng ở đây. Quy tắc:

- **Chỉ dùng số có trong các file của repo.** Số của bài báo ghi là "theo bài".
- Phân biệt mức bằng chứng: **B3** (SegFormer-B3, 3 seed, đã khóa) > **B0** (SegFormer-B0, 1–2 seed, khám phá)
  > **proxy** (SegFormer Cityscapes zero-shot, không train lại). Kết luận chính phải dựa trên B3.
- Mỗi thí nghiệm có **tiêu chí đặt trước** (ghi trong docstring script hoặc README). Kết luận đọc theo tiêu chí đó,
  không tự đổi ngưỡng sau khi thấy số.
- `*` sau một hiệu = khoảng tin cậy 95% (bootstrap theo khối ảnh liền nhau, B = 2000) không chứa 0.
- Hướng cải tiến phải **hơn cả TB và VIS** theo đúng thiết lập B3 (xem `CAU_HOI_MO.md` mục 3), và phải chỉnh/
  chọn phiên bản trên **Dev**, không trên test.

## Cấu trúc

```text
README.md, BOI_CANH.md, KET_QUA_E.md, CAU_HOI_MO.md, LIEN_QUAN.md
b3_baseline/            báo cáo B3 đã khóa + bảng chênh lệch, tổng hợp, ca thắng/thua
thi_nghiem/
  b0_0811/              E0–E14, E20, E21 và so sánh tỉ lệ IR (SegFormer-B0 / proxy; seg/ = phần có train)
  b3_e15/               E15, E15b–E15e: làm hỏng VIS test, chấm bằng 12 model B3 (anh_mau/ = hình mẫu)
  e17/                  Student SAGE train lại không SAM (đang chạy)
```

Cập nhật: 14/11/2026. Người chạy thí nghiệm: Mân.
