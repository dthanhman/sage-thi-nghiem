# Tìm gap – E0, E1, E2 (FMB test)

Thư mục ngoài repo, như `thu_viec3/`. Không train gì. Tiêu chí dưới đây ghi **trước khi chạy** (08/11/2026).

## Chạy

```
D:/Miniconda3/envs/DoAn/python.exe chay.py            # sinh biến thể + metric + SegFormer, ~?? phút
D:/Miniconda3/envs/DoAn/python.exe tong_hop.py        # bảng + bootstrap + kết luận theo tiêu chí
```
Thêm `--n 10` vào `chay.py` để chạy thử 10 ảnh.

## Dữ liệu và biến thể

- 280 ảnh FMB test theo `xulianh/data/manifests/test.csv`. Báo cáo cả 280 và **273** (bỏ 7 ảnh `leak`). Kết luận dựa trên 273.
- Biến thể (đều so trên cùng ảnh):
  - `vis`: chỉ VIS. `ir`: chỉ IR. `tb`: Y = (Y_vis + IR)/2.
  - `sage`: SAGE Student pretrained (`thu_viec3/SAGE_IVIF/checkpoints/checkpoints.pt`), FP32.
  - `sage_usm{a}`: unsharp mask trên Y của SAGE, Y + a·(Y − Gauss(Y, σ=1,5)), a = 0,25 / 0,5 / 1,0.
- Ảnh màu: Y ghép + CrCb của VIS (OpenCV YCrCb), làm tròn uint8 như output thật.

## Metric

- Ghép (E0, E1): code PR #6 của Lộc lớn (worktree `pr6_eval`, bản `origin/feature/loclon-evaluation`, chưa merge).
  EN, SD, SF, AG (không tham chiếu), SCD, MS-SSIM (tổng 2 nguồn, như bài), Q_ABF (có tham chiếu), NIQE, BRISQUE.
  Lưu ý: EN/SD/SF/AG tăng gần như chắc chắn khi làm sắc, nên **không dùng để kết luận E1**.
- Proxy mIoU (E2): SegFormer-B2 `nvidia/segformer-b2-finetuned-cityscapes-1024-1024`, **zero-shot**, không fine-tune.
  Ảnh vào 800×600 gốc, chuẩn hóa ImageNet. Map 19 lớp Cityscapes → FMB:
  road→1, sidewalk→2, building→3, traffic light→4, traffic sign→5, vegetation→6, terrain→6, sky→7,
  person→8, rider→8, car→9, truck→10, bus→11, motorcycle→12, bicycle→13, pole→14;
  wall, fence, train → "khác" (tính là đoán sai).
  Một ma trận nhầm lẫn trên cả tập, bỏ nhãn 0, mIoU trên các lớp có trong GT của tập (13 lớp, không có bicycle).
  **Đây là proxy, không phải mIoU chính thức** (SegFormer học trên RGB ban ngày).
- CI95: bootstrap theo **khối id liền nhau** (20 khối trên 280 ảnh), 2.000 lần, vì ảnh liền số là cùng cảnh.

## Tiêu chí đặt trước

**Kiểm tra hợp lệ của proxy.** mIoU proxy của `vis` trên 273 ảnh phải ≥ 20%. Nếu thấp hơn, proxy quá yếu, không kết luận E2 và phần ngữ nghĩa của E1.

**E0** chỉ ghi số làm mốc. Phải có 280/280 ảnh hợp lệ cho mọi biến thể.

**E1 – USM có bù được khoảng chênh "bỏ SAM" ở Bảng 3 không** (theo bài, bỏ SAM thì SCD +0,5%, MS-SSIM +0,9% so với SAGE).
Ở một mức a nào đó, so với `sage` (chênh tương đối, theo cặp):
1. ΔSCD ≥ +0,5% và cận dưới CI95 > 0;
2. ΔMS-SSIM ≥ +0,9% và cận dưới CI95 > 0;
3. ΔQ_ABF có cận dưới CI95 > 0;
4. ΔmIoU proxy ≥ −0,5 điểm.

- Đạt cả 4 ở cùng một mức a: **USM bù được, Gap A coi như chết**, vì adapter khó hơn một phép lọc 0 tham số.
- Không đạt: Gap A **chưa chết**. Ghi rõ metric nào USM không bù được, và đó là chỗ adapter có thể nhắm vào.

**E2 – SAGE có giúp ngữ nghĩa không (proxy)**, trên 273 ảnh:
- Δ(sage − vis) có CI95 > 0: SAGE giúp. CI chứa 0: không thấy lợi. CI < 0: SAGE kém VIS.
- Phụ: Δ(sage − tb) có CI95 > 0 thì phần "ngữ nghĩa SAM" có đóng góp so với ghép thô.
- "Không thấy lợi" hoặc "kém" → làm tiếp E3 (chia theo độ sáng). File kết quả đã có cột `y_mean` để chia.

## E3 – chia theo độ sáng (ghi trước khi tính, 08/11/2026 tối)

Kích hoạt vì E2 ra "không thấy khác". Dùng proxy mIoU như E2, 273 ảnh.
- Chia 3 nhóm bằng nhau theo `y_mean` (độ sáng Y trung bình của VIS): tối / vừa / sáng (theo tam phân vị).
- So Δ(sage − vis) và Δ(sage − tb) trong từng nhóm, CI95 bootstrap theo khối id trong nhóm.
- **Đạt (có gap "ghép thích nghi theo điều kiện")**: nhóm tối Δ(sage − vis) cận dưới CI > 0 **và** nhóm sáng Δ(sage − vis) cận trên CI < 0.
- **Đạt một phần**: nhóm tối CI > 0, nhóm sáng CI chứa 0 → ghép có lợi lúc tối, không hại lúc sáng (gap yếu hơn).
- **Trượt**: còn lại.
- Sẽ làm lại bằng mIoU thật khi SegFormer train xong.

## E5 – mức trần của "chọn ảnh thích nghi" (ghi trước khi tính, 08/11/2026 tối)

Proxy mIoU, 273 ảnh. Oracle: với mỗi ảnh chọn biến thể có mIoU từng ảnh cao hơn (giữa vis và sage; phụ: giữa vis, tb, sage), rồi cộng ma trận nhầm lẫn các ảnh đã chọn.
- Lợi ích = mIoU(oracle) − max(mIoU vis, mIoU sage). CI95 bootstrap theo khối.
- **≥ +3 điểm**: chọn thích nghi có tiềm năng, đáng làm. **< +1,5 điểm**: bỏ hướng "ghép thích nghi". Ở giữa: yếu.
- Lưu ý: oracle chọn theo chính nhãn test nên **thiên cao** (ăn cả nhiễu). Đây là mức trần, không phải kết quả đạt được.

## E6 – tính lại BatchNorm của Student trên dữ liệu mới (AdaBN), M3FD 300 cặp

- Chia M3FD theo tên ảnh đã sắp xếp: nửa đầu (150) để ước lượng lại thống kê BN, nửa sau (150) để chấm. Không train trọng số nào.
- So SAGE gốc với SAGE + AdaBN trên 150 ảnh chấm, metric PR #6: SCD, MS-SSIM tổng, Q_ABF (kèm EN, SD, NIQE để tham khảo).
- **Đạt**: ΔSCD và ΔQ_ABF có CI95 > 0, và ΔMS-SSIM có cận trên CI ≥ 0 (không giảm chắc chắn).
- Kiểm tra hợp lệ: làm y hệt trên FMB (ước lượng trên dev, chấm trên 273 ảnh test). Vì checkpoint vốn hợp FMB, kỳ vọng thay đổi nhỏ (|Δ| < 1% mọi metric). Nếu FMB cũng đổi nhiều thì AdaBN đang làm gì đó khác, không kết luận.

## E8 – SAGE có tự cân bằng IR/VIS theo độ sáng không (ghi trước khi tính)

Lý do: CtrlFuse chê SAGE "quá nghiêng về IR ở cảnh ngày", P2Fusion chê "quá nghiêng về VIS". Dùng kq_metric.csv (273 ảnh).
- Độ nghiêng của ảnh ghép: L = MS_SSIM_VIS − MS_SSIM_IR (dương = giống VIS hơn).
- Tương quan Spearman ρ giữa độ sáng VIS (y_mean) và L của `sage`. So với `tb` (trọng số cố định 0,5/0,5) làm mốc.
- **SAGE thích nghi**: ρ(sage) − ρ(tb) ≥ +0,2 (ảnh càng sáng càng nghiêng về VIS rõ hơn ghép cố định). **Không thích nghi**: |ρ(sage) − ρ(tb)| < 0,1. Ở giữa: yếu.
- Phụ: L của sage theo 3 nhóm độ sáng (như E3).
