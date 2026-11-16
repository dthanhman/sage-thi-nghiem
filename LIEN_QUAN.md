# Bài liên quan

## SegMiF (ICCV 2023) – cùng nhóm tác giả với SAGE, nơi tạo ra FMB

*Multi-interactive Feature Learning and a Full-time Multi-modality Benchmark for Image Fusion and Segmentation*
(arXiv 2308.02097). Tác giả: Jinyuan Liu, Zhu Liu, **Guanyao Wu**, Long Ma, **Risheng Liu**, Wei Zhong, Zhongxuan Luo,
**Xin Fan** (ĐH Công nghệ Đại Liên). In đậm = cũng là tác giả SAGE.

- **Mục đích:** train **chung** mạng ghép ảnh và SegFormer (cascade, attention HIA giữa hai nhánh, trọng số động giữa
  2 loss) để vừa ra ảnh ghép đẹp vừa phân đoạn đúng. FMB là đóng góp thứ 4 (1.500 cặp 800×600, 14 lớp, 98,16% pixel
  có nhãn).
- **Loss gradient** của SegMiF cũng là ‖∇ᵏu − max(∇ᵏx, ∇ᵏy)‖ – cùng ý với SAGE.
- **Cấu hình chấm phân đoạn (theo bài, dùng cho cả các phương pháp ghép khác):** SegFormer (bài không ghi bản; code MFNet
  dùng MiT-B3), phóng/thu 0,5–2,0, crop 360×360, **đổi độ sáng ngẫu nhiên**, Adam poly, khởi động 3k vòng ở 1e-6 rồi lr
  8e-5, batch 8, **8 vòng × 10k = 80k vòng lặp**, V100. Không ghi cách tính mIoU.
- **Bảng 2 (FMB, theo bài):** VIS 50,5 / IR 43,9 / SegMiF 54,8 → **+4,3** so với VIS.
- Nhận xét của nhóm: VIS/IR của SAGE (49,1 / 42,7) rất gần SegMiF → có thể cùng kiểu chấm (mức ~50 khớp kiểu tính có
  unlabeled, lớp vắng = 0; theo kiểu đó VIS của nhóm là 53,7). SAGE ghi 100 epoch, khác 80k vòng của SegMiF.

| | SegMiF (bài) | B3 nhóm |
|---|---|---|
| Vòng lặp | 80k | 15,2k (100 epoch) |
| Crop | 360 | 512 |
| Đổi độ sáng | Có | Không |
| Lật ngang | Không nhắc | Có |
| Optimizer / lr | Adam / 8e-5 | AdamW / 6e-5 (đầu ra ×10) |

## Các bài chấm lại SAGE trên FMB (số "theo bài", tra đến 11/2026)

| Bài | Mạng chấm | mIoU SAGE (theo bài) | mIoU phương pháp của bài | Bài đó − SAGE |
|---|---|---|---|---|
| SSVIF (arXiv 2509.22450, 2025) | SegFormer-B3 | 62,04 | 62,07 | +0,03 |
| AD-SRF (arXiv 2610.12897, 2026) | OverLock-B | 61,39 | 63,87 | +2,5 |
| P2Fusion (arXiv 2608.13045, 2026) | SegFormer-B5 | 56,48 | 57,46 | +1,0 |
| PCDFusion (NeurIPS 2026, arXiv 2610.23663) | SegFormer-B2 | 56,33 | 60,05 | +3,7 |
| CLDyN (CVPR 2026, arXiv 2604.08924) | SegFormer-B2 | 54,89 \* | 60,34 | +5,5 |

- ⚠ **Sửa 16/11/2026:** bản trước ghi cột cuối là "SAGE hơn VIS" – **sai**. Đối chiếu bảng gốc từng bài (SSVIF Bảng I, AD-SRF Bảng III, P2Fusion Bảng 2, PCDFusion Bảng 2 và 14, CLDyN Bảng 2): cột đó là *phương pháp của bài − SAGE*. **Không bài nào trong 5 bài có dòng "chỉ dùng VIS" trên FMB.**
- → **Chưa bài nào kiểm lại mức SAGE − VIS = +12,1** (Bảng 1 bài SAGE: VIS 49,1 → SAGE 61,2). Số đối chiếu duy nhất hiện có là của nhóm: **+0,8** (SegFormer-B3, 3 seed).
- Cùng là SAGE trên FMB nhưng mIoU dao động **54,9 → 62,0** tùy cấu hình đánh giá (mạng, số lớp, cách train) → cấu hình phải ghi đủ.
- Trong bảng của các bài, SAGE chỉ ngang các cách ghép khác: SSVIF đặt SAGE 62,04 cạnh SegMiF 62,14 và CDDFuse 61,64 (không dùng nhãn); CLDyN xếp SAGE thấp nhất trong 11 phương pháp.
- \* CLDyN: bảng không ghi rõ dataset của cột phân đoạn (bài chỉ nói có đánh giá trên FMB).
- Lời chê SAGE trong các bài trên (lệch IR/VIS, mờ chi tiết) mâu thuẫn nhau và chỉ định tính; E10 bác "mờ biên".
- PCDFusion tự nhận yếu ở vùng tối / IR không tin cậy.

## Khác

- **MRFS (CVPR 2024)** – có code chấm FMB công khai: 15 lớp, background là lớp 0 và được train; mIoU = nanmean 15 lớp;
  AdamW 6e-5, batch 6, crop 480×640, scale 0,5–1,75, MiT-B4.
- **SEA (arXiv 2410.06811)** – chấm ảnh ghép bằng phân đoạn zero-shot (X-Decoder/SEEM/Grounded-SAM) trên FMB; không có
  SAGE, không có trộn TB, không chia theo độ sáng.
- **Ghép theo độ sáng:** PIAFusion (2022), AMFusion (ACM MM 2024), IAIFNet, DIVFusion – đã làm cân bằng IR/VIS theo
  độ sáng, nhưng đo bằng chỉ số ghép hoặc phát hiện, chưa đo mIoU theo nhóm sáng.
