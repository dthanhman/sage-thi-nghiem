# Kết quả từng thí nghiệm

Mức bằng chứng: **B3** = SegFormer-B3 đã khóa, 3 seed (mạnh nhất) · **B0** = SegFormer-B0, 1–2 seed (khám phá) ·
**proxy** = SegFormer-B2 Cityscapes zero-shot, không train lại (yếu) · **ghép** = chỉ đo trên ảnh ghép, không cần
model phân đoạn. Tất cả trên **273 ảnh test sạch**. `*` = CI95 không chứa 0.

## Bảng tóm tắt

| E | Câu hỏi | Mức | Kết luận theo tiêu chí đặt trước |
|---|---|---|---|
| E0 | Bảng nền chỉ số ghép | ghép + proxy | Mốc (SAGE đứng đầu SCD, gần đầu Q_ABF) |
| E1 | Làm sắc ảnh SAGE (USM) có bù được "bỏ SAM" không? | ghép + proxy | **Không** – làm sắc còn hạ ngữ nghĩa |
| E2 | SAGE có giúp phân đoạn hơn VIS, TB không? | proxy | Không thấy khác |
| E3 | Lợi ích SAGE có tập trung ở ảnh tối không? | proxy → B0 | Proxy trượt; B0 thật: ảnh tối +4,5\* (B3 sau đó: +2,5, mất ý nghĩa) |
| E5 | Chọn đầu vào tốt nhất cho từng ảnh được bao nhiêu? | proxy | **Yếu** (+1,9) |
| E6 | AdaBN cho Student trên M³FD có lợi không? | ghép | Không kết luận |
| E8 | SAGE có tự đổi mức dựa vào VIS theo độ sáng hơn TB không? | ghép | **Không thích nghi hơn TB** |
| E9 | SAGE kém vì train crop nhỏ (192×256)? | ghép | **Không** |
| E10 | SAGE có làm mờ biên vật thể không? | B0 | **Không** (biên còn tốt hơn) |
| E11 | SAGE thắng/thua ở ảnh nào? | B0 | Thắng ở tối/khói, thua khi IR tương phản cao |
| E12 | Trần của chọn đầu vào thích nghi theo ảnh? | B0 | **Trần nhỏ** (+1,0…+1,6 so với TB); luật theo độ sáng kém TB |
| IR 30/70 | Trộn IR cố định 30%/70% so với SAGE | B0 | ir70 ≈ TB ≥ SAGE ≈ ir30 |
| E13 | SAGE lấy IR ở vùng nào? | ghép | Người ≈ toàn IR, nền chủ yếu VIS; tối lấy IR nhiều hơn |
| E14 | Chỉ số ghép có xếp hạng giống mIoU không? | B0 + B3 | **Chỉ MS-SSIM**; SCD, Q_ABF, EN, SD, SF, AG không |
| E20 | SAGE thua TB vì lấy IR ở nền quá ít? | B0 + B3 | **Không ủng hộ** (B0 ngược chiều, B3 bác) |
| E21 | Ca thắng/thua SAGE so với VIS | B3 | Thắng nhờ lớp người; thua ban ngày, xe lớn |
| E15 | VIS càng hỏng (tối/sương) thì ghép càng có ích? SAGE hơn TB ở đó? | B3 | Sương: SAGE ≫ VIS. **Tối: TB > SAGE\* ở mọi mức** |
| E15b | SAGE có tăng IR khi VIS tối không? | ghép | **Có** (w IR 0,22 → 0,74) |
| E15c | SAGE thua khi tối do model phân đoạn lạ ảnh? | B3 | **Không** – ảnh SAGE tối thật sự kém (−2,4\*) |
| E15d | SAGE có giữ nhiễu/vệt của VIS tối? | ghép | **Có** – nhiễu gấp ~1,8 lần TB, vệt 2,2–2,5 lần |
| E15e | Khử nhiễu VIS trước khi ghép có thu hẹp TB − SAGE? | B3 | **Bác** – nhiễu không phải nguyên nhân chính |
| E17 | SAM có làm ảnh ghép tốt hơn cho phân đoạn? | B3 | **Đang chạy** |

---

## Nhóm 1 – SAGE có giúp phân đoạn không (B3 là câu trả lời chính)

**B3** (xem `BOI_CANH.md` mục 6): SAGE − VIS +0,82 [−0,86; +3,04]; **TB − SAGE +1,05\*** [+0,08; +2,16].

**E2 (proxy)** – `thi_nghiem/b0_0811/ket_qua.md`. SegFormer Cityscapes zero-shot: SAGE 47,0 ≈ VIS 46,2 ≈ TB 47,4;
SAGE − VIS +0,79 [−1,68; +2,92]. IR giúp lớp người (VIS 37,6 → SAGE 49,6) nhưng IR mù biển báo.

**E3** – `e3.py`, `e3.txt` (proxy) và `seg/so_sanh.txt` (B0). Chia 3 nhóm độ sáng (91 ảnh mỗi nhóm).
Proxy: tối +3,1 [−0,4; +6,4], sáng −1,8\* → trượt tiêu chí. B0 thật (seed 0): tối +4,47 [+0,38; +9,08]\*,
vừa +2,1, sáng −0,3. **B3 3 seed:** tối +2,5 [−2,9; +8,1] – mất ý nghĩa khi mạng chấm mạnh hơn.

**E21** – `seg/e21_ca_loi.py`, `b3_baseline/ca_thang_thua.txt`, hình `b3_baseline/sage_{thang,thua}.png`.
Theo từng ảnh (3 seed): thắng 159 / thua 51 / hòa 63; ảnh tối 72/7. Trong ảnh chênh > 3 điểm, lớp quyết định là
**người** ở 68 ảnh SAGE thắng. Ví dụ thắng: 00089 (khói; IoU người 0 → 85). Thua: 00388 (xe buýt → xe con), 00167
(ban ngày, mất đèn giao thông). Ở ca thua, TB cũng thường tốt hơn SAGE.

## Nhóm 2 – Các lời chê/giả thuyết về chất lượng ảnh SAGE

**E1 (USM)** – `chay.py`, `tong_hop.py`, `ket_qua.md`. Làm sắc Y của SAGE với a = 0,25/0,5/1,0. Không mức nào
đạt 4 tiêu chí; mIoU proxy giảm (a = 0,25: −0,55\*; a = 1,0: −3,30\*). → Ảnh "sắc hơn" không giúp ngữ nghĩa.

**E9** – `e9_kich_thuoc.py`, `e9.txt`. Chạy Student theo ô 192×256 hoặc đổi tỉ lệ ảnh: đều kém chạy nguyên ảnh
(ô: SCD −0,66\*, MS-SSIM −1,08\*, Q_ABF −1,11\*). Student được train crop **đúng tỉ lệ gốc** → giả thuyết "train
crop nhỏ làm hỏng" sai từ gốc.

**E10** – `seg/e10_bien.py`, `seg/e10.txt` (B0). mIoU ở dải biên 2 px: SAGE − VIS +1,94\*; 5 px +2,60\*; biên −
trong ≈ 0 → SAGE **không** làm mờ cạnh (lời chê của một số bài không đúng cho phân đoạn).

**E14** – `seg/e14_metric_vs_miou.py`, `seg/e14.txt`, `seg/kq_e14_metric.csv`. Kendall τ giữa xếp hạng theo chỉ số
ghép và theo mIoU (B0: 6 biến thể, B3: 4 biến thể). **Chỉ MS-SSIM (tổng) đáng tin** (τ 0,87 / 0,91). SCD không
(τ 0,07 / 0,33; SCD bất biến với tỉ lệ trộn tuyến tính: ir30 = tb = ir70 = 1,385). SF, AG xếp **ngược** mIoU (τ −1,0
ở B0). SAGE đứng 1 về SCD, 2 về Q_ABF nhưng chỉ hạng 4/6 mIoU B0. → Bảng chỉ số ghép của bài không chứng minh được
lợi ích hiểu cảnh.

**E6** – `e6.py`, `e6.txt`. Ước lượng lại BatchNorm của Student trên M³FD: đổi chỉ số ±1–5% nhưng kiểm tra hợp lệ
trên FMB cũng đổi (Q_ABF +2,5%, SD −5%) → không kết luận. Ghi chú: thống kê BN lớp đầu của checkpoint khớp FMB train
tốt nhất → checkpoint nhiều khả năng được train trên FMB.

## Nhóm 3 – SAGE dùng IR thế nào, thích nghi theo cảnh?

**E8** – `e8.txt`. Tương quan giữa độ sáng ảnh và mức ảnh ghép dựa vào VIS: SAGE ρ 0,648, TB ρ 0,654;
ρ(SAGE) − ρ(TB) = −0,005 [−0,15; +0,16] → **không thích nghi hơn TB** ở cấp ảnh.

**E13** – `seg/e13_vung_ir.py`, `seg/e13.txt`. Ước lượng w = tỉ lệ IR hiệu dụng (Y_ghép − Y_vis ≈ w·(IR − Y_vis))
theo vùng nhãn. SAGE cả ảnh w = 0,22 [0,19; 0,29] (dùng IR ít); **người ≈ 0,99** (gần như chép IR); cây 0,12, trời
0,15, cột 0,08, nhà 0,22; mặt đường/xe R² thấp (không tin). Ảnh tối w người cao hơn ảnh sáng +0,22\*; cả ảnh tối 0,56
so với sáng 0,17. → SAGE **đã** tự chọn IR theo vùng và theo độ sáng.

**E5 / E12** – `e5.py`, `e5.txt` (proxy); `seg/e12_thich_nghi.py`, `seg/e12.txt` (B0, 2 seed). Chọn loại ảnh tốt
nhất cho **từng ảnh** (oracle, lạc quan): {vis, sage} +1,87 so với tốt nhất đơn lẻ (proxy, YẾU); {vis, tb} +0,96\* /
+1,10\* so với TB; {vis, sage, tb} +1,21\* / +1,61\*. Luật "tối dùng TB, sáng dùng VIS" **kém TB** (−1,33\* / −0,89).
→ IR giúp cả ảnh sáng; thích nghi ở cấp ảnh có trần nhỏ.

**Tỉ lệ IR cố định** – `seg/so_sanh_ir.py`, `seg/so_sanh_ir.txt` (B0, seed 0). VIS 51,8 / ir30 53,7 / TB 54,3 /
ir70 54,8 / SAGE 53,8. ir70 − SAGE +1,05 [−1,3; +2,5]; SAGE − ir30 +0,11. Ảnh tối: TB − ir30 +2,18\*. → Trộn IR
nhiều hơn không hại; SAGE ngang trộn tuyến tính 30% IR.

**E20** – `seg/e20_nen_ir.py`, `seg/e20.txt`. Giả thuyết "SAGE thua TB vì lấy IR ở nền quá ít": tương quan theo ảnh
giữa w_nền của SAGE và d = TB − SAGE lại **dương** (B0 ρ +0,29\*, B3 +0,19 CI chứa 0) → ảnh SAGE lấy nhiều IR ở nền
hơn lại là ảnh TB thắng nhiều hơn. **Không ủng hộ.** (w_nền hẹp, 0,10–0,23.)

**E11** – `seg/e11_loi.py`, `seg/e11.txt` (B0, seed 0, khám phá). Theo ảnh, SAGE − VIS trung vị +1,62, thắng 191/273.
d tương quan âm với độ sáng VIS (ρ −0,34) và với độ tương phản IR (ρ −0,48): SAGE lợi nhất khi VIS tối và IR không quá
tương phản. Lớp người ở ảnh tối: SAGE − VIS +17 đến +24 điểm IoU theo cỡ người.

## Nhóm 4 – Vì sao TB ≥ SAGE? (chuỗi E15, B3 seed 0–1 hoặc 0–2)

Làm hỏng **VIS test** (IR giữ nguyên), ghép lại, chấm bằng model B3 đã train trên ảnh sạch (không train lại).
`toi1–3`: tối tuyến tính ×0,30/0,12/0,05 + nhiễu Poisson (2.000 photon) + Gauss 0,002. `suong1–3`: x·t + 0,85(1 − t),
t = 0,60/0,35/0,15. Nhiễu cố định theo id ảnh.

**E15** – `thi_nghiem/b3_e15/e15_lam_hong_vis.py`, `e15.txt`, `kq_e15_*.csv`. mIoU TB 2 seed (VIS/IR/SAGE/TB):
sạch 62,3/55,7/62,9/64,1; **toi3 50,7/55,7/52,1/54,6**; **suong3 47,0/55,7/58,5/57,8**.
Tiêu chí: sương – SAGE ≫ VIS (+11,4, ĐẠT). Tối – SAGE − VIS không tăng; **TB − SAGE +2,4\* / +2,8\* (GAP, có ý
nghĩa ở mọi mức tối, cả 2 seed)**; IR − SAGE +3,0\* / +4,2\*. Giới hạn: hỏng tổng hợp; model học trên ảnh sạch.

**E15b** – `e15b_w_ir.py`, `e15b.txt`. w IR của SAGE: sạch 0,22 → toi1 0,68 / toi2 0,78 / toi3 0,74; Δ toi3 +0,52
[+0,40; +0,61], chủ yếu ở nền (0,20 → 0,74); người ~0,9 giữ nguyên. → Giả thuyết "SAGE không chuyển sang IR khi VIS
tối" **bị bác**.

**E15c** – `e15c_cheo.py`, `e15c.txt`, hình `anh_mau/luoi_{00089,00518}.jpg`. Chấm chéo ảnh {SAGE, TB} × model
{vis, ir, sage, tb} (seed 0). Trên ảnh SAGE toi3, model "sage" vẫn tốt nhất (51,7) → **không phải lệch phân phối**.
Ảnh SAGE toi3 kém ảnh TB −2,42 [−4,30; −0,69]\*; ngay trên model "sage", ảnh TB toi3 53,4 > ảnh SAGE 51,7. Nhìn hình:
SAGE toi3 có vệt/loang ở vùng phẳng (tường), TB mượt.

**E15d** – `e15d_nhieu.py`, `e15d.txt`, `kq_e15d_anh.csv`, hình `anh_mau/e15d_*.jpg`. Ở vùng phẳng (std cục bộ 9×9
của VIS sạch và IR < 2,5), đo phương sai cao tần thêm vào so với mức sạch. R = sqrt(nhiễu thêm SAGE / nhiễu thêm TB):
**toi3 1,79 [1,76; 1,82]**, toi1 1,67, toi2 1,78 (tham chiếu tuyến tính VIS/TB = 2,0) → SAGE giữ **gần hết** nhiễu VIS
dù w IR 0,74 (w đo tần thấp; tần cao vẫn lấy từ VIS). Dải vệt (σ 1–4): R 2,2–2,5 > 2,0 → SAGE **khuếch đại** vệt.
Theo ảnh, R không tương quan với TB − SAGE (ρ −0,07 [−0,29; +0,17]). Nghi cơ chế: loss gradient max(|∇VIS|, |∇IR|)
giữ nhiễu VIS như chi tiết.

**E15e** – `e15e_khu_nhieu.py`, `e15e.txt`, `e15e_h.json`, `e15e_psnr.json`. Khử nhiễu VIS tối bằng NLM (độ mạnh h chọn
trên 30 ảnh **train** theo MSE so với ảnh tối không nhiễu: toi3 h = 13, toi2 h = 5), đưa cùng ảnh đã khử vào cả SAGE
và TB, chấm B3 3 seed.

| toi3, gộp 3 seed | SAGE | TB | TB − SAGE |
|---|---|---|---|
| Không khử nhiễu | 52,19 | 54,84 | +2,66\* [+1,36; +4,02] |
| Đã khử nhiễu | 48,36 | 50,69 | +2,33\* [+1,20; +3,76] |
| Thu hẹp Δ | | | **+0,33 [−1,03; +1,34]** (seed 2: −0,83) |

toi2: Δ −0,08. → **BÁC**: khử nhiễu không thu hẹp khoảng cách; nhiễu (dù có thật, E15d) không phải nguyên nhân chính.
Khử nhiễu làm cả hai tụt ~4 điểm (model học ảnh sạch, NLM xóa chi tiết) – như nhau cho hai bên nên không đổi kết luận.

## Nhóm 5 – Vai trò của SAM

**E17** – `thi_nghiem/e17/`. Train lại Student (cùng kiến trúc) **chỉ với loss nội dung** (MSE + gradient), bỏ
Teacher/SAM, trên 1.220 ảnh train; rồi chấm y hệt B3, 3 seed. Tiêu chí đặt trước trong `e17/README.md`
(d = SAGE − nosam; SAM có ích nếu d > 0 cả 3 seed và CI gộp > 0). Bài tuyên bố −4,7. **Đang chạy.**

## Ý tưởng đã nêu nhưng chưa làm

- E16: giảm dữ liệu train phân đoạn còn 25/50% – lợi ích của ghép có lớn hơn khi mạng chấm yếu?
- E18: cận trên – SegFormer nhận 4 kênh RGB + IR trực tiếp (không ghép).
- X-Decoder zero-shot (mạng phân đoạn bài dùng trong Teacher) để chấm ảnh ghép.
