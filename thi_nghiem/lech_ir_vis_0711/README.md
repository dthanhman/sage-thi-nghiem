# Lệch IR–VIS (problem A) – 07/11/2026, **đã bác bỏ**

Câu hỏi: khi IR lệch so với VIS, SAGE Student có bị hại **riêng** (nhiều hơn ghép trung bình TB) không? Căn lại
đơn giản (phase correlation, ECC) có đáng dùng không?

Bằng chứng ban đầu (audit FMB): trên 150 cặp tin cậy cao, lệch trung vị 3,2 px, P90 6,1 px; lệch lớn chỉ ở ảnh
tối/sương (đo hỏng).

Cách làm: dịch IR 0/2/5/10 px (cả căn lại bằng phase/ECC), ghép bằng SAGE và TB, chấm bằng chỉ số ghép (code PR #6
của Lộc lớn: MS-SSIM tổng, SCD…), CI theo khối ảnh. Thẻ đầy đủ (mục tiêu chí, lịch sử 3 bản): [`problem_card_v3.md`](problem_card_v3.md).

**Kết luận (bản 3):** IR lệch 5–10 px thì SAGE có giảm điểm, nhưng giảm **không nhiều hơn** TB. Kiểm lại trên 5 khối
train lớn nhất (635 ảnh): E(5) MS-SSIM +0,53, CI95 [+0,35; +0,82] → đạt ngưỡng bác bỏ đặt trước; SCD cùng chiều. Mức
giảm do nội dung IR bị dời, không phải điểm yếu riêng của SAGE. Căn lại bằng phase/ECC tự nó cũng làm giảm điểm
(`kiem_d0.txt`: ở d = 0, căn lại làm MS-SSIM tổng −0,8…−1,4%).

File: `run_thu.py` (chạy), `tong_hop.py`/`tong_hop.txt` (bảng theo mức dịch), `kiem_d0.*`, `thu_d*.{py,txt}`
(độ ổn định của ước lượng dịch), `xem_v2/v3` (chỉ số theo nhóm độ sáng), `kiem_pr6.py`/`tong_hop_pr6.py` (chấm lại
bằng metric PR #6), `xem_bong_doi.png` (hình bóng đôi).

Ghi chú phụ (`xem_v3.txt`): SCD của SAGE cao hơn cả TB và "max" ở 100% ảnh khi độ sáng ≥ 60, nhưng MS-SSIM tổng
thì TB cao hơn SAGE ở mọi nhóm – khớp với E14 (SCD không phản ánh hiểu cảnh).
