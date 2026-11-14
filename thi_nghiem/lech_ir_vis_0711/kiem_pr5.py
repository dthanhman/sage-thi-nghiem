"""PR #5: 10 ID repackaged có trùng ảnh chính thức cùng số không?

Đọc ROI của Lộc bé (toạ độ 640x512), đổi sang 800x600, xem nhãn FMB chính thức cùng id có lớp
được mô tả không. Đối chứng: cùng ROI đặt lên các ảnh khác để biết tỉ lệ trúng ngẫu nhiên.
"""
from pathlib import Path

import cv2
import numpy as np

root = Path(r"D:\CS406\DoAn\Sage\xulianh\data\raw\FMB\train\Label")
ROIS = [("00064", "R1", 8, 270, 291, 321, 376), ("00064", "R2", 8, 186, 307, 226, 355),
        ("00104", "R3", 8, 33, 303, 77, 362), ("00104", "R4", 8, 122, 308, 173, 365),
        ("00109", "R5", 8, 26, 266, 65, 334), ("00109", "R6", 8, 104, 285, 157, 330),
        ("00112", "R7", 8, 175, 281, 197, 328), ("00112", "R8", 8, 49, 285, 75, 320),
        ("00145", "R9", 12, 125, 328, 227, 467)]
files = sorted(root.glob("*.png"))
rng = np.random.default_rng(42)
mau = rng.choice(len(files), 200, replace=False)
sx, sy = 800 / 640, 600 / 512


def ti_le(lb, c, b):
    x1, y1, x2, y2 = [int(v) for v in (b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy)]
    return float((lb[y1:y2, x1:x2] == c).mean())


trung = 0
for sid, rid, c, *b in ROIS:
    p = root / f"{sid}.png"
    t = ti_le(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), c, b)
    nen = np.array([ti_le(cv2.imread(str(files[i]), cv2.IMREAD_GRAYSCALE), c, b) for i in mau])
    print(f"{sid} {rid} lop {c}: cung id {t * 100:5.1f}% | nen: >=10% o {(nen >= .1).mean() * 100:4.1f}% anh")
    trung += t >= 0.1
print("ROI trung (>=10%):", trung, "/ 9")
