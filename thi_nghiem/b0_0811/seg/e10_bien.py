"""E10 (kiểm G2 – SAGE có làm mờ cạnh không?): mIoU ở dải biên và ở vùng trong, sage so với vis.
Dải biên = pixel cách biên lớp trong nhãn GT <= D px (max-filter != min-filter trên cửa sổ 2D+1).
Dùng pred đã lưu của SegFormer-B0 seed 0. 273 ảnh (bỏ 7 ảnh rò). CI95 bootstrap theo khối id.
Tiêu chí đặt trước: G2 có bằng chứng nếu (sage − vis) ở dải biên < 0 với CI < 0,
hoặc nhỏ hơn rõ so với (sage − vis) ở vùng trong (hiệu của hai chênh lệch có CI < 0)."""
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from chung import FMB, MAN, NC, IGNORE, doc_nhan, miou, nham
from tong_hop import boot_idx, ci, SEED

anh = pd.read_csv(HERE.parent / "kq_anh.csv", dtype={"image_id": str})
te = pd.read_csv(MAN / "test.csv", dtype={"image_id": str}).set_index("image_id")
KQ = {}
for D in (2, 5):
    k = np.ones((2 * D + 1, 2 * D + 1), np.uint8)
    cb = {b: np.zeros((len(anh), NC, NC), np.int64) for b in ("sage", "vis")}
    ct = {b: np.zeros((len(anh), NC, NC), np.int64) for b in ("sage", "vis")}
    for i, iid in enumerate(anh.image_id):
        gt = doc_nhan(FMB / te.loc[iid, "label_path"])
        g8 = np.where(gt == IGNORE, 0, gt + 1).astype(np.uint8)
        bien = cv2.dilate(g8, k) != cv2.erode(g8, k)
        for b in ("sage", "vis"):
            p = cv2.imread(str(HERE / "runs" / f"{b}_b0_s0" / "pred" / f"{iid}.png"), cv2.IMREAD_UNCHANGED).astype(np.int64) - 1
            p[p < 0] = 0  # không xảy ra (model chỉ dự đoán 14 lớp)
            g1, g2 = gt.copy(), gt.copy()
            g1[~bien] = IGNORE; g2[bien] = IGNORE
            cb[b][i], ct[b][i] = nham(g1, p), nham(g2, p)
    sach = ~anh.leak.to_numpy(); khoi = anh.khoi[sach].to_numpy()
    rng = np.random.default_rng(SEED)
    m = lambda c, idx: miou(c[sach][idx].sum(0))[0]
    allidx = np.arange(sach.sum())
    d_b = m(cb["sage"], allidx) - m(cb["vis"], allidx)
    d_t = m(ct["sage"], allidx) - m(ct["vis"], allidx)
    bs_b, bs_t = [], []
    for idx in boot_idx(khoi, rng):
        bs_b.append(m(cb["sage"], idx) - m(cb["vis"], idx)); bs_t.append(m(ct["sage"], idx) - m(ct["vis"], idx))
    bs_b, bs_t = np.array(bs_b), np.array(bs_t)
    tile = cb["sage"][sach].sum() / (cb["sage"][sach].sum() + ct["sage"][sach].sum()) * 100
    print(f"\nD = {D} px (dải biên chiếm {tile:.1f}% pixel có nhãn)")
    print(f"  biên : vis {m(cb['vis'], allidx):.2f}  sage {m(cb['sage'], allidx):.2f}  sage−vis {d_b:+.2f} {np.round(ci(bs_b), 2)}")
    print(f"  trong: vis {m(ct['vis'], allidx):.2f}  sage {m(ct['sage'], allidx):.2f}  sage−vis {d_t:+.2f} {np.round(ci(bs_t), 2)}")
    print(f"  (biên − trong) {d_b - d_t:+.2f} {np.round(ci(bs_b - bs_t), 2)}")
    KQ[D] = (ci(bs_b), ci(bs_b - bs_t))
ok = any(c[1] < 0 or h[1] < 0 for c, h in KQ.values())
print("\nKết luận theo tiêu chí:", "G2 CÓ bằng chứng (SAGE kém ở biên)" if ok else "G2 KHÔNG có bằng chứng ở biên")
