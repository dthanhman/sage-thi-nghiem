"""E13 – SAGE lấy IR ở vùng nào? (không train, 273 ảnh test sạch)

Mỗi vùng (theo nhãn GT) ước lượng tỉ lệ IR hiệu dụng w bằng bình phương nhỏ nhất:
    Y_ghep - Y_vis ≈ w * (IR - Y_vis)   ->   w = Σ(Yg-Yv)(IR-Yv) / Σ(IR-Yv)²
w = 0: chỉ VIS, w = 1: chỉ IR. Kèm R² (SAGE không phải trộn tuyến tính nên R² cho biết mô tả bằng w khớp tới đâu).
Kiểm thử: tb phải ra 0,5, ir30 ra 0,3.

TIÊU CHÍ ĐẶT TRƯỚC (ghi 13/11, trước khi chạy):
  C1  w toàn ảnh của SAGE < 0,4                         -> SAGE dùng IR ít (khớp sage ≈ ir30 ở B0)
  C2  max-min w giữa các vùng ≥ 0,15 VÀ CI(w_người - w_mặt đường) không chứa 0 -> SAGE CÓ chọn theo vùng
      max-min < 0,05                                     -> gần như trộn cố định
  C3  w_người(tối) - w_người(sáng) ≥ 0,1, CI không chứa 0 -> thích nghi theo độ sáng ở vùng người
CI 95% bootstrap theo khối id (như các E trước).
Chạy: python e13_vung_ir.py  -> e13.txt
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from chung import ANH, FMB, MAN, thu_muc

B0 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(B0))
from tong_hop import boot_idx, ci, SEED  # noqa: E402

# nhãn FMB gốc (0 unlabeled, 1..14)
VUNG = {"người": [8], "xe": [9, 10, 11, 12, 13], "mặt đường": [1, 2], "nhà": [3], "cây": [6],
        "trời": [7], "biển/đèn": [4, 5], "cột": [14], "cả ảnh": list(range(0, 15))}
BT = ["sage", "tb", "ir30"]


def doc_y(p):
    im = cv2.imread(str(p))
    assert im is not None, p
    return cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float64)


def main():
    anh = pd.read_csv(B0 / "kq_anh.csv", dtype={"image_id": str})
    anh = anh[~anh.leak].reset_index(drop=True)
    man = pd.read_csv(MAN / "test.csv", dtype={"image_id": str}).set_index("image_id")
    V = list(VUNG)
    # S[bt][i, v] = (tử, mẫu, SS_res, SS_tot, số pixel)
    S = {b: np.zeros((len(anh), len(V), 5)) for b in BT}
    for i, iid in enumerate(anh.image_id):
        r = man.loc[iid]
        r2 = type("R", (), {"vis_path": r.vis_path})
        vis = cv2.imread(str(FMB / r.vis_path))
        yv = cv2.cvtColor(vis, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float64)
        ir = cv2.cvtColor(cv2.imread(str(FMB / r.ir_path)), cv2.COLOR_BGR2GRAY).astype(np.float64)
        lab = cv2.imread(str(FMB / r.label_path), cv2.IMREAD_UNCHANGED)
        x = ir - yv
        for b in BT:
            y = doc_y(ANH / b / thu_muc(r2) / f"{iid}.png") - yv
            for j, v in enumerate(V):
                m = np.isin(lab, VUNG[v])
                if not m.any():
                    continue
                xm, ym = x[m], y[m]
                num, den = (xm * ym).sum(), (xm * xm).sum()
                w = num / den if den > 0 else 0
                S[b][i, j] = (num, den, ((ym - w * xm) ** 2).sum(), ((ym - ym.mean()) ** 2).sum(), m.sum())
        if i % 50 == 0:
            print(f"  {i}/{len(anh)}", flush=True)

    def w_of(s, idx, j):
        return s[idx, j, 0].sum() / s[idx, j, 1].sum()

    q = np.quantile(anh.y_mean, [1 / 3, 2 / 3])
    nhom = np.where(anh.y_mean < q[0], "tối", np.where(anh.y_mean < q[1], "vừa", "sáng"))
    out = []
    p = lambda t="": (out.append(t), print(t))
    p("E13 – tỉ lệ IR hiệu dụng w theo vùng (273 ảnh test sạch). w=0 chỉ VIS, w=1 chỉ IR.")
    p("R² = mức khớp của mô tả 'trộn tuyến tính'; %px = tỉ lệ pixel của vùng.\n")
    tat = np.arange(len(anh))
    for g in ["tất cả", "tối", "vừa", "sáng"]:
        idx = tat if g == "tất cả" else tat[nhom == g]
        p(f"== [{g}] {len(idx)} ảnh")
        p(f"  {'vùng':10s} " + "  ".join(f"{b:>12s}" for b in BT) + "   R²(sage)  %px  số ảnh")
        tong_px = S["sage"][idx, V.index("cả ảnh"), 4].sum()
        for j, v in enumerate(V):
            s = S["sage"]
            co = s[idx, j, 4] > 0
            if s[idx, j, 1].sum() == 0:
                continue
            r2 = 1 - s[idx, j, 2].sum() / s[idx, j, 3].sum()
            p(f"  {v:10s} " + "  ".join(f"{w_of(S[b], idx, j):12.3f}" for b in BT)
              + f"   {r2:7.2f}  {100 * s[idx, j, 4].sum() / tong_px:4.1f}  {co.sum():4d}")
        p()

    # Bootstrap: C2 (người - mặt đường), C3 (người tối - sáng)
    s = S["sage"]
    jn, jd, jc = V.index("người"), V.index("mặt đường"), V.index("cả ảnh")
    rng = np.random.default_rng(SEED)
    d2, d3, wc, rng_v = [], [], [], []
    khoi = anh.khoi.to_numpy()
    vung_chinh = [V.index(v) for v in V if v != "cả ảnh"]
    for bi in boot_idx(khoi, rng):
        d2.append(w_of(s, bi, jn) - w_of(s, bi, jd))
        wc.append(w_of(s, bi, jc))
        t, sa = bi[nhom[bi] == "tối"], bi[nhom[bi] == "sáng"]
        d3.append(w_of(s, t, jn) - w_of(s, sa, jn))
        ws = [w_of(s, bi, j) for j in vung_chinh if s[bi, j, 1].sum() > 0]
        rng_v.append(max(ws) - min(ws))
    W = {v: w_of(s, tat, V.index(v)) for v in V if v != "cả ảnh"}
    c1, c2, c3, cr = ci(wc), ci(d2), ci(d3), ci(rng_v)
    tn, ts = tat[nhom == "tối"], tat[nhom == "sáng"]
    p("== TIÊU CHÍ (SAGE)")
    wca = w_of(s, tat, jc)
    p(f"C1 w cả ảnh = {wca:.3f} [{c1[0]:.3f}; {c1[1]:.3f}] -> "
      + ("ĐẠT: SAGE dùng IR ít" if c1[1] < 0.4 else "KHÔNG ĐẠT"))
    rg = max(W.values()) - min(W.values())
    p(f"C2 max-min w giữa vùng = {rg:.3f} [{cr[0]:.3f}; {cr[1]:.3f}] "
      f"({max(W, key=W.get)} {max(W.values()):.3f} / {min(W, key=W.get)} {min(W.values()):.3f}); "
      f"người - mặt đường = {W['người'] - W['mặt đường']:+.3f} [{c2[0]:+.3f}; {c2[1]:+.3f}] -> "
      + ("CÓ chọn theo vùng" if rg >= 0.15 and (c2[0] > 0 or c2[1] < 0)
         else "gần như trộn cố định" if rg < 0.05 else "LƯNG CHỪNG"))
    v3 = w_of(s, tn, jn) - w_of(s, ts, jn)
    p(f"C3 w_người tối - sáng = {v3:+.3f} [{c3[0]:+.3f}; {c3[1]:+.3f}] -> "
      + ("ĐẠT: thích nghi theo độ sáng" if v3 >= 0.1 and c3[0] > 0 else "KHÔNG ĐẠT"))
    Path(__file__).with_name("e13.txt").write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
