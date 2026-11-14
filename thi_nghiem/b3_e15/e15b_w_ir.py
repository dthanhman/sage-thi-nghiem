"""E15b – Khi VIS hỏng, SAGE có lấy IR nhiều hơn không? (không train, 273 ảnh test sạch, vài phút)

Dùng lại cách đo của E13: trong mỗi vùng, ước lượng tỉ lệ IR hiệu dụng w bằng bình phương nhỏ nhất
    Y_ghep - Y_vis_hong ≈ w * (IR - Y_vis_hong)   ->   w = Σ(Yg-Yv)(IR-Yv) / Σ(IR-Yv)²
w = 0: chỉ VIS (đã hỏng), w = 1: chỉ IR. Y_vis_hong là kênh Y của VIS ĐÃ làm hỏng (đúng thứ SAGE nhận vào).
Điều kiện làm hỏng giống hệt E15 (cùng hàm lam_hong). Kiểm thử: tb phải ra 0,5 ở mọi mức.
Vùng: cả ảnh, người (nhãn 8), nền (road, sidewalk, building, vegetation, sky, pole).

TIÊU CHÍ ĐẶT TRƯỚC (ghi 14/11, trước khi chạy), Δw = w(mức 3) − w(sạch), cả ảnh, CI95 bootstrap theo khối:
  D1 [toi]   Δw ≥ +0,10 và CI > 0          -> SAGE CÓ tăng IR khi VIS tối+nhiễu (thua tb vì cách trộn, không vì thiếu IR)
             Δw < +0,05 hoặc CI chứa 0     -> SAGE KHÔNG thích nghi theo chất lượng VIS (xác nhận gap của E15)
             còn lại                        -> LƯNG CHỪNG
  D2 [suong] cùng ngưỡng, để đối chiếu (E15: SAGE xử lý sương tốt)
Chạy: python e15b_w_ir.py -> e15b.txt
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e15_lam_hong_vis import B0, DK, GhepSAGE, lam_hong  # noqa: E402  (đã thêm b3/ và B0 vào sys.path)
from chung import MAN, duong  # noqa: E402
from tong_hop import boot_idx, ci, SEED  # noqa: E402

VUNG = {"cả ảnh": list(range(15)), "người": [8], "nền": [1, 2, 3, 6, 7, 14]}
BT = ["sage", "tb"]


def main():
    anh = pd.read_csv(B0 / "kq_anh.csv", dtype={"image_id": str})
    anh = anh[~anh.leak].reset_index(drop=True)
    man = pd.read_csv(MAN / "test.csv", dtype={"image_id": str}).set_index("image_id")
    sage = GhepSAGE("cuda" if torch.cuda.is_available() else "cpu")
    V = list(VUNG)
    # S[(bt, dk)][i, v] = (tử, mẫu, SS_res, SS_tot, số pixel)
    S = {(b, d): np.zeros((len(anh), len(V), 5)) for b in BT for d in DK}
    for i, iid in enumerate(anh.image_id):
        r = man.loc[iid]
        vis = cv2.imread(str(duong(r.vis_path)))
        ir = cv2.cvtColor(cv2.imread(str(duong(r.ir_path))), cv2.COLOR_BGR2GRAY)
        lab = cv2.imread(str(duong(r.label_path)), cv2.IMREAD_UNCHANGED)
        mask = [np.isin(lab, VUNG[v]) for v in V]
        for d in DK:
            yv8 = cv2.cvtColor(lam_hong(vis, d, iid), cv2.COLOR_BGR2YCrCb)[:, :, 0]
            yv = yv8.astype(np.float64)
            x = ir - yv
            yg = {"sage": sage(yv8, ir).astype(np.float64),
                  "tb": ((yv + ir) / 2).round()}
            for b in BT:
                y = yg[b] - yv
                for j, m in enumerate(mask):
                    if not m.any():
                        continue
                    xm, ym = x[m], y[m]
                    num, den = (xm * ym).sum(), (xm * xm).sum()
                    w = num / den if den > 0 else 0
                    S[(b, d)][i, j] = (num, den, ((ym - w * xm) ** 2).sum(), ((ym - ym.mean()) ** 2).sum(), m.sum())
        if i % 50 == 0:
            print(f"  {i}/{len(anh)}", flush=True)

    def w_of(s, idx, j):
        return s[idx, j, 0].sum() / s[idx, j, 1].sum()

    out = []
    p = lambda t="": (out.append(t), print(t))
    p(__doc__.split("Chạy:")[0].strip() + "\n")
    tat = np.arange(len(anh))
    q = np.quantile(anh.y_mean, [1 / 3, 2 / 3])
    nhom = np.where(anh.y_mean < q[0], "tối", np.where(anh.y_mean < q[1], "vừa", "sáng"))
    for g in ["tất cả", "tối", "sáng"]:
        idx = tat if g == "tất cả" else tat[nhom == g]
        p(f"== w của SAGE [{g} – theo độ sáng ảnh GỐC] {len(idx)} ảnh   (tb cả ảnh để kiểm, phải = 0,5)")
        p(f"  {'mức':7s} " + "  ".join(f"{v:>8s}" for v in V) + "   R²(cả ảnh)  tb")
        for d in DK:
            s = S[("sage", d)]
            r2 = 1 - s[idx, 0, 2].sum() / s[idx, 0, 3].sum()
            p(f"  {d:7s} " + "  ".join(f"{w_of(s, idx, j):8.3f}" for j in range(len(V)))
              + f"   {r2:9.2f}  {w_of(S[('tb', d)], idx, 0):.3f}")
        p()

    p("== TIÊU CHÍ (cả ảnh, tất cả 273 ảnh); kèm Δw vùng người và nền")
    khoi = anh.khoi.to_numpy()
    for k, ten in (("toi", "D1"), ("suong", "D2")):
        for j, v in enumerate(V):
            s0, s3 = S[("sage", "sach")], S[("sage", f"{k}3")]
            dw = w_of(s3, tat, j) - w_of(s0, tat, j)
            rng = np.random.default_rng(SEED)
            c = ci([w_of(s3, bi, j) - w_of(s0, bi, j) for bi in boot_idx(khoi, rng)])
            tt = ""
            if v == "cả ảnh":
                tt = (" -> SAGE CÓ tăng IR" if dw >= 0.10 and c[0] > 0
                      else " -> SAGE KHÔNG thích nghi theo chất lượng VIS" if dw < 0.05 or c[0] <= 0 < c[1]
                      else " -> LƯNG CHỪNG")
            p(f"{ten if v == 'cả ảnh' else '  '} [{k}] {v:6s} w sạch {w_of(s0, tat, j):.3f} -> mức 3 {w_of(s3, tat, j):.3f}"
              f" | Δw {dw:+.3f} [{c[0]:+.3f}; {c[1]:+.3f}]{tt}")
    (HERE / "e15b.txt").write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
