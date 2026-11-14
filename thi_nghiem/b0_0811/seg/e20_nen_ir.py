"""E20 – SAGE lấy IR ở NỀN quá ít nên thua ghép TB? (CPU, không train, 273 ảnh test sạch)

Từ E13: SAGE lấy IR ~0,99 ở người nhưng chỉ ~0,1–0,2 ở nền (nhà, cây, trời); tb lấy 0,5 mọi nơi; mIoU tb ≥ sage.
Mỗi ảnh i:
  w_nen_i = tỉ lệ IR hiệu dụng của SAGE trên vùng nền (nhãn nhà 3, cây 6, trời 7) – cách tính như E13
  d_i     = mIoU_ảnh(tb) - mIoU_ảnh(sage)   (B0: cộng ma trận 2 seed; B3: cộng các seed có)
Giả thuyết: ảnh SAGE lấy IR ở nền càng ít thì tb thắng càng nhiều -> ρ(w_nen, d) < 0.
Nhiễu: độ sáng (ảnh tối SAGE lấy nhiều IR hơn VÀ tb thắng nhiều hơn) -> thêm tương quan riêng phần giữ độ sáng.

TIÊU CHÍ ĐẶT TRƯỚC (13/11, trước khi chạy):
  ỦNG HỘ: Spearman ρ ≤ -0,2 (CI không chứa 0) VÀ riêng phần (giữ y_mean) ≤ -0,15, ở CẢ B0 và B3
  BÁC:    |ρ| < 0,1 hoặc CI chứa 0, ở cả B0 và B3
  còn lại: lưng chừng
CI 95% bootstrap theo khối id. Chạy: python e20_nen_ir.py -> e20.txt
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from chung import ANH, FMB, GOC, MAN, thu_muc

HERE = Path(__file__).resolve().parent
B0 = HERE.parent
B3 = GOC / "tai_lap/_kaggle_out/runs_gop"
sys.path.insert(0, str(B0))
from tong_hop import boot_idx, ci, SEED  # noqa: E402

NEN = [3, 6, 7]


def miou_anh(conf):
    """mIoU từng ảnh (%), lớp có trong GT ảnh đó. conf (N,14,14) hoặc (N,15,15) với lớp 0 = unlabeled
    (bỏ pixel GT unlabeled, đoán ra lớp 0 vẫn là sai)."""
    if conf.shape[1] == 15:
        c = conf[:, 1:, :].astype(float)
        tp = np.stack([c[:, k, k + 1] for k in range(14)], 1)
        g, p = c.sum(2), c[:, :, 1:].sum(1)
    else:
        c = conf.astype(float)
        tp = np.einsum("nii->ni", c)
        g, p = c.sum(2), c.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(g > 0, tp / (g + p - tp), np.nan)
    return np.nanmean(iou, 1) * 100


def nap(b, kieu, sach):
    cs = []
    for s in (0, 1, 2):
        f = (HERE / "runs" / f"{b}_b0_s{s}" / "test_nham.npz") if kieu == "B0" else \
            (B3 / f"{b}_b3_s{s}_1220_15lop" / "test_nham_last.npz")
        if f.exists():
            cs.append(np.load(f)["conf"][sach])
    return sum(cs), len(cs)


def rieng_phan(x, y, z):
    """Spearman riêng phần: tương quan hạng của x, y sau khi bỏ phần tuyến tính theo hạng z."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    A = np.c_[np.ones_like(rz), rz]
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return np.corrcoef(ex, ey)[0, 1]


def main():
    anh = pd.read_csv(B0 / "kq_anh.csv", dtype={"image_id": str})
    sach = ~anh.leak.to_numpy()
    anh = anh[sach].reset_index(drop=True)
    man = pd.read_csv(MAN / "test.csv", dtype={"image_id": str}).set_index("image_id")

    w_nen, w_ca, px_nen = [], [], []
    for iid in anh.image_id:
        r = man.loc[iid]
        r2 = type("R", (), {"vis_path": r.vis_path})
        yv = cv2.cvtColor(cv2.imread(str(FMB / r.vis_path)), cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(float)
        ir = cv2.cvtColor(cv2.imread(str(FMB / r.ir_path)), cv2.COLOR_BGR2GRAY).astype(float)
        ys = cv2.cvtColor(cv2.imread(str(ANH / "sage" / thu_muc(r2) / f"{iid}.png")),
                          cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(float)
        lab = cv2.imread(str(FMB / r.label_path), cv2.IMREAD_UNCHANGED)
        x, y = ir - yv, ys - yv
        m = np.isin(lab, NEN)
        den = (x[m] ** 2).sum()
        w_nen.append((x[m] * y[m]).sum() / den if den > 0 else np.nan)
        px_nen.append(m.mean())
        w_ca.append((x * y).sum() / (x * x).sum())
    anh["w_nen"], anh["w_ca"], anh["px_nen"] = w_nen, w_ca, px_nen
    # bỏ ảnh nền quá ít (< 5% pixel) – w không ổn định
    ok = (anh.px_nen >= 0.05) & anh.w_nen.notna()

    out = []
    p = lambda t="": (out.append(t), print(t))
    p("E20 – w_nen (tỉ lệ IR SAGE lấy ở nền nhà/cây/trời) vs d = mIoU_ảnh(tb) - mIoU_ảnh(sage)")
    p(f"{ok.sum()}/{len(anh)} ảnh có nền ≥ 5% pixel. w_nen: trung vị {anh.w_nen[ok].median():.3f}, "
      f"P10 {anh.w_nen[ok].quantile(.1):.3f}, P90 {anh.w_nen[ok].quantile(.9):.3f}\n")
    kl = {}
    for kieu in ("B0", "B3"):
        ct, nt = nap("tb", kieu, sach)
        cs, ns = nap("sage", kieu, sach)
        d = (miou_anh(ct) - miou_anh(cs))[ok.to_numpy()]
        a = anh[ok].reset_index(drop=True)
        rho = spearmanr(a.w_nen, d).statistic
        rp = rieng_phan(a.w_nen, d, a.y_mean)
        rc = spearmanr(a.w_ca, d).statistic
        rng = np.random.default_rng(SEED)
        bs, bp = [], []
        for bi in boot_idx(a.khoi.to_numpy(), rng):
            bs.append(spearmanr(a.w_nen.to_numpy()[bi], d[bi]).statistic)
            bp.append(rieng_phan(a.w_nen.to_numpy()[bi], d[bi], a.y_mean.to_numpy()[bi]))
        c1, c2 = ci(bs), ci(bp)
        # chia 3 nhóm theo w_nen
        q = np.quantile(a.w_nen, [1 / 3, 2 / 3])
        g = np.where(a.w_nen < q[0], 0, np.where(a.w_nen < q[1], 1, 2))
        p(f"== {kieu} (tb {nt} seed, sage {ns} seed)")
        p(f"  Spearman ρ(w_nen, d)            = {rho:+.3f} [{c1[0]:+.3f}; {c1[1]:+.3f}]")
        p(f"  riêng phần, giữ độ sáng         = {rp:+.3f} [{c2[0]:+.3f}; {c2[1]:+.3f}]")
        p(f"  (tham khảo) ρ(w cả ảnh, d)      = {rc:+.3f}")
        for k, ten in enumerate(["w_nen thấp", "w_nen vừa", "w_nen cao"]):
            p(f"  {ten:11s} (w {a.w_nen[g == k].min():+.2f}..{a.w_nen[g == k].max():+.2f}): "
              f"d TB {d[g == k].mean():+.2f}, sáng TB {a.y_mean[g == k].mean():.0f}")
        ung = rho <= -0.2 and c1[1] < 0 and rp <= -0.15
        bac = abs(rho) < 0.1 or (c1[0] <= 0 <= c1[1])
        kl[kieu] = "ủng hộ" if ung else ("bác" if bac else "lưng chừng")
        p(f"  -> {kl[kieu]}\n")
    tong = ("ỦNG HỘ" if all(v == "ủng hộ" for v in kl.values()) else
            "BÁC" if all(v == "bác" for v in kl.values()) else "LƯNG CHỪNG")
    p(f"KẾT LUẬN theo tiêu chí: {tong} ({kl})")
    (HERE / "e20.txt").write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
