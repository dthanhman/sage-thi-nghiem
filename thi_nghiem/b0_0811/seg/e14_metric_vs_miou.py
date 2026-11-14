"""E14 – Chỉ số ghép (không cần nhãn) có xếp hạng ảnh giống mIoU (phân đoạn thật) không? Không train.

Chỉ số: SCD, MS_SSIM_sum, Q_ABF (chỉ số ghép chính của bài SAGE) + EN, SD, SF, AG.
 - vis, ir, tb, sage: lấy từ ../kq_metric.csv (E0, code metric PR #6, kênh Y).
 - clahe, ir30, ir70: tính mới ở đây bằng cùng code, từ Y của ảnh đã xuất ở anh/.
mIoU: B0 (seg/runs, TB các seed có: vis, clahe, sage, tb x2; ir30, ir70 x1), B3 (tai_lap runs_gop: vis, ir, sage, tb).

TIÊU CHÍ ĐẶT TRƯỚC (13/11, trước khi chạy), 273 ảnh test sạch:
  Đáng tin làm proxy:  Kendall τ (xếp hạng biến thể theo chỉ số TB vs mIoU) ≥ 0,6 ở CẢ B0 và B3
                       VÀ tỉ lệ đồng ý theo ảnh ≥ 60 %
  Không đáng tin:      τ ≤ 0,2 (B0 hoặc B3) HOẶC đồng ý theo ảnh ≤ 55 %
  Đồng ý theo ảnh: với mỗi ảnh, mỗi cặp biến thể B0, dấu(Δ chỉ số) == dấu(Δ mIoU của ảnh đó). 50 % = đoán bừa.
Chạy: python e14_metric_vs_miou.py -> e14.txt, kq_e14_metric.csv
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from chung import ANH, FMB, GOC, MAN, thu_muc

HERE = Path(__file__).resolve().parent
B0 = HERE.parent
PR6 = GOC / "pr6_eval"
B3 = GOC / "tai_lap/_kaggle_out/runs_gop"
MOI = ["clahe", "ir30", "ir70"]
CS = ["SCD", "MS_SSIM_sum", "Q_ABF", "EN", "SD", "SF", "AG"]
# chiều "tốt hơn": mọi chỉ số ở đây càng cao càng tốt theo quy ước bài ghép ảnh


def _ha_uu_tien():
    """Hạ độ ưu tiên tiến trình (Windows) để máy không lag."""
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)


def _init():
    _ha_uu_tien()
    sys.path.insert(0, str(PR6))


def cham(job):
    from src.evaluation.metrics import (average_gradient, entropy, fusion_ms_ssim, q_abf,
                                        spatial_frequency, standard_deviation,
                                        sum_of_correlation_differences)
    iid, bt, fp, vp, ip = job
    f = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float64)
    y = cv2.cvtColor(cv2.imread(vp), cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float64)
    ir = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2GRAY).astype(np.float64)
    ms = fusion_ms_ssim(f, ir, y, data_range=255.0, aggregation="sum")
    return {"image_id": iid, "bien_the": bt, "EN": entropy(f), "SD": standard_deviation(f),
            "SF": spatial_frequency(f), "AG": average_gradient(f),
            "SCD": sum_of_correlation_differences(f, ir, y), "MS_SSIM_sum": ms["MS_SSIM_sum"],
            "Q_ABF": q_abf(f, ir, y)}


def miou_anh(conf):
    """mIoU từng ảnh (%), trên các lớp có trong GT của ảnh đó. conf: (N, C, C)."""
    tp = np.einsum("nii->ni", conf).astype(float)
    g, p = conf.sum(2), conf.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(g > 0, tp / (g + p - tp), np.nan)
    return np.nanmean(iou, 1) * 100


def miou_tap(conf):
    c = conf.sum(0)
    if len(c) == 15:  # model B3 15 lớp: bỏ pixel GT unlabeled, đoán ra lớp 0 vẫn tính là sai (như cac_miou)
        c = c[1:].astype(float)
        tp = np.array([c[k, k + 1] for k in range(14)])
        g, p = c.sum(1), c[:, 1:].sum(0)
        ok = g > 0
        return (tp[ok] / (g[ok] + p[ok] - tp[ok])).mean() * 100
    tp = np.diag(c).astype(float)
    g, p = c.sum(1), c.sum(0)
    ok = g > 0
    return (tp[ok] / (g[ok] + p[ok] - tp[ok])).mean() * 100


def main():
    _ha_uu_tien()
    out_csv = HERE / "kq_e14_metric.csv"
    man = pd.read_csv(MAN / "test.csv", dtype={"image_id": str})
    if out_csv.exists():
        moi = pd.read_csv(out_csv, dtype={"image_id": str})
    else:
        jobs = [(r.image_id, b, str(ANH / b / thu_muc(r) / f"{r.image_id}.png"), str(FMB / r.vis_path),
                 str(FMB / r.ir_path)) for b in MOI for r in man.itertuples()]
        res = []
        with ProcessPoolExecutor(int(os.environ.get("SO_TIEN_TRINH", 4)), initializer=_init) as ex:
            for n, o in enumerate(ex.map(cham, jobs, chunksize=4), 1):
                res.append(o)
                if n % 100 == 0:
                    print(f"  metric {n}/{len(jobs)}", flush=True)
        moi = pd.DataFrame(res)
        moi.to_csv(out_csv, index=False)
    cu = pd.read_csv(B0 / "kq_metric.csv", dtype={"image_id": str})
    cu = cu[cu.bien_the.isin(["vis", "ir", "tb", "sage"])]
    M = pd.concat([cu[["image_id", "bien_the"] + CS], moi[["image_id", "bien_the"] + CS]])

    anh = pd.read_csv(B0 / "kq_anh.csv", dtype={"image_id": str})
    sach_id = list(anh.image_id[~anh.leak])
    M = M[M.image_id.isin(sach_id)]
    tb_metric = M.groupby("bien_the")[CS].mean()

    # ---- mIoU
    def conf_b0(b):
        cs = []
        for s in (0, 1):
            f = HERE / "runs" / f"{b}_b0_s{s}" / "test_nham.npz"
            if f.exists():
                z = np.load(f)
                assert list(z["image_id"].astype(str)) == list(anh.image_id)
                cs.append(z["conf"][~anh.leak.to_numpy()])
        return cs

    def conf_b3(b):
        cs = []
        for s in (0, 1, 2):
            f = B3 / f"{b}_b3_s{s}_1220_15lop" / "test_nham_last.npz"
            if f.exists():
                z = np.load(f)
                cs.append(z["conf"][~anh.leak.to_numpy()])  # 15x15, unlabeled = lớp 0
        return cs

    BT0 = ["vis", "clahe", "sage", "tb", "ir30", "ir70"]
    BT3 = ["vis", "ir", "sage", "tb"]
    C0 = {b: conf_b0(b) for b in BT0}
    C3 = {b: conf_b3(b) for b in BT3}
    m0 = {b: np.mean([miou_tap(c) for c in C0[b]]) for b in BT0}
    m3 = {b: np.mean([miou_tap(c) for c in C3[b]]) for b in BT3}
    # mIoU từng ảnh B0: cộng ma trận các seed rồi tính
    a0 = {b: miou_anh(sum(C0[b])) for b in BT0}

    out = []
    p = lambda t="": (out.append(t), print(t))
    p("E14 – chỉ số ghép vs mIoU (273 ảnh test sạch)\n")
    p("== Giá trị TB theo biến thể")
    bang = tb_metric.copy()
    bang["mIoU_B0"] = pd.Series(m0)
    bang["mIoU_B3"] = pd.Series(m3)
    order = ["vis", "ir", "clahe", "ir30", "sage", "tb", "ir70"]
    p(bang.loc[order].round(3).to_string())
    p("\n== Xếp hạng (1 = cao nhất)")
    p(bang.loc[order].rank(ascending=False).to_string())

    p("\n== Kendall τ giữa chỉ số và mIoU (cấp tập), đồng ý theo ảnh (B0, 15 cặp biến thể)")
    p(f"  {'chỉ số':12s} {'τ B0 (6)':>9s} {'τ B3 (4)':>9s} {'đồng ý/ảnh':>11s}  kết luận")
    for c in CS:
        t0 = kendalltau([tb_metric.loc[b, c] for b in BT0], [m0[b] for b in BT0]).statistic
        t3 = kendalltau([tb_metric.loc[b, c] for b in BT3], [m3[b] for b in BT3]).statistic
        Mi = M.pivot(index="image_id", columns="bien_the", values=c).loc[sach_id]
        dung = tong = 0
        for a, b in combinations(BT0, 2):
            dm = Mi[a].to_numpy() - Mi[b].to_numpy()
            di = a0[a] - a0[b]
            ok = (dm != 0) & (di != 0) & ~np.isnan(di)
            dung += (np.sign(dm[ok]) == np.sign(di[ok])).sum()
            tong += ok.sum()
        dy = 100 * dung / tong
        if t0 >= 0.6 and t3 >= 0.6 and dy >= 60:
            kl = "ĐÁNG TIN"
        elif t0 <= 0.2 or t3 <= 0.2 or dy <= 55:
            kl = "KHÔNG ĐÁNG TIN"
        else:
            kl = "lưng chừng"
        p(f"  {c:12s} {t0:9.2f} {t3:9.2f} {dy:10.1f}%  {kl}")
    (HERE / "e14.txt").write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
