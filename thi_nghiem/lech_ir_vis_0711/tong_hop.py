"""Tổng hợp ket_qua.csv theo mục 12 của card. Delta tính theo từng ảnh so với (none, d=0)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

f = Path(__file__).resolve().parent / (sys.argv[1] if len(sys.argv) > 1 else "ket_qua.csv")
df = pd.read_csv(f, dtype={"image_id": str})
print("anh:", df.image_id.nunique(), "dong:", len(df))
goc = df[(df.cach == "none") & (df.d == 0)].set_index("image_id")


def bang(metric):
    print(f"\n=== {metric}: mức thay đổi so với d=0 không căn (%) theo từng ảnh ===")
    print(f"{'cach':6} {'d':>3} {'mean':>9} {'median':>9} {'%median':>8} {'%anh giam':>9}")
    for c in ("none", "phase", "ecc"):
        for d in (0, 2, 5, 10):
            s = df[(df.cach == c) & (df.d == d)].set_index("image_id")[metric]
            b = goc[metric].reindex(s.index)
            pct = (s - b) / b.abs() * 100
            print(f"{c:6} {d:>3} {s.mean():>9.4f} {s.median():>9.4f} {pct.median():>8.2f} {(s < b).mean() * 100:>9.1f}")


for m in ("ms_ssim_tong", "scd", "ms_ssim_vis", "ms_ssim_ir"):
    bang(m)


def hoi_phuc(metric):
    print(f"\n=== Hồi phục {metric} (median theo ảnh, tỉ lệ phần mất đã lấy lại) ===")
    for d in (2, 5, 10):
        n = df[(df.cach == "none") & (df.d == d)].set_index("image_id")[metric]
        b = goc[metric].reindex(n.index)
        mat = (b - n).median()
        for c in ("phase", "ecc"):
            s = df[(df.cach == c) & (df.d == d)].set_index("image_id")[metric].reindex(n.index)
            print(f"d={d:>2} {c:5} mất={mat:.4f} lấy lại={(s - n).median():.4f} tỉ lệ={(s - n).median() / mat * 100:6.1f}%")


hoi_phuc("ms_ssim_tong")
hoi_phuc("scd")

print("\n=== Độ chính xác của phép căn (px, sai số so với lượng cần dịch thật) ===")
a = df[df.cach != "none"].copy()
a["sai"] = np.hypot(a.est_sx - a.true_sx, a.est_sy - a.true_sy)
print(a.groupby(["cach", "d"]).agg(ok=("ok", "mean"), median_sai=("sai", "median"),
                                   p90_sai=("sai", lambda s: s.quantile(0.9)),
                                   sai_hon_2px=("sai", lambda s: (s > 2).mean() * 100)).round(2))
print("thời gian căn (ms):", a.groupby("cach").align_ms.median().round(1).to_dict())

print("\n=== none, d=5, theo độ sáng VIS (%median MS-SSIM tổng) ===")
g = df[(df.cach == "none") & (df.d == 5)].set_index("image_id")
pct = (g.ms_ssim_tong - goc.ms_ssim_tong.reindex(g.index)) / goc.ms_ssim_tong.reindex(g.index) * 100
nhom = pd.cut(g.vis_mean, [0, 60, 90, 130, 256], labels=["<60", "60-90", "90-130", ">=130"])
print(pd.DataFrame({"n": pct.groupby(nhom, observed=False).size(),
                    "median_%": pct.groupby(nhom, observed=False).median().round(2)}))
