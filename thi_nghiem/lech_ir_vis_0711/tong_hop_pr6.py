"""Tóm tắt E(d) từ kq_pr6_<tap>.csv theo tiêu chí mục 12 của card (ngưỡng đã chốt từ trước)."""
import sys

import numpy as np
import pandas as pd

tap = sys.argv[1]
r = pd.read_csv(f"kq_pr6_{tap}.csv", dtype={"image_id": str})
rng = np.random.default_rng(42)
print(f"Tập {tap}: {r.image_id.nunique()} ảnh, {r.khoi.nunique()} khối")
for m in ("ms_sum", "ms_mean", "scd", "ms_ir", "ms_vis"):
    p = r.pivot_table(index=["image_id", "khoi"], columns=["cach", "d"], values=m).reset_index()
    print(f"\n[{m}]")
    for d in (5, 10):
        L = {c: (p[(c, d)] - p[(c, 0)]) / p[(c, 0)] * 100 for c in ("sage", "tb")}
        e = L["sage"] - L["tb"]
        ks = p.khoi.unique()
        B = [np.median(np.concatenate([e[p.khoi == k].values for k in rng.choice(ks, len(ks))]))
             for _ in range(5000)]
        lo, hi = np.percentile(B, [2.5, 97.5])
        kl = ""
        if d == 5 and m in ("ms_sum", "ms_mean"):
            kl = ("XÁC NHẬN" if e.median() <= -1 and hi < 0 else
                  "BÁC BỎ" if lo > -0.5 else "CHƯA KẾT LUẬN")
        print(f"  d={d:2d}: L_SAGE {L['sage'].median():+.2f}%  L_TB {L['tb'].median():+.2f}%  "
              f"E median {e.median():+.2f}  CI95 [{lo:+.2f}; {hi:+.2f}]  "
              f"SAGE kém hơn TB ở {(e < 0).mean() * 100:.0f}% ảnh  {kl}")
        print("        E theo khối:", e.groupby(p.khoi).median().round(2).tolist())
