"""Thăm dò D mở rộng trên 1.220 ảnh train chính thức (train.csv + dev.csv). Không dùng test.

Tiêu chí đặt TRƯỚC khi chạy:
- Chỉ xét các khối có >= 5 ảnh VIS rất tối (vis_mean < 60).
- "Dấu hiệu thật" nếu ở >= 60% số khối đó, median SCD của SAGE thấp hơn MAX,
  và SAGE thua MAX ở > 50% tổng số ảnh rất tối.
- Nếu có < 3 khối đủ điều kiện: chưa kết luận được.
- Nếu IR-only cũng thắng SAGE ở nhóm tối, phải xem SCD có đang ưu ái ảnh giống IR không.
Metric là bản tạm, kết quả chỉ để thăm dò.
"""
import subprocess

import cv2
import numpy as np
import pandas as pd
import torch

import run_thu as R

repo = R.HERE.parent / "xulianh"
tr = subprocess.check_output(["git", "-C", str(repo), "show", "origin/feature/man-dataset:data/manifests/train.csv"], text=True, encoding="utf-8")
(R.HERE / "train.csv").write_text(tr, encoding="utf-8")
ds = pd.concat([pd.read_csv(R.HERE / "train.csv", dtype={"image_id": str}),
                pd.read_csv(R.HERE / "dev.csv", dtype={"image_id": str})]).sort_values("image_id")
ids = ds.image_id.astype(int).values
ds["khoi"] = np.r_[0, np.cumsum(np.diff(ids) != 1)]
assert len(ds) == 1220 and ds.image_id.is_unique
print("so khoi", ds.khoi.nunique(), flush=True)
root = repo / "data/raw/FMB"
net = R.Network()
net.load_state_dict(torch.load("SAGE_IVIF/checkpoints/checkpoints.pt", map_location="cpu", weights_only=True))
net.cuda().eval()
k = R.gauss("cuda")
t = lambda a: torch.from_numpy(a).float().div(255)[None, None].cuda()


def en(f):
    h = torch.histc(f * 255, 256, 0, 255)
    p = h / h.sum()
    p = p[p > 0]
    return float(-(p * p.log2()).sum())


rows = []
for n, r in enumerate(ds.itertuples(), 1):
    y = cv2.cvtColor(cv2.imread(str(root / r.vis_path)), cv2.COLOR_BGR2YCrCb)[:, :, 0]
    ir = cv2.cvtColor(cv2.imread(str(root / r.ir_path)), cv2.COLOR_BGR2GRAY)
    ty, ti = t(y), t(ir)
    with torch.inference_mode():
        fs = net(ty, ti)[0].clamp(0, 1)
    for ten, f in (("SAGE", fs), ("TB", (ty + ti) / 2), ("MAX", torch.max(ty, ti)), ("IR", ti)):
        m = R.cham(f, ty, ti, k)
        rows.append({"image_id": r.image_id, "khoi": r.khoi, "tap": r.split, "vis_mean": float(y.mean()),
                     "ir_mean": float(ir.mean()), "cach": ten, "scd": m["scd"], "ms": m["ms_ssim_tong"],
                     "en": en(f), "sd": float(f.std() * 255)})
    if n % 200 == 0:
        print(n, flush=True)
pd.DataFrame(rows).to_csv("ket_qua_d2.csv", index=False)

d = pd.DataFrame(rows)
nhan = ["<60", "60-90", "90-130", ">=130"]
d["nhom"] = pd.cut(d.vis_mean, [0, 60, 90, 130, 256], labels=nhan)
s = d[d.cach == "SAGE"]
print(pd.DataFrame({"anh": s.groupby("nhom", observed=False).size(), "khoi": s.groupby("nhom", observed=False).khoi.nunique()}))
for m in ("scd", "ms", "en", "sd"):
    print("\n", m)
    print(d.pivot_table(index="nhom", columns="cach", values=m, aggfunc="median", observed=False)[["SAGE", "TB", "MAX", "IR"]].round(3))
p = d.pivot_table(index=["image_id", "khoi"], columns="cach", values="scd").reset_index()
p = p.merge(s[["image_id", "vis_mean", "nhom"]], on="image_id")
print("\n% ảnh SAGE thắng", {c: (p.SAGE > p[c]).groupby(p.nhom, observed=False).mean().mul(100).round(0).to_dict() for c in ("TB", "MAX", "IR")})
toi = p[p.nhom == "<60"]
c = toi.groupby("khoi").agg(n=("SAGE", "size"), id_dau=("image_id", "min"), SAGE=("SAGE", "median"), MAX=("MAX", "median"), IR=("IR", "median"), vis=("vis_mean", "median"))
print(c.round(3))
du = c[c.n >= 5]
print("Khối đủ điều kiện:", len(du), "| SAGE < MAX ở", int((du.SAGE < du.MAX).sum()), "khối | SAGE thua MAX ở %.0f%% ảnh tối" % ((toi.SAGE < toi.MAX).mean() * 100))
