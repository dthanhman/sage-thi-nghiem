"""Thí nghiệm trần tỉ lệ IR (seed 0): vis / ir30 / tb(=ir50) / ir70 / sage, 273 ảnh sạch.
CI95 bootstrap theo khối id (như so_sanh4.py). Chia tối/vừa/sáng theo y_mean."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from chung import miou, TEN_LOP
from tong_hop import boot_idx, ci, SEED
anh = pd.read_csv(HERE.parent / "kq_anh.csv", dtype={"image_id": str})
def nap(run):
    z = np.load(HERE / "runs" / run / "test_nham.npz")
    assert list(z["image_id"]) == list(anh.image_id), run
    return z["conf"]
BT = ["vis", "ir30", "tb", "ir70", "sage"]
C = {b: nap(f"{b}_b0_s0") for b in BT}
sach = ~anh.leak.to_numpy()
q = np.quantile(anh.y_mean[sach], [1/3, 2/3])
nhom = np.where(anh.y_mean < q[0], "toi", np.where(anh.y_mean < q[1], "vua", "sang"))
CAP = [("ir30", "vis"), ("tb", "vis"), ("ir70", "vis"), ("ir70", "tb"), ("ir70", "sage"), ("tb", "ir30"), ("sage", "ir30")]
P = TEN_LOP.index("person")
for g, chon in [("273 ảnh", sach), ("tối", sach & (nhom == "toi")),
                ("vừa", sach & (nhom == "vua")), ("sáng", sach & (nhom == "sang"))]:
    khoi = anh.khoi[chon].to_numpy()
    m = {b: miou(C[b][chon].sum(0)) for b in BT}
    print(f"\n== [{g}] {chon.sum()} ảnh | " + "  ".join(f"{b} {m[b][0]:.2f}" for b in BT)
          + "\n   person " + "  ".join(f"{b} {m[b][1][P]:.1f}" for b in BT))
    rng = np.random.default_rng(SEED)
    ds = {c: [] for c in CAP}
    for i in boot_idx(khoi, rng):
        mm = {b: miou(C[b][chon][i].sum(0))[0] for b in BT}
        for a, b in CAP: ds[(a, b)].append(mm[a] - mm[b])
    for a, b in CAP:
        c = ci(ds[(a, b)]); s = "*" if c[0] > 0 or c[1] < 0 else " "
        print(f"  {a:5s}−{b:5s} {m[a][0]-m[b][0]:+.2f} [{c[0]:+.2f}; {c[1]:+.2f}]{s}")
print("\nIoU từng lớp (273 ảnh):")
m = {b: miou(C[b][sach].sum(0))[1] for b in BT}
for k, t in enumerate(TEN_LOP):
    if np.isnan(m["vis"][k]): continue
    print(f"  {t:14s} " + "  ".join(f"{b} {m[b][k]:5.1f}" for b in BT))
