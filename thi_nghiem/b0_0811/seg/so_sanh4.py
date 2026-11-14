"""So 4 biến thể vis/sage/tb/clahe (SegFormer-B0) trên FMB test 273 ảnh sạch.
Cặp so trong từng seed (s0, s1); CI95 bootstrap theo khối id như so_sanh.py.
Kèm bảng seed: 4 biến thể × s0, s1 (TB ± SD)."""
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
BT = ["vis", "clahe", "sage", "tb"]
C = {b: nap(f"{b}_b0_s0") for b in BT}
C1 = {b: nap(f"{b}_b0_s1") for b in BT}
sach = ~anh.leak.to_numpy()
q = np.quantile(anh.y_mean[sach], [1/3, 2/3])
nhom = np.where(anh.y_mean < q[0], "toi", np.where(anh.y_mean < q[1], "vua", "sang"))
CAP = [("sage", "vis"), ("clahe", "vis"), ("sage", "clahe"), ("tb", "vis"), ("tb", "sage")]
P = TEN_LOP.index("person")

print("== Seed (273 ảnh): mIoU")
for b in BT:
    v = [miou(C[b][sach].sum(0))[0], miou(C1[b][sach].sum(0))[0]]
    print(f"  {b:6s} s0 {v[0]:.2f}  s1 {v[1]:.2f}  TB {np.mean(v):.2f} ± {np.std(v, ddof=1):.2f}")
d0 = miou(C['sage'][sach].sum(0))[0] - miou(C['vis'][sach].sum(0))[0]
d1 = miou(C1['sage'][sach].sum(0))[0] - miou(C1['vis'][sach].sum(0))[0]
print(f"  sage−vis: s0 {d0:+.2f}  s1 {d1:+.2f}")
for a, b in [("clahe", "vis"), ("tb", "vis"), ("tb", "sage")]:
    e = [miou(X[a][sach].sum(0))[0] - miou(X[b][sach].sum(0))[0] for X in (C, C1)]
    print(f"  {a}−{b}: s0 {e[0]:+.2f}  s1 {e[1]:+.2f}  TB {np.mean(e):+.2f}")

for sd, C in [(0, C), (1, C1)]:
  print(f"\n######## SEED {sd} ########")
  for g, chon in [("273 ảnh", sach), ("tối", sach & (nhom == "toi")),
                  ("vừa", sach & (nhom == "vua")), ("sáng", sach & (nhom == "sang"))]:
      khoi = anh.khoi[chon].to_numpy()
      m = {b: miou(C[b][chon].sum(0)) for b in BT}
      print(f"\n== [{g}] {chon.sum()} ảnh | " + "  ".join(f"{b} {m[b][0]:.2f}" for b in BT)
            + " | person " + "  ".join(f"{b} {m[b][1][P]:.1f}" for b in BT))
      rng = np.random.default_rng(SEED)
      ds = {c: [] for c in CAP}; dp = {c: [] for c in CAP}
      for i in boot_idx(khoi, rng):
          mm = {b: miou(C[b][chon][i].sum(0)) for b in BT}
          for a, b in CAP:
              ds[(a, b)].append(mm[a][0] - mm[b][0]); dp[(a, b)].append(mm[a][1][P] - mm[b][1][P])
      for a, b in CAP:
          c = ci(ds[(a, b)]); cp = np.nanpercentile(dp[(a, b)], [2.5, 97.5])
          s = "*" if c[0] > 0 or c[1] < 0 else " "
          print(f"  {a:5s}−{b:5s} {m[a][0]-m[b][0]:+.2f} [{c[0]:+.2f}; {c[1]:+.2f}]{s}  "
                f"person {m[a][1][P]-m[b][1][P]:+.1f} [{cp[0]:+.1f}; {cp[1]:+.1f}]")
