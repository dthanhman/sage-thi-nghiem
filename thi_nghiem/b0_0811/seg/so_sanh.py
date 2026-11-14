"""So sánh mIoU thật sage và vis (SegFormer-B0, seed 0) trên FMB test.
E2 thật: sage − vis toàn bộ + từng lớp. E3 thật: theo 3 nhóm độ sáng (ngưỡng như e3.py).
CI95 = bootstrap theo khối id (B=2000), như tong_hop.py."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from chung import miou, TEN_LOP
from tong_hop import boot_idx, ci, SEED

anh = pd.read_csv(HERE.parent / "kq_anh.csv", dtype={"image_id": str})
C = {}
for b in ["sage", "vis"]:
    z = np.load(HERE / "runs" / f"{b}_b0_s0" / "test_nham.npz")
    assert list(z["image_id"]) == list(anh.image_id), "thứ tự ảnh khác"
    C[b] = z["conf"]
sach = ~anh.leak.to_numpy()
q = np.quantile(anh.y_mean[sach], [1/3, 2/3])
nhom = np.where(anh.y_mean < q[0], "toi", np.where(anh.y_mean < q[1], "vua", "sang"))


def so(chon, ten):
    rng = np.random.default_rng(SEED)
    cs, cv = C["sage"][chon], C["vis"][chon]
    khoi = anh.khoi[chon].to_numpy()
    ms, ios = miou(cs.sum(0)); mv, iov = miou(cv.sum(0))
    ds, dl = [], []
    for i in boot_idx(khoi, rng):
        a, b = miou(cs[i].sum(0)), miou(cv[i].sum(0))
        ds.append(a[0] - b[0]); dl.append(a[1] - b[1])
    c = ci(ds)
    print(f"\n[{ten}] {chon.sum()} ảnh, {len(np.unique(khoi))} khối | vis {mv:.2f}  sage {ms:.2f}  "
          f"sage−vis {ms-mv:+.2f} [{c[0]:+.2f}; {c[1]:+.2f}]")
    return ios, iov, np.array(dl)


ios, iov, dl = so(sach, "273 ảnh")
lo, hi = np.nanpercentile(dl, [2.5, 97.5], axis=0)
print("\nIoU từng lớp (273 ảnh): lớp | vis | sage | Δ [CI95]")
for k, t in enumerate(TEN_LOP):
    if np.isnan(iov[k]):
        continue
    dau = "*" if lo[k] > 0 or hi[k] < 0 else " "
    print(f"  {t:14s} {iov[k]:6.1f} {ios[k]:6.1f} {ios[k]-iov[k]:+6.1f} [{lo[k]:+.1f}; {hi[k]:+.1f}] {dau}")
so(np.ones(len(anh), bool), "280 ảnh")
print(f"\nE3 thật – ngưỡng y_mean {q[0]:.1f} / {q[1]:.1f}")
for g in ["toi", "vua", "sang"]:
    i_s, i_v, _ = so(sach & (nhom == g), g)
    print(f"     person vis {i_v[7]:.1f} → sage {i_s[7]:.1f}; sign vis {i_v[4]:.1f} → sage {i_s[4]:.1f}")
