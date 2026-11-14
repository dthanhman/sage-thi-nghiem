"""E3: proxy mIoU theo nhóm độ sáng (xem README)."""
import numpy as np, pandas as pd
from tong_hop import miou, boot_idx, ci, HERE, SEED
anh = pd.read_csv(HERE / "kq_anh.csv", dtype={"image_id": str})
z = np.load(HERE / "kq_seg.npz", allow_pickle=True); BT = list(z["bien_the"]); conf = z["conf"]
sach = ~anh.leak.to_numpy()
q = np.quantile(anh.y_mean[sach], [1/3, 2/3])
nhom = np.where(anh.y_mean < q[0], "toi", np.where(anh.y_mean < q[1], "vua", "sang"))
print(f"Ngưỡng y_mean: {q[0]:.1f} / {q[1]:.1f}\n")
rng = np.random.default_rng(SEED); kq = {}
for g in ["toi", "vua", "sang"]:
    chon = sach & (nhom == g); cf = conf[:, chon]; khoi = anh.khoi[chon].to_numpy()
    idxs = list(boot_idx(khoi, rng))
    m = {b: miou(cf[BT.index(b)].sum(0))[0] for b in ["vis", "ir", "tb", "sage"]}
    print(f"[{g}] {chon.sum()} ảnh, {len(np.unique(khoi))} khối | " + "  ".join(f"{b} {v:.1f}" for b, v in m.items()))
    for a, b in [("sage", "vis"), ("sage", "tb"), ("tb", "vis")]:
        ja, jb = BT.index(a), BT.index(b)
        f = lambda i: miou(cf[ja, i].sum(0))[0] - miou(cf[jb, i].sum(0))[0]
        d, c = f(np.arange(cf.shape[1])), ci([f(i) for i in idxs]); kq[(g, a, b)] = (d, c)
        print(f"     {a} − {b}: {d:+.2f} [{c[0]:+.2f}; {c[1]:+.2f}]")
    # IoU lớp person, sign
    io = {b: miou(cf[BT.index(b)].sum(0))[1] for b in ["vis", "sage"]}
    print(f"     IoU person vis {io['vis'][8]:.1f} → sage {io['sage'][8]:.1f}; sign vis {io['vis'][5]:.1f} → sage {io['sage'][5]:.1f}")
t, s = kq[("toi", "sage", "vis")][1], kq[("sang", "sage", "vis")][1]
kl = "ĐẠT" if t[0] > 0 and s[1] < 0 else ("ĐẠT MỘT PHẦN" if t[0] > 0 and s[0] <= 0 <= s[1] else "TRƯỢT")
print(f"\nKết luận theo tiêu chí: {kl}")
