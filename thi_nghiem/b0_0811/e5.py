"""E5: mức trần oracle chọn ảnh theo từng ảnh (xem README)."""
import numpy as np, pandas as pd
from tong_hop import miou, boot_idx, ci, HERE, SEED
anh = pd.read_csv(HERE / "kq_anh.csv", dtype={"image_id": str})
z = np.load(HERE / "kq_seg.npz", allow_pickle=True); BT = list(z["bien_the"]); conf = z["conf"]
sach = ~anh.leak.to_numpy(); cf = conf[:, sach]; khoi = anh.khoi[sach].to_numpy()

def oracle(cands, idx):
    J = [BT.index(c) for c in cands]
    pi = np.array([[miou(cf[j, i])[0] for j in J] for i in idx])   # mIoU từng ảnh
    pick = np.nanargmax(np.nan_to_num(pi, nan=-1), axis=1)
    tong = sum(cf[J[p], i] for p, i in zip(pick, idx))
    return miou(tong)[0], pick

idx_all = np.arange(cf.shape[1])
# mIoU từng ảnh tính 1 lần cho nhanh
cache = {}
def per_img(j, i):
    k = (j, i)
    if k not in cache: cache[k] = miou(cf[j, i])[0]
    return cache[k]
def oracle_fast(cands, idx):
    J = [BT.index(c) for c in cands]
    pick = [max(range(len(J)), key=lambda p: np.nan_to_num(per_img(J[p], i), nan=-1)) for i in idx]
    return miou(sum(cf[J[p], i] for p, i in zip(pick, idx)))[0], pick

rng = np.random.default_rng(SEED); idxs = list(boot_idx(khoi, rng))
for cands in (["vis", "sage"], ["vis", "tb", "sage"]):
    o, pick = oracle_fast(cands, idx_all)
    base = {c: miou(cf[BT.index(c)].sum(0))[0] for c in cands}
    best = max(base, key=base.get)
    f = lambda idx: oracle_fast(cands, idx)[0] - max(miou(cf[BT.index(c), idx].sum(0))[0] for c in cands)
    d = f(idx_all); c = ci([f(i) for i in idxs[:500]])
    cnt = pd.Series([cands[p] for p in pick]).value_counts().to_dict()
    print(f"{cands}: oracle {o:.2f} | tốt nhất đơn lẻ {best} {base[best]:.2f} | lợi ích {d:+.2f} [{c[0]:+.2f}; {c[1]:+.2f}] | số lần chọn {cnt}")
    if cands == ["vis", "sage"]:
        kl = "ĐÁNG LÀM" if d >= 3 else ("BỎ" if d < 1.5 else "YẾU")
        y = anh.y_mean[sach].to_numpy(); ps = np.array(pick) == 1
        print(f"   ảnh chọn sage có y_mean TB {y[ps].mean():.1f}, ảnh chọn vis {y[~ps].mean():.1f}")
        print(f"   Kết luận theo tiêu chí: {kl}")
