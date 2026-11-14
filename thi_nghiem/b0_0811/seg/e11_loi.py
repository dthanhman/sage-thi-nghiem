"""E11 – soi lỗi sage so với vis (SegFormer-B0 seed 0, 273 ảnh), chỉ đọc pred đã lưu, không cần GPU.
(1) chênh theo ảnh: liên hệ với độ sáng VIS, độ tương phản IR; ảnh sage thua/hơn nhiều nhất.
(2) kiểu nhầm lẫn: cặp (GT -> dự đoán) tăng/giảm nhiều nhất khi đổi vis -> sage.
(3) lớp person theo kích thước người (thành phần liên thông trong GT): tỉ lệ pixel người được đoán đúng.
Đây là thí nghiệm khám phá (sinh giả thuyết), không có tiêu chí đạt/trượt."""
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from chung import FMB, MAN, IGNORE, TEN_LOP, doc_nhan, miou
from tong_hop import boot_idx, ci, SEED

anh = pd.read_csv(HERE.parent / "kq_anh.csv", dtype={"image_id": str})
te = pd.read_csv(MAN / "test.csv", dtype={"image_id": str}).set_index("image_id")
C = {b: np.load(HERE / "runs" / f"{b}_b0_s0" / "test_nham.npz")["conf"] for b in ("sage", "vis")}
sach = ~anh.leak.to_numpy()

# (1) chênh theo ảnh (mIoU từng ảnh trên các lớp có trong ảnh đó)
d = np.array([miou(C["sage"][i])[0] - miou(C["vis"][i])[0] for i in range(len(anh))])
ir_std, ir_ct = [], []
for iid in anh.image_id:
    ir = cv2.imread(str(FMB / te.loc[iid, "ir_path"]), cv2.IMREAD_GRAYSCALE).astype(np.float64)
    ir_std.append(ir.std()); ir_ct.append(np.percentile(ir, 99) - np.percentile(ir, 1))
anh["d"], anh["ir_std"], anh["ir_ct"] = d, ir_std, ir_ct
a = anh[sach]
print("(1) mIoU từng ảnh, sage − vis: trung vị %+.2f, sage hơn ở %d/%d ảnh" % (np.median(a.d), (a.d > 0).sum(), len(a)))
for c in ["y_mean", "ir_mean", "ir_std", "ir_ct"]:
    r, p = spearmanr(a[c], a.d); print(f"    Spearman(d, {c:7s}) = {r:+.2f} (p = {p:.3f})")
print("    10 ảnh sage THUA nhiều nhất:")
print(a.nsmallest(10, "d")[["image_id", "khoi", "y_mean", "ir_std", "d"]].round(1).to_string(index=False))
print("    10 ảnh sage HƠN nhiều nhất:")
print(a.nlargest(10, "d")[["image_id", "khoi", "y_mean", "ir_std", "d"]].round(1).to_string(index=False))
print("    Theo khối (cảnh): mIoU khối sage − vis")
for k, g in a.groupby("khoi"):
    i = g.index.to_numpy()
    print(f"      khối {k:2d}: {len(i):3d} ảnh, sáng {g.y_mean.mean():5.1f}, ir_std {g.ir_std.mean():4.1f} | "
          f"{miou(C['sage'][i].sum(0))[0] - miou(C['vis'][i].sum(0))[0]:+6.2f}")

# (2) kiểu nhầm lẫn (tỉ lệ pixel GT lớp i bị đoán thành j)
cs, cv = C["sage"][sach].sum(0).astype(float), C["vis"][sach].sum(0).astype(float)
rs, rv = cs / cs.sum(1, keepdims=True).clip(1), cv / cv.sum(1, keepdims=True).clip(1)
dd = (rs - rv) * 100; np.fill_diagonal(dd, 0)
o = np.dstack(np.unravel_index(np.argsort(dd.ravel()), dd.shape))[0]
print("\n(2) Nhầm lẫn TĂNG nhiều nhất khi dùng sage (điểm % của pixel GT):")
for i, j in o[::-1][:8]:
    print(f"    {TEN_LOP[i]:14s} -> {TEN_LOP[j]:14s} {rv[i, j]*100:5.1f}% -> {rs[i, j]*100:5.1f}%")
print("    Nhầm lẫn GIẢM nhiều nhất:")
for i, j in o[:8]:
    print(f"    {TEN_LOP[i]:14s} -> {TEN_LOP[j]:14s} {rv[i, j]*100:5.1f}% -> {rs[i, j]*100:5.1f}%")

# (3) person theo kích thước
P = TEN_LOP.index("person"); rows = []
for i in np.flatnonzero(sach):
    iid = anh.image_id[i]
    gt = doc_nhan(FMB / te.loc[iid, "label_path"])
    n, lab, st, _ = cv2.connectedComponentsWithStats((gt == P).astype(np.uint8), connectivity=8)
    if n <= 1:
        continue
    pr = {b: cv2.imread(str(HERE / "runs" / f"{b}_b0_s0" / "pred" / f"{iid}.png"), cv2.IMREAD_UNCHANGED).astype(int) - 1
          for b in ("sage", "vis")}
    for k in range(1, n):
        m = lab == k; s = st[k, cv2.CC_STAT_AREA]
        if s < 20:
            continue
        rows.append({"khoi": anh.khoi[i], "y_mean": anh.y_mean[i], "dt": s, "cao": st[k, cv2.CC_STAT_HEIGHT],
                     **{b: (pr[b][m] == P).mean() for b in pr}})
ng = pd.DataFrame(rows)
ng["co"] = pd.cut(ng.cao, [0, 40, 100, 1000], labels=["nhỏ (<40px)", "vừa (40–100)", "lớn (>100)"])
ng["toi"] = np.where(ng.y_mean < np.quantile(anh.y_mean[sach], 1/3), "tối", "vừa/sáng")
print(f"\n(3) Người trong GT: {len(ng)} người (diện tích ≥ 20 px). Tỉ lệ pixel người đoán đúng (recall):")
print(ng.groupby(["co", "toi"], observed=True)[["vis", "sage"]].agg(["mean", "count"]).round(3).to_string())
rng = np.random.default_rng(SEED)
for c, g in ng.groupby("co", observed=True):
    v = (g.sage - g.vis).to_numpy(); kh = g.khoi.to_numpy()
    bs = [v[i].mean() for i in boot_idx(kh, rng)]
    print(f"    {c}: sage − vis = {v.mean()*100:+.1f} điểm [{ci(bs)[0]*100:+.1f}; {ci(bs)[1]*100:+.1f}]")
