"""E12: kiểm nhanh RQ2 (thích nghi theo ảnh) bằng model B0 có sẵn, không train.
- Trần (oracle): mỗi ảnh chọn đầu vào cho mIoU từng ảnh cao nhất -> mức tối đa của "chọn theo ảnh".
- Quy tắc: Y_mean VIS < ngưỡng (tính trên TRAIN, không nhìn test) -> dùng đầu vào 'toi', ngược lại 'sang'.
Mỗi đầu vào dùng model SegFormer riêng của nó (đây là proxy cho ghép thích nghi, không phải phương pháp thật).
273 ảnh sạch, CI95 bootstrap theo khối id."""
import sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from chung import miou, FMB, MAN
from tong_hop import boot_idx, ci, SEED

anh = pd.read_csv(HERE.parent / "kq_anh.csv", dtype={"image_id": str})
sach = ~anh.leak.to_numpy()
khoi = anh.khoi[sach].to_numpy()
y = anh.y_mean[sach].to_numpy()

# ngưỡng tối/sáng từ train (1/3 tối nhất của train)
tr = pd.read_csv(MAN / "train.csv", dtype={"image_id": str})
ytr = np.array([cv2.cvtColor(cv2.imread(str(FMB / p)), cv2.COLOR_BGR2YCrCb)[:, :, 0].mean() for p in tr.vis_path])
NG = np.quantile(ytr, 1 / 3)
print(f"Ngưỡng tối (1/3 train): Y_mean < {NG:.1f}; test sạch có {(y < NG).sum()}/{len(y)} ảnh tối")

def nap(run):
    z = np.load(HERE / "runs" / run / "test_nham.npz")
    assert list(z["image_id"]) == list(anh.image_id), run
    return z["conf"][sach]

for s in (0, 1):
    C = {b: nap(f"{b}_b0_s{s}") for b in ("vis", "sage", "tb", "clahe")}
    pi = {b: np.array([np.nan_to_num(miou(c)[0], nan=-1) for c in C[b]]) for b in C}
    def tron(chon):  # chon[i] = tên đầu vào cho ảnh i
        return sum(C[chon[i]][i] for i in range(len(chon)))
    def oracle(ds):
        return [max(ds, key=lambda b: pi[b][i]) for i in range(len(y))]
    luat = {"toi tb / sang vis": np.where(y < NG, "tb", "vis"),
            "toi tb / sang sage": np.where(y < NG, "tb", "sage"),
            "toi sage / sang vis": np.where(y < NG, "sage", "vis")}
    pa = {"trần {vis,tb}": oracle(["vis", "tb"]), "trần {vis,sage,tb}": oracle(["vis", "sage", "tb"])} | luat
    print(f"\n#### SEED {s} ####  đơn lẻ: " + "  ".join(f"{b} {miou(C[b].sum(0))[0]:.2f}" for b in C))
    rng = np.random.default_rng(SEED); idxs = list(boot_idx(khoi, rng))
    for ten, chon in pa.items():
        chon = np.asarray(chon)
        m = miou(tron(chon).sum(0) if False else tron(chon))[0]
        d = [miou(sum(C[chon[i]][i] for i in ix))[0] - miou(C["tb"][ix].sum(0))[0] for ix in idxs]
        c = ci(d); sao = "*" if c[0] > 0 or c[1] < 0 else " "
        dem = {b: int((chon == b).sum()) for b in np.unique(chon)}
        print(f"  {ten:22s} {m:.2f} | so với tb {m - miou(C['tb'].sum(0))[0]:+.2f} [{c[0]:+.2f}; {c[1]:+.2f}]{sao} | {dem}")
