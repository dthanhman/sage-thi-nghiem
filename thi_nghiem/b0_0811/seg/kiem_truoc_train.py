"""Kiểm tra trước khi train SegFormer."""
import sys
import cv2, numpy as np, pandas as pd, torch, random
from chung import ANH, FMB, MAN, GOC, IGNORE, NC, doc_nhan, thu_muc
from train_seg import FMBSeg, tao_model
ok = True
def kiem(dk, msg):
    global ok
    print(("OK   " if dk else "LỖI ") + msg); ok &= bool(dk)

d = {s: pd.read_csv(MAN / f"{s}.csv", dtype={"image_id": str}) for s in ("train", "dev", "test")}
key = {s: set(thu_muc(r) + "/" + r.image_id for r in d[s].itertuples()) for s in d}
kiem(len(d["train"]) == 1045 and len(d["dev"]) == 175 and len(d["test"]) == 280, f"số ảnh {[len(v) for v in d.values()]}")
kiem(not (key["train"] & key["dev"]) and not (key["train"] & key["test"]) and not (key["dev"] & key["test"]), "3 tập không giao nhau")
kiem(all(len(v) == len(d[s]) for s, v in key.items()), "không trùng id trong từng tập")
for bt in ("vis", "sage"):
    n = sum((ANH / bt / thu_muc(r) / f"{r.image_id}.png").exists() for df in d.values() for r in df.itertuples())
    kiem(n == 1500, f"đủ ảnh {bt}: {n}/1500")

# ảnh vis xuất ra phải giống hệt VIS gốc
r = d["test"].iloc[5]
kiem(np.array_equal(cv2.imread(str(ANH / "vis/test" / f"{r.image_id}.png")), cv2.imread(str(FMB / r.vis_path))), "ảnh vis = VIS gốc")
# ảnh sage: Y phải khớp chạy lại SAGE, CrCb khớp VIS (sai số làm tròn)
sys.path.insert(0, str(GOC / "thu_viec3/SAGE_IVIF"))
from model_sub.model import Network
net = Network(); net.load_state_dict(torch.load(GOC / "thu_viec3/SAGE_IVIF/checkpoints/checkpoints.pt", weights_only=True)); net.cuda().eval()
for r in [d["train"].iloc[100], d["dev"].iloc[3], d["test"].iloc[200]]:
    vis = cv2.imread(str(FMB / r.vis_path)); ycc = cv2.cvtColor(vis, cv2.COLOR_BGR2YCrCb)
    ir = cv2.cvtColor(cv2.imread(str(FMB / r.ir_path)), cv2.COLOR_BGR2GRAY)
    t = lambda z: torch.from_numpy(z).float().div(255)[None, None].cuda()
    with torch.inference_mode():
        ys = (net(t(ycc[:, :, 0]), t(ir))[0].clamp(0, 1) * 255).round().byte().cpu().numpy()[0, 0]
    f = cv2.cvtColor(cv2.imread(str(ANH / "sage" / thu_muc(r) / f"{r.image_id}.png")), cv2.COLOR_BGR2YCrCb)
    dy = np.abs(f[:, :, 0].astype(int) - ys).mean(); dc = np.abs(f[:, :, 1:].astype(int) - ycc[:, :, 1:]).mean()
    kiem(dy < 1 and dc < 1.5, f"ảnh sage {r.split}/{r.image_id}: lệch Y {dy:.2f}, lệch CrCb {dc:.2f} (do làm tròn)")

# nhãn: 0 -> 255, 1..14 -> 0..13
lab = cv2.imread(str(FMB / d["train"].iloc[0].label_path), cv2.IMREAD_UNCHANGED); m = doc_nhan(FMB / d["train"].iloc[0].label_path)
kiem(np.array_equal(m[lab == 0], np.full((lab == 0).sum(), IGNORE)) and np.array_equal(m[lab > 0], lab[lab > 0] - 1), "đổi nhãn đúng")
# augmentation: ảnh và nhãn biến đổi cùng nhau -> kiểm bằng cách đưa nhãn vào chỗ ảnh
random.seed(1)
ds = FMBSeg("train", "vis", True)
x, y = ds[0]
kiem(x.shape == (3, 512, 512) and y.shape == (512, 512) and y.dtype == torch.int64, f"kích thước mẫu {tuple(x.shape)} {tuple(y.shape)}")
u = set(torch.unique(y).tolist()); kiem(u <= set(range(NC)) | {IGNORE}, f"giá trị nhãn sau aug {sorted(u)}")
class Gia(FMBSeg):  # thay ảnh bằng chính nhãn để xem có khớp sau aug không
    def __getitem__(self, i):
        global cv2
        im0 = cv2.imread; lab = cv2.imread(str(FMB / self.df.iloc[i].label_path), cv2.IMREAD_UNCHANGED)
        cv2.imread = lambda p, *a: np.dstack([lab] * 3) if not a else im0(p, *a)
        try: return super().__getitem__(i)
        finally: cv2.imread = im0
import train_seg
for k in range(5):
    random.seed(k); gx, gy = Gia("train", "vis", True)[k]
    from chung import MEAN, STD
    im = np.round((gx.numpy()[0] * STD[0] + MEAN[0]) * 255).astype(int) - 1
    im[im < 0] = IGNORE
    vung = gy.numpy() != IGNORE
    khop = (im[vung] == gy.numpy()[vung]).mean()
    kiem(khop > 0.97, f"aug mẫu {k}: ảnh và nhãn khớp {khop*100:.1f}% (biên nội suy làm lệch chút)")

# bộ nhớ GPU 1 bước train thật, batch 8
model = tao_model("b0").cuda().train(); opt = torch.optim.AdamW(model.parameters()); torch.cuda.reset_peak_memory_stats()
xb = torch.randn(8, 3, 512, 512).cuda(); yb = torch.randint(0, NC, (8, 512, 512)).cuda()
for _ in range(2):
    with torch.autocast("cuda", dtype=torch.float16):
        lg = model(pixel_values=xb).logits
    loss = torch.nn.functional.cross_entropy(torch.nn.functional.interpolate(lg.float(), size=(512, 512), mode="bilinear"), yb)
    opt.zero_grad(); loss.backward(); opt.step()
pk = torch.cuda.max_memory_allocated() / 2**30
kiem(pk < 5.0, f"GPU đỉnh {pk:.2f} GB / 6 GB")
print("\nTẤT CẢ OK" if ok else "\nCÓ LỖI")
