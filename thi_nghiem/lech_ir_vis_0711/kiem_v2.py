"""Các phép kiểm tra để thiết kế lại cách chấm trong card (chỉ dùng dev)."""
import cv2
import numpy as np
import pandas as pd
import torch

import run_thu as R

df = pd.read_csv("ket_qua.csv", dtype={"image_id": str})
dev = pd.read_csv("dev.csv", dtype={"image_id": str}).set_index("image_id")
root = R.HERE.parent / "xulianh/data/raw/FMB"
BANG = [(52, 84), (91, 120), (172, 206), (591, 630), (1383, 1419)]
khoi = lambda i: next(k for k, (a, b) in enumerate(BANG) if a <= int(i) <= b)
M = 12  # bỏ mép khi quét
S = 8


def bien_nhan(lb):
    b = np.zeros(lb.shape, bool)
    b[:, :-1] |= lb[:, :-1] != lb[:, 1:]
    b[:-1, :] |= lb[:-1, :] != lb[1:, :]
    return b


def diem_bien(e, b, sx, sy):
    """Độ mạnh cạnh của ảnh e (đã dịch sx, sy) đọc trên biên nhãn b."""
    m = np.float32([[1, 0, sx], [0, 1, sy]])
    s = cv2.warpAffine(e, m, (e.shape[1], e.shape[0]), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
    return float(s[M + S:-(M + S), M + S:-(M + S)][b[M + S:-(M + S), M + S:-(M + S)]].mean())


net = R.Network()
net.load_state_dict(torch.load("SAGE_IVIF/checkpoints/checkpoints.pt", map_location="cpu",
                               weights_only=True))
net.cuda().eval()
k = R.gauss("cuda")
t = lambda a: torch.from_numpy(a).float().div(255)[None, None].cuda()


def ghep(y, ir):
    with torch.inference_mode():
        return net(t(y), t(ir))[0].clamp(0, 1)


rows, scan = [], []
for iid, r in dev.iterrows():
    y = cv2.cvtColor(cv2.imread(str(root / r.vis_path)), cv2.COLOR_BGR2YCrCb)[:, :, 0]
    ir = cv2.cvtColor(cv2.imread(str(root / r.ir_path)), cv2.COLOR_BGR2GRAY)
    lb = cv2.imread(str(root / r.label_path), cv2.IMREAD_GRAYSCALE)
    b = bien_nhan(lb)
    ev, ei = R.canh(y), R.canh(ir)
    best = {}
    for ten, e in (("vis", ev), ("ir", ei)):
        tot = {(sx, sy): diem_bien(e, b, sx, sy) for sx in range(-S, S + 1) for sy in range(-S, S + 1)}
        best[ten] = max(tot, key=tot.get)
        # độ nhọn đỉnh: đỉnh so với trung vị vùng quét
        best[ten + "_tp"] = (max(tot.values()) - np.median(list(tot.values()))) / np.median(list(tot.values()))
    q = df[(df.image_id == iid) & (df.d == 0)].set_index("cach")
    o = {"image_id": iid, "khoi": khoi(iid), "vis_mean": float(y.mean()),
         "vis_bx": best["vis"][0], "vis_by": best["vis"][1], "vis_tp": best["vis_tp"],
         "ir_bx": best["ir"][0], "ir_by": best["ir"][1], "ir_tp": best["ir_tp"],
         "b_goc": diem_bien(ei, b, 0, 0), "b_vis": diem_bien(ev, b, 0, 0)}
    tir0, ty = t(ir), t(y)
    fg = ghep(y, ir)
    goc = R.cham(fg, ty, tir0, k)
    o["goc_tong"] = goc["ms_ssim_tong"]
    for c in ("phase", "ecc"):
        sx, sy = q.loc[c, "est_sx"], q.loc[c, "est_sy"]
        o[f"{c}_sx"], o[f"{c}_sy"] = sx, sy
        o[f"b_{c}"] = diem_bien(ei, b, sx, sy)
        o[f"b_{c}_nguoc"] = diem_bien(ei, b, -sx, -sy)
        # làm tròn px: không nội suy
        rx, ry = int(round(sx)), int(round(sy))
        f_r = ghep(y, R.dich(ir, rx, ry))
        o[f"{c}_tron_tong"] = R.cham(f_r, ty, tir0, k)["ms_ssim_tong"]
    # chỉ phần lẻ: dịch dưới 1 px, gần như chỉ còn tác dụng làm mờ của nội suy
    sx, sy = q.loc["ecc", "est_sx"], q.loc["ecc", "est_sy"]
    fx, fy = sx - round(sx), sy - round(sy)
    o["ecc_le_tong"] = R.cham(ghep(y, R.ap_dich(ir, fx, fy)), ty, tir0, k)["ms_ssim_tong"]
    # đối chứng: ghép trung bình đơn giản, và điểm so với IR thật sự đưa vào
    hx, hy = R.huong(iid)
    o["tb_goc"] = R.cham((ty + tir0) / 2, ty, tir0, k)["ms_ssim_tong"]
    o["sage_vao_0"] = goc["ms_ssim_tong"]
    for d in (5, 10):
        ird = R.dich(ir, hx * d, hy * d)
        tird = t(ird)
        o[f"tb_goc_{d}"] = R.cham((ty + tird) / 2, ty, tir0, k)["ms_ssim_tong"]
        o[f"sage_vao_{d}"] = R.cham(ghep(y, ird), ty, tird, k)["ms_ssim_tong"]
    rows.append(o)
pd.DataFrame(rows).to_csv("kiem_v2.csv", index=False)
print("xong", len(rows))
