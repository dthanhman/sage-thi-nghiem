"""Chạy thử riêng thí nghiệm trong problem card của Mân (mục 10-11).

Dịch IR d = 0/2/5/10 px, thử 3 cách căn (none/phase/ecc), chạy SAGE Student, chấm so với IR gốc.
Metric ở đây là bản TẠM, không phải code chính thức của Lộc lớn.
"""
import argparse
import csv
import time
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE / "SAGE_IVIF"))
from model_sub.model import Network  # noqa: E402

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # trái, phải, lên, xuống
CROP = 15
WEIGHTS = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])


def huong(image_id):
    return DIRS[int(np.random.default_rng([42, int(image_id)]).integers(4))]


def dich(img, dx, dy):
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def canh(img):
    g = cv2.GaussianBlur(img, (0, 0), 1.0).astype(np.float32) / 255
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    m = cv2.magnitude(gx, gy)
    return m / (m.max() + 1e-8)


def ap_dich(img, sx, sy):
    """Dịch ảnh một lượng (sx, sy) px, có thể lẻ."""
    m = np.float32([[1, 0, sx], [0, 1, sy]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def can_phase(vis_e, ir_e, win):
    (x, y), _ = cv2.phaseCorrelate(np.float64(ir_e * win), np.float64(vis_e * win))
    return float(x), float(y), True  # lượng cần dịch IR để về VIS


def can_ecc(vis_e, ir_e):
    w = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    try:
        _, w = cv2.findTransformECC(vis_e, ir_e, w, cv2.MOTION_TRANSLATION, crit, None, 5)
    except cv2.error:
        return 0.0, 0.0, False
    return -float(w[0, 2]), -float(w[1, 2]), True


def gauss(device):
    x = torch.arange(11, dtype=torch.float32, device=device) - 5
    g = torch.exp(-x ** 2 / (2 * 1.5 ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None]


def ssim(a, b, k):
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ma, mb = F.conv2d(a, k), F.conv2d(b, k)
    va = F.conv2d(a * a, k) - ma ** 2
    vb = F.conv2d(b * b, k) - mb ** 2
    cov = F.conv2d(a * b, k) - ma * mb
    cs = (2 * cov + c2) / (va + vb + c2)
    s = (2 * ma * mb + c1) / (ma ** 2 + mb ** 2 + c1) * cs
    return s.mean(), cs.mean()


def ms_ssim(a, b, k):
    w = WEIGHTS.to(a.device)
    vals = []
    for i in range(5):
        s, cs = ssim(a, b, k)
        vals.append((s if i == 4 else cs).clamp(min=0))
        if i < 4:
            a, b = F.avg_pool2d(a, 2), F.avg_pool2d(b, 2)
    return float(torch.prod(torch.stack(vals) ** w))


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (torch.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


def cham(f, vis, ir, k):
    c = CROP
    f, vis, ir = f[..., c:-c, c:-c], vis[..., c:-c, c:-c], ir[..., c:-c, c:-c]
    mv, mi = ms_ssim(f, vis, k), ms_ssim(f, ir, k)
    return {"ms_ssim_vis": mv, "ms_ssim_ir": mi, "ms_ssim_tong": mv + mi,
            "scd": corr(f - ir, vis) + corr(f - vis, ir)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="ket_qua.csv")
    a = ap.parse_args()
    root = HERE.parent / "xulianh" / "data" / "raw" / "FMB"
    df = pd.read_csv(HERE / "dev.csv", dtype={"image_id": str})
    if a.limit:
        df = df.iloc[np.linspace(0, len(df) - 1, a.limit).astype(int)]
    dev = torch.device("cuda")
    net = Network()
    state = torch.load(HERE / "SAGE_IVIF/checkpoints/checkpoints.pt", map_location="cpu",
                       weights_only=True)
    net.load_state_dict(state, strict=True)
    net.to(dev).eval()
    k = gauss(dev)
    win = cv2.createHanningWindow((800, 600), cv2.CV_32F)
    cot = ["image_id", "vis_mean", "d", "cach", "true_sx", "true_sy", "est_sx", "est_sy", "ok",
           "ms_ssim_vis", "ms_ssim_ir", "ms_ssim_tong", "scd", "align_ms", "net_ms"]
    out = HERE / a.out
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, cot)
        w.writeheader()
        t0 = time.time()
        for n, r in enumerate(df.itertuples(), 1):
            bgr = cv2.imread(str(root / r.vis_path))
            ir0 = cv2.cvtColor(cv2.imread(str(root / r.ir_path)), cv2.COLOR_BGR2GRAY)
            y = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
            ty = torch.from_numpy(y).float().div(255)[None, None].to(dev)
            tir0 = torch.from_numpy(ir0).float().div(255)[None, None].to(dev)
            ve = canh(y)
            hx, hy = huong(r.image_id)
            for d in (0, 2, 5, 10):
                ird = dich(ir0, hx * d, hy * d)
                ie = canh(ird)
                for cach in ("none", "phase", "ecc"):
                    t1 = time.perf_counter()
                    if cach == "none":
                        sx, sy, ok, ir_in = 0.0, 0.0, True, ird
                    else:
                        sx, sy, ok = can_phase(ve, ie, win) if cach == "phase" else can_ecc(ve, ie)
                        ir_in = ap_dich(ird, sx, sy) if ok else ird
                    t2 = time.perf_counter()
                    tir = torch.from_numpy(ir_in).float().div(255)[None, None].to(dev)
                    with torch.inference_mode():
                        fy, _ = net(ty, tir)
                        fy = fy.clamp(0, 1)
                        m = cham(fy, ty, tir0, k)  # luôn so với IR gốc
                    t3 = time.perf_counter()
                    w.writerow({"image_id": r.image_id, "vis_mean": r_mean(y), "d": d, "cach": cach,
                                "true_sx": -hx * d, "true_sy": -hy * d, "est_sx": round(sx, 3),
                                "est_sy": round(sy, 3), "ok": ok, **{x: round(v, 6) for x, v in m.items()},
                                "align_ms": round((t2 - t1) * 1000, 1),
                                "net_ms": round((t3 - t2) * 1000, 1)})
            fh.flush()
            print(f"{n}/{len(df)} {r.image_id} {time.time() - t0:.0f}s", flush=True)


def r_mean(y):
    return round(float(y.mean()), 2)


if __name__ == "__main__":
    main()
