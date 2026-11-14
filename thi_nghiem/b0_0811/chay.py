"""E0 + E1 + E2 trên FMB test. Xem README.md cho tiêu chí.

Ra:
  kq_metric.csv : mỗi dòng = (ảnh, biến thể), metric ghép của PR #6
  kq_seg.npz    : ma trận nhầm lẫn 15x15 từng ảnh, từng biến thể (SegFormer zero-shot)
  kq_anh.csv    : thông tin ảnh (leak, khối, độ sáng Y trung bình của VIS)
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

GOC = Path(__file__).resolve().parent.parent
FMB = GOC / "xulianh/data/raw/FMB"
MAN = GOC / "xulianh/data/manifests/test.csv"
PR6 = GOC / "pr6_eval"
SAGE = GOC / "thu_viec3/SAGE_IVIF"
SEG = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
HERE = Path(__file__).resolve().parent

USM = (0.25, 0.5, 1.0)
BIEN_THE = ["vis", "ir", "tb", "sage"] + [f"sage_usm{a}" for a in USM]
NC = 15  # 0 = unlabeled/khác, 1..14 lớp FMB

# trainId Cityscapes 0..18 -> lớp FMB (0 = khác)
CS2FMB = np.array([1, 2, 3, 0, 0, 14, 4, 5, 6, 6, 7, 8, 8, 9, 10, 11, 0, 12, 13], dtype=np.int64)


def usm(y, a, sigma=1.5):
    yf = y.astype(np.float64)
    return np.clip(yf + a * (yf - cv2.GaussianBlur(yf, (0, 0), sigma)), 0, 255).round().astype(np.uint8)


def sang_bgr(y, crcb):
    return cv2.cvtColor(np.dstack([y, crcb]), cv2.COLOR_YCrCb2BGR)


# ---------------- metric (chạy ở process con) ----------------
def _init_metric():
    sys.path.insert(0, str(PR6))


def cham(job):
    from src.evaluation.metrics import (average_gradient, entropy, fusion_ms_ssim, q_abf,
                                        spatial_frequency, standard_deviation,
                                        sum_of_correlation_differences)
    from src.evaluation.iqa_metrics import brisque, niqe
    iid, bt, f, ir, y = job
    f, ir, y = (a.astype(np.float64) for a in (f, ir, y))
    ms = fusion_ms_ssim(f, ir, y, data_range=255.0, aggregation="sum")
    return {"image_id": iid, "bien_the": bt,
            "EN": entropy(f), "SD": standard_deviation(f), "SF": spatial_frequency(f),
            "AG": average_gradient(f), "SCD": sum_of_correlation_differences(f, ir, y),
            "MS_SSIM_sum": ms["MS_SSIM_sum"], "MS_SSIM_IR": ms["MS_SSIM_IR"],
            "MS_SSIM_VIS": ms["MS_SSIM_VIS"], "Q_ABF": q_abf(f, ir, y),
            "NIQE": niqe(f), "BRISQUE": brisque(f)}


# ---------------- SegFormer ----------------
class Seg:
    def __init__(self, torch):
        from transformers import SegformerForSemanticSegmentation
        self.t = torch
        self.m = SegformerForSemanticSegmentation.from_pretrained(SEG).cuda().eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
        self.map = torch.from_numpy(CS2FMB).cuda()

    def nham(self, bgr, gt):
        t = self.t
        x = t.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).cuda().permute(2, 0, 1)[None].float() / 255
        x = (x - self.mean) / self.std
        with t.inference_mode(), t.autocast("cuda", dtype=t.float16):
            lg = self.m(pixel_values=x).logits
        lg = t.nn.functional.interpolate(lg.float(), size=gt.shape, mode="bilinear", align_corners=False)
        p = self.map[lg.argmax(1)[0]]
        g = t.from_numpy(gt.astype(np.int64)).cuda()
        return t.bincount(g.flatten() * NC + p.flatten(), minlength=NC * NC).view(NC, NC).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="chỉ chạy n ảnh đầu (chạy thử)")
    a = ap.parse_args()

    import torch
    sys.path.insert(0, str(SAGE))
    from model_sub.model import Network
    net = Network()
    net.load_state_dict(torch.load(SAGE / "checkpoints/checkpoints.pt", map_location="cpu",
                                   weights_only=True), strict=True)
    net.cuda().eval()
    seg = Seg(torch)

    df = pd.read_csv(MAN, dtype={"image_id": str})
    df["khoi"] = (df.image_id.astype(int).diff() != 1).cumsum()
    if a.n:
        df = df.head(a.n)
    t2 = lambda z: torch.from_numpy(z).float().div(255)[None, None].cuda()

    conf = np.zeros((len(BIEN_THE), len(df), NC, NC), dtype=np.int64)
    anh, jobs = [], []
    t0 = time.time()
    for k, r in enumerate(df.itertuples()):
        vis = cv2.imread(str(FMB / r.vis_path))
        ycc = cv2.cvtColor(vis, cv2.COLOR_BGR2YCrCb)
        y, crcb = ycc[:, :, 0], ycc[:, :, 1:]
        ir = cv2.cvtColor(cv2.imread(str(FMB / r.ir_path)), cv2.COLOR_BGR2GRAY)
        gt = cv2.imread(str(FMB / r.label_path), cv2.IMREAD_UNCHANGED)
        assert gt.ndim == 2 and gt.max() < NC, r.image_id
        with torch.inference_mode():
            ys = (net(t2(y), t2(ir))[0].clamp(0, 1) * 255).round().byte().cpu().numpy()[0, 0]
        cac = {"vis": y, "ir": ir, "tb": ((y.astype(np.float64) + ir) / 2).round().astype(np.uint8),
               "sage": ys}
        for s in USM:
            cac[f"sage_usm{s}"] = usm(ys, s)
        for j, bt in enumerate(BIEN_THE):
            bgr = vis if bt == "vis" else (cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR) if bt == "ir"
                                           else sang_bgr(cac[bt], crcb))
            conf[j, k] = seg.nham(bgr, gt)
            jobs.append((r.image_id, bt, cac[bt], ir, y))
        anh.append({"image_id": r.image_id, "leak": bool(r.leak), "khoi": int(r.khoi),
                    "y_mean": float(y.mean()), "ir_mean": float(ir.mean())})
        if (k + 1) % 20 == 0:
            print(f"GPU {k + 1}/{len(df)}  {time.time() - t0:.0f}s", flush=True)

    pd.DataFrame(anh).to_csv(HERE / "kq_anh.csv", index=False)
    np.savez_compressed(HERE / "kq_seg.npz", conf=conf, bien_the=np.array(BIEN_THE),
                        image_id=df.image_id.to_numpy().astype(str))
    print("Xong SegFormer, chấm metric", len(jobs), "job", flush=True)

    out = []
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 2), initializer=_init_metric) as ex:
        for n, o in enumerate(ex.map(cham, jobs, chunksize=4), 1):
            out.append(o)
            if n % 140 == 0:
                print(f"metric {n}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(out).to_csv(HERE / "kq_metric.csv", index=False)
    print(f"Xong, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
