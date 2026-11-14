"""Chấm lại problem A bằng metric của PR #6 (Lộc lớn), chưa merge, lấy từ worktree ../pr6_eval.

Chỉ đo E(d) = L_SAGE(d) - L_TB(d), d = 0/5/10, không căn. Cùng hướng dịch, cùng cắt 15 px như run_thu.py.
Ảnh ghép làm tròn về uint8 như output thật.
  --tap dev   : 175 ảnh dev, để đối chiếu với metric tạm
  --tap train : 5 khối lớn nhất trong train.csv (lần kiểm tra lại card đã hẹn ở mục 12)
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PR6 = HERE.parent / "pr6_eval"
sys.path.insert(0, str(PR6))
from src.evaluation.metrics import fusion_ms_ssim, sum_of_correlation_differences  # noqa: E402

C = 15
DS = (0, 5, 10)


def cham(job):
    iid, khoi, d, cach, f, y, ir0 = job
    f, y, ir0 = (a[C:-C, C:-C].astype(np.float64) for a in (f, y, ir0))
    m = fusion_ms_ssim(f, ir0, y, aggregation="sum")
    return {"image_id": iid, "khoi": khoi, "d": d, "cach": cach,
            "ms_ir": m["MS_SSIM_IR"], "ms_vis": m["MS_SSIM_VIS"], "ms_sum": m["MS_SSIM_sum"],
            "ms_mean": m["MS_SSIM_mean"], "scd": sum_of_correlation_differences(f, ir0, y)}


def chon(tap):
    man = PR6 / "data/manifests"
    df = pd.read_csv(man / f"{tap}.csv", dtype={"image_id": str})
    i = df.image_id.astype(int)
    df["khoi"] = (i.diff() != 1).cumsum()
    if tap == "train":
        top = df.khoi.value_counts().head(5).index
        df = df[df.khoi.isin(top)]
    return df


def jobs(df, net, torch, R):
    root = HERE.parent / "xulianh/data/raw/FMB"
    t = lambda a: torch.from_numpy(a).float().div(255)[None, None].cuda()
    u8 = lambda x: (x.clamp(0, 1) * 255).round().byte().cpu().numpy()[0, 0]
    for r in df.itertuples():
        y = cv2.cvtColor(cv2.imread(str(root / r.vis_path)), cv2.COLOR_BGR2YCrCb)[:, :, 0]
        ir0 = cv2.cvtColor(cv2.imread(str(root / r.ir_path)), cv2.COLOR_BGR2GRAY)
        hx, hy = R.huong(r.image_id)
        for d in DS:
            ird = R.dich(ir0, hx * d, hy * d)
            with torch.inference_mode():
                fs = u8(net(t(y), t(ird))[0])
            tb = ((y.astype(np.float64) + ird) / 2).round().astype(np.uint8)
            yield (r.image_id, int(r.khoi), d, "sage", fs, y, ir0)
            yield (r.image_id, int(r.khoi), d, "tb", tb, y, ir0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tap", choices=["dev", "train"], required=True)
    a = ap.parse_args()
    import torch
    import run_thu as R
    net = R.Network()
    net.load_state_dict(torch.load(HERE / "SAGE_IVIF/checkpoints/checkpoints.pt",
                                   map_location="cpu", weights_only=True), strict=True)
    net.cuda().eval()
    df = chon(a.tap)
    print(a.tap, len(df), "ảnh,", df.khoi.nunique(), "khối", flush=True)
    out = []
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 2)) as ex:
        for n, o in enumerate(ex.map(cham, jobs(df, net, torch, R), chunksize=4), 1):
            out.append(o)
            if n % 300 == 0:
                print(n, "/", len(df) * len(DS) * 2, flush=True)
    pd.DataFrame(out).to_csv(HERE / f"kq_pr6_{a.tap}.csv", index=False)


if __name__ == "__main__":
    main()
