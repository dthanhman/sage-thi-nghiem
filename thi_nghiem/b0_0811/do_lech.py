"""Đo lệch IR–VIS theo ĐÚNG cách audit FMB (cell 17 của FMB_audit_gop_final.ipynb):
bản đồ cạnh Sobel chuẩn hóa × cửa sổ Hanning → cv2.phaseCorrelate → (dx, dy), độ tin cậy.
Chỉ đo dịch chuyển toàn ảnh (không đo xoay, thị sai).

  python do_lech.py --ten fmb  --vis <thư mục VIS> --ir <thư mục IR> --n 300
  python do_lech.py --ten m3fd --vis <...> --ir <...>
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DUOI = {".png", ".jpg", ".jpeg", ".bmp"}


def to_gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def edge(img):
    g = cv2.GaussianBlur(to_gray(img).astype(np.float32), (5, 5), 0)
    e = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return (e - e.mean()) / (e.std() + 1e-6)


def tk(d):
    return pd.Series({"số cặp": len(d), "trung vị dx": d.dx.median(), "trung vị dy": d.dy.median(),
                      "trung vị |lệch|": d.lech_px.median(), "P90 |lệch|": d.lech_px.quantile(0.9),
                      "% > 3 px": 100 * (d.lech_px > 3).mean(), "% > 10 px": 100 * (d.lech_px > 10).mean(),
                      "độ sáng VIS TB": d.vis_mean.mean()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ten", required=True)
    ap.add_argument("--vis", required=True)
    ap.add_argument("--ir", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    vis = {p.stem: p for p in Path(a.vis).iterdir() if p.suffix.lower() in DUOI}
    ir = {p.stem: p for p in Path(a.ir).iterdir() if p.suffix.lower() in DUOI}
    ten = sorted(vis.keys() & ir.keys())
    print(f"{a.ten}: {len(vis)} VIS, {len(ir)} IR, {len(ten)} cặp khớp tên")
    ten = random.Random(a.seed).sample(ten, min(a.n, len(ten)))

    rows = []
    for t in ten:
        v, i = cv2.imread(str(vis[t])), cv2.imread(str(ir[t]))
        if v is None or i is None or v.shape[:2] != i.shape[:2]:
            print("bỏ", t, None if v is None else v.shape, None if i is None else i.shape)
            continue
        win = cv2.createHanningWindow(v.shape[1::-1], cv2.CV_32F)
        (dx, dy), resp = cv2.phaseCorrelate(edge(v) * win, edge(i) * win)
        rows.append({"name": t, "dx": dx, "dy": dy, "do_tin_cay": resp, "vis_mean": float(to_gray(v).mean()),
                     "h": v.shape[0], "w": v.shape[1]})
    df = pd.DataFrame(rows)
    df["lech_px"] = np.hypot(df.dx, df.dy)
    df["tin_cay_cao"] = df.do_tin_cay >= df.do_tin_cay.median()
    df.to_csv(HERE / f"lech_{a.ten}.csv", index=False)
    bang = pd.DataFrame({"tất cả": tk(df), "tin cậy cao": tk(df[df.tin_cay_cao]),
                         "tin cậy thấp": tk(df[~df.tin_cay_cao])}).round(2)
    print("kích thước:", df.groupby(["w", "h"]).size().to_dict())
    print(bang.to_string())
    print("Tương quan tin cậy – độ sáng VIS:", round(df[["do_tin_cay", "vis_mean"]].corr().iloc[0, 1], 2))


if __name__ == "__main__":
    main()
