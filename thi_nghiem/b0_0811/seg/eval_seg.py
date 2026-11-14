"""Bước 3: chấm checkpoint tốt nhất (chọn bằng dev) trên FMB test. Chỉ chạy SAU khi đã train xong.

Ra (trong thư mục run):
  test_nham.npz   : ma trận nhầm lẫn 14x14 từng ảnh (để bootstrap, chia theo điều kiện)
  test_ket_qua.json, test_iou_lop.csv : mIoU trên 280 và 273 ảnh, IoU từng lớp
  pred/{id}.png   : nhãn dự đoán (0 = unlabeled, 1..14 như FMB) nếu có --luu_anh

Ví dụ: python eval_seg.py --run runs/sage_b0_s0 --luu_anh
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from chung import ANH, FMB, MAN, NC, TEN_LOP, doc_nhan, miou, nham, thu_muc
from train_seg import FMBSeg, du_doan, tao_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--luu_anh", action="store_true")
    a = ap.parse_args()
    run = Path(a.run)
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))

    model = tao_model(cfg["model"])
    model.load_state_dict(torch.load(run / a.ckpt, map_location="cpu", weights_only=True))
    model.cuda().eval()

    ds = FMBSeg("test", cfg["bien_the"], False)
    df = ds.df
    conf = np.zeros((len(df), NC, NC), np.int64)
    if a.luu_anh:
        (run / "pred").mkdir(exist_ok=True)
    for i in range(len(df)):
        x, y = ds[i]
        p = du_doan(model, x[None].cuda()).argmax(1)[0].cpu().numpy()
        conf[i] = nham(y.numpy(), p)
        if a.luu_anh:
            cv2.imwrite(str(run / "pred" / f"{df.image_id.iloc[i]}.png"), (p + 1).astype(np.uint8))

    np.savez_compressed(run / "test_nham.npz", conf=conf, image_id=df.image_id.to_numpy().astype(str),
                        leak=df.leak.to_numpy())
    kq, bang = {"bien_the": cfg["bien_the"], "model": cfg["model"], "ckpt": a.ckpt,
                "best_dev_miou": cfg.get("best_dev_miou")}, {}
    for ten, chon in [("280", np.ones(len(df), bool)), ("273", ~df.leak.to_numpy())]:
        c = conf[chon].sum(0)
        m, iou = miou(c)
        kq[f"mIoU_{ten}"] = m
        kq[f"so_lop_{ten}"] = int((c.sum(1) > 0).sum())
        kq[f"pixel_acc_{ten}"] = float(np.diag(c).sum() / c.sum() * 100)
        bang[f"IoU_{ten}"] = iou
    (run / "test_ket_qua.json").write_text(json.dumps(kq, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(bang, index=TEN_LOP).round(2).to_csv(run / "test_iou_lop.csv")
    print(json.dumps(kq, indent=2, ensure_ascii=False))
    print(pd.DataFrame(bang, index=TEN_LOP).round(1).to_string())


if __name__ == "__main__":
    main()
