"""Phần dùng chung cho xuat_anh.py, train_seg.py, eval_seg.py."""
from pathlib import Path

import cv2
import numpy as np

GOC = Path(__file__).resolve().parents[3]          # D:/CS406/DoAn/Sage (thư mục đã chuyển vào luu_tru/)
FMB = GOC / "xulianh/data/raw/FMB"
MAN = GOC / "xulianh/data/manifests"
ANH = Path(__file__).resolve().parent / "anh"      # anh/{bien_the}/{train|test}/{id}.png
RUNS = Path(__file__).resolve().parent / "runs"

NC = 14  # lớp 1..14 của FMB -> chỉ số 0..13; nhãn 0 (unlabeled) -> 255 (bỏ qua)
IGNORE = 255
TEN_LOP = ["road", "sidewalk", "building", "traffic_light", "traffic_sign", "vegetation", "sky",
           "person", "car", "truck", "bus", "motorcycle", "bicycle", "pole"]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def thu_muc(r):
    """'train' hoặc 'test' – thư mục gốc FMB của ảnh (dev nằm trong train)."""
    return r.vis_path.split("/")[0]


def doc_nhan(path):
    lab = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert lab is not None and lab.ndim == 2 and lab.max() <= NC, path
    lab = lab.astype(np.int64) - 1
    lab[lab < 0] = IGNORE
    return lab


def nham(gt, pred, n=NC):
    """Ma trận nhầm lẫn n x n (hàng = GT, cột = dự đoán), bỏ pixel IGNORE."""
    m = gt != IGNORE
    return np.bincount(gt[m] * n + pred[m], minlength=n * n).reshape(n, n)


def miou(c):
    """mIoU % trên các lớp có mặt trong GT (quy ước chính của nhóm). Trả (mIoU, IoU từng lớp)."""
    tp = np.diag(c).astype(np.float64)
    g, p = c.sum(1), c.sum(0)
    iou = np.full(len(c), np.nan)
    ok = g > 0
    iou[ok] = tp[ok] / (g[ok] + p[ok] - tp[ok])
    return float(np.nanmean(iou) * 100), iou * 100
