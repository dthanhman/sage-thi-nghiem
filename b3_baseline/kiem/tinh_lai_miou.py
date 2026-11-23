"""Tính lại mIoU từ ma trận nhầm lẫn từng ảnh, KHÔNG dùng code/chung.py. Chạy: python kiem/tinh_lai_miou.py"""
import json
from pathlib import Path
import numpy as np

RUNS = Path(__file__).resolve().parents[1] / "runs"


def miou(M):
    """M 15x15, hàng = GT, cột = dự đoán. Bỏ pixel GT 0; GT 1..14 bị đoán 0 vẫn là FN. TB các lớp có trong GT."""
    r = M[1:].astype(float)
    tp = np.diag(M)[1:].astype(float)
    g, p = r.sum(1), r[:, 1:].sum(0)
    co = g > 0
    return (tp[co] / (g[co] + p[co] - tp[co])).mean() * 100


def miou_bo_cot0(M):
    """Đối chiếu: bỏ luôn các pixel bị đoán thành 0 (KHÔNG phải số chính)."""
    n = M[1:, 1:].astype(float)
    co = M[1:].sum(1) > 0
    tp, u = np.diag(n), n.sum(0) + n.sum(1) - np.diag(n)
    return (tp[co] / u[co]).mean() * 100


print("bien_the | 273 ảnh s0 s1 s2 | TB 273 | TB 280 | TB 273 nếu bỏ cột đoán 0")
for bt in ("vis", "ir", "tb", "sage"):
    a, b, c = [], [], []
    for s in range(3):
        d = RUNS / f"{bt}_b3_s{s}_1220_15lop"
        z = np.load(d / "test_nham_last.npz")
        kq = json.loads((d / "test_ket_qua_last.json").read_text(encoding="utf-8"))
        m273, m280 = z["conf"][~z["leak"]].sum(0), z["conf"].sum(0)
        a.append(miou(m273)); b.append(miou(m280)); c.append(miou_bo_cot0(m273))
        assert len(z["conf"]) == 280 and z["leak"].sum() == 7, d
        assert abs(a[-1] - kq["273"]["mIoU_co_mat"]) < 1e-6 and abs(b[-1] - kq["280"]["mIoU_co_mat"]) < 1e-6, d
    print(bt, "|", *(f"{x:.2f}" for x in a), f"| {np.mean(a):.2f} | {np.mean(b):.2f} | {np.mean(c):.2f}")
print("Khớp test_ket_qua_last.json ở cả 12 run.")
