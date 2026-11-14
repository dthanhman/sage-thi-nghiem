"""E17 – Train lại Student của SAGE KHÔNG có SAM (và không có Teacher), chỉ dùng loss nội dung.

Câu hỏi: phần SAM/Teacher của SAGE có làm ảnh ghép tốt hơn cho phân đoạn không?
Bài (Bảng 3): bỏ SAM thì mIoU FMB 61,2 -> 56,5 (-4,7). Nhóm kiểm lại bằng cùng thiết lập B3.

Giữ nguyên theo external/SAGE_IVIF/train.py + model_sub/losses.py:
  - kiến trúc Student (sage_student/model.py, 0,136M tham số)
  - loss nội dung: loss_fuse = 3*MSE(o,Y) + 2*MSE(o,IR); loss_grad = 9*Grad(o,Y,IR); loss = 3*loss_fuse + 3*loss_grad
  - Adam lr 2e-3, CosineAnnealingLR(T_max = 100 epoch * số batch, eta_min 1e-5), batch 4, crop 192x256 + lật ngang
  - 100 epoch; code gốc chỉ cập nhật Student ở batch chẵn (xen kẽ với Teacher) -> số bước cập nhật = nửa số batch,
    và scheduler cũng chỉ bước theo Student -> lr chỉ giảm nửa chu kỳ cosine. Nhóm làm GIỐNG HỆT điều này.
BỎ: các số hạng có ref (ảnh Teacher), loss tương phản theo mask SAM, loss cosine đặc trưng trung gian.
Dữ liệu: 1.220 ảnh train chính thức (train + dev của nhóm), KHÔNG đụng test.

Ra: ../b3/sage_student/nosam.pt (+ nosam_log.csv). Sau đó: python ../b3/xuat_anh.py --bien_the nosam
Ví dụ: python train_student_nosam.py           (chạy thật)
       python train_student_nosam.py --thu 50  (chạy thử 50 bước: đo tốc độ)
"""
import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_DAY = Path(__file__).resolve().parent
B3 = _DAY if (_DAY / "chung.py").exists() else _DAY.parent / "b3"  # gói Kaggle: cùng thư mục
sys.path.insert(0, str(B3))
from chung import MAN, STUDENT, duong  # noqa: E402

sys.path.insert(0, str(STUDENT))
from model import Network  # noqa: E402

SOBEL_X = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]])[None, None]
SOBEL_Y = torch.tensor([[1., 2, 1], [0, 0, 0], [-1, -2, -1]])[None, None]


def grad(x, kx, ky):
    return F.conv2d(x, kx, padding=1).abs() + F.conv2d(x, ky, padding=1).abs()


def grad_loss(o, s1, s2, kx, ky):
    """Như GradientLoss của SAGE: L1(|∇o|, max(|∇s1|, |∇s2|))."""
    return F.l1_loss(grad(o, kx, ky), torch.maximum(grad(s1, kx, ky), grad(s2, kx, ky)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lr_min", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thu", type=int, default=0, help="chỉ chạy N bước để đo tốc độ, không lưu")
    ap.add_argument("--ten", default="nosam")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("Thiết bị:", dev, flush=True)

    df = pd.concat([pd.read_csv(MAN / f"{s}.csv", dtype={"image_id": str}) for s in ("train", "dev")])
    assert len(df) == 1220 and not df.vis_path.str.startswith("test").any()
    print(f"Nạp {len(df)} ảnh train vào RAM ...", flush=True)
    Y = np.stack([cv2.cvtColor(cv2.imread(str(duong(p))), cv2.COLOR_BGR2YCrCb)[:, :, 0] for p in df.vis_path])
    IR = np.stack([cv2.cvtColor(cv2.imread(str(duong(p))), cv2.COLOR_BGR2GRAY) for p in df.ir_path])
    N, H, W = Y.shape
    ch, cw = 192, 256

    net = Network().to(dev).train()
    print(f"Student: {sum(p.numel() for p in net.parameters()) / 1e6:.3f}M tham số", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    so_batch = math.ceil(N / a.batch)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * so_batch, eta_min=a.lr_min)
    kx, ky = SOBEL_X.to(dev), SOBEL_Y.to(dev)

    log, buoc, t0 = [], 0, time.time()
    for ep in range(1, a.epochs + 1):
        thu_tu = rng.permutation(N)
        tong = dict(loss=0., fuse=0., grad=0., n=0)
        for bi in range(so_batch):
            if bi % 2 == 1:      # code gốc: batch lẻ dành cho Teacher, Student chỉ học ở batch chẵn
                continue
            idx = thu_tu[bi * a.batch:(bi + 1) * a.batch]
            ys, irs = [], []
            for k in idx:
                i0, j0 = rng.integers(0, H - ch + 1), rng.integers(0, W - cw + 1)
                y, ir = Y[k, i0:i0 + ch, j0:j0 + cw], IR[k, i0:i0 + ch, j0:j0 + cw]
                if rng.random() < 0.5:
                    y, ir = y[:, ::-1], ir[:, ::-1]
                ys.append(y.copy()), irs.append(ir.copy())
            y = torch.from_numpy(np.stack(ys)).float().div(255)[:, None].to(dev)
            ir = torch.from_numpy(np.stack(irs)).float().div(255)[:, None].to(dev)
            o = net(y, ir)[0]
            l_fuse = 3 * F.mse_loss(o, y) + 2 * F.mse_loss(o, ir)
            l_grad = 9 * grad_loss(o, y, ir, kx, ky)
            loss = 3 * l_fuse + 3 * l_grad
            if not torch.isfinite(loss):
                sys.exit(f"loss không hữu hạn ở bước {buoc}")
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
            buoc += 1
            tong["loss"] += loss.item(); tong["fuse"] += l_fuse.item(); tong["grad"] += l_grad.item(); tong["n"] += 1
            if a.thu and buoc >= a.thu:
                dt = time.time() - t0
                print(f"Thử {buoc} bước: {dt:.1f}s -> ước tính cả bài {dt / buoc * a.epochs * so_batch / 2 / 60:.0f} phút,"
                      f" GPU {torch.cuda.max_memory_allocated() / 2**20 if dev == 'cuda' else 0:.0f} MB, loss {loss.item():.4f}")
                return
        n = tong.pop("n")
        dong = {"epoch": ep, "buoc": buoc, "lr": sch.get_last_lr()[0], **{k: v / n for k, v in tong.items()},
                "phut": (time.time() - t0) / 60}
        log.append(dong)
        if ep % 5 == 0 or ep == 1:
            print(f"epoch {ep:3d} bước {buoc} lr {dong['lr']:.2e} loss {dong['loss']:.4f} "
                  f"(fuse {dong['fuse']:.4f}, grad {dong['grad']:.4f}) {dong['phut']:.1f} phút", flush=True)
    torch.save(net.state_dict(), STUDENT / f"{a.ten}.pt")
    pd.DataFrame(log).to_csv(STUDENT / f"{a.ten}_log.csv", index=False)
    print(f"Lưu {STUDENT / (a.ten + '.pt')}")


if __name__ == "__main__":
    main()
