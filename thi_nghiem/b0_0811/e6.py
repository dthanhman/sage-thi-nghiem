"""E6: AdaBN cho SAGE Student – tính lại thống kê BatchNorm trên dữ liệu mới, không train trọng số (xem README).

  M3FD: ước lượng BN trên 150 ảnh đầu (theo tên), chấm 150 ảnh sau.
  FMB (kiểm tra hợp lệ): ước lượng trên dev, chấm 273 ảnh test sạch.
"""
import copy
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
GOC = HERE.parent
PR6 = GOC / "pr6_eval"
SAGE = GOC / "thu_viec3/SAGE_IVIF"
FMB = GOC / "xulianh/data/raw/FMB"
MAN = GOC / "xulianh/data/manifests"
M3 = GOC / "data_ngoai/m3fd_fusion"
B, SEED = 2000, 0


def _init():
    sys.path.insert(0, str(PR6))


def cham(job):
    from src.evaluation.metrics import (entropy, fusion_ms_ssim, q_abf, standard_deviation,
                                        sum_of_correlation_differences)
    ten, bt, f, ir, y = job
    f, ir, y = (a.astype(np.float64) for a in (f, ir, y))
    return {"name": ten, "bt": bt, "SCD": sum_of_correlation_differences(f, ir, y),
            "MS_SSIM_sum": fusion_ms_ssim(f, ir, y, data_range=255.0, aggregation="sum")["MS_SSIM_sum"],
            "Q_ABF": q_abf(f, ir, y), "EN": entropy(f), "SD": standard_deviation(f)}


def doc(vis_p, ir_p):
    v = cv2.imread(str(vis_p))
    return cv2.cvtColor(v, cv2.COLOR_BGR2YCrCb)[:, :, 0], cv2.cvtColor(cv2.imread(str(ir_p)), cv2.COLOR_BGR2GRAY)


def t2(z):
    import torch
    return torch.from_numpy(z).float().div(255)[None, None].cuda()


def ghep(net, y, ir):
    import torch
    with torch.inference_mode():
        return _ghep(net, y, ir)


def _ghep(net, y, ir):
    return (net(t2(y), t2(ir))[0].clamp(0, 1) * 255).round().byte().cpu().numpy()[0, 0]


def adabn(net0, cap):
    import torch
    with torch.no_grad():
        return _adabn(net0, cap)


def _adabn(net0, cap):
    import torch
    net = copy.deepcopy(net0)
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None  # trung bình cộng dồn trên mọi ảnh
    net.train()
    for vp, ip in cap:
        y, ir = doc(vp, ip)
        net(t2(y), t2(ir))
    return net.eval()


def tong_hop(df, nhom, ten_bo):
    print(f"\n== {ten_bo}: {df.name.nunique()} ảnh chấm ==")
    rng = np.random.default_rng(SEED)
    g = np.array([nhom[n] for n in sorted(df.name.unique())])
    ug = np.unique(g)
    idxs = [np.concatenate([np.flatnonzero(g == k) for k in rng.choice(ug, len(ug))]) for _ in range(B)]
    kq = {}
    for k in ["SCD", "MS_SSIM_sum", "Q_ABF", "EN", "SD"]:
        p = df.pivot(index="name", columns="bt", values=k).sort_index()
        a, b = p["adabn"].to_numpy(), p["goc"].to_numpy()
        f = lambda i: (a[i].mean() / b[i].mean() - 1) * 100
        d = f(np.arange(len(a)))
        c = np.percentile([f(i) for i in idxs], [2.5, 97.5])
        kq[k] = (d, c)
        print(f"  {k:12s} gốc {b.mean():.4f}  AdaBN {a.mean():.4f}  Δ {d:+.2f}% [{c[0]:+.2f}; {c[1]:+.2f}]")
    return kq


def main():
    import torch
    sys.path.insert(0, str(SAGE))
    from model_sub.model import Network
    net = Network()
    net.load_state_dict(torch.load(SAGE / "checkpoints/checkpoints.pt", map_location="cpu", weights_only=True))
    net.cuda().eval()

    # M3FD
    ten = sorted(p.stem for p in (M3 / "Vis").glob("*.png"))
    cap = lambda ns: [(M3 / "Vis" / f"{n}.png", M3 / "Ir" / f"{n}.png") for n in ns]
    uoc, cham_ = ten[:150], ten[150:]
    net_m = adabn(net, cap(uoc))
    # FMB
    dev = pd.read_csv(MAN / "dev.csv", dtype={"image_id": str})
    test = pd.read_csv(MAN / "test.csv", dtype={"image_id": str})
    test = test[~test.leak]
    net_f = adabn(net, [(FMB / r.vis_path, FMB / r.ir_path) for r in dev.itertuples()])

    bn0, bn1 = net.module1[1], net_m.module1[1]
    print("BN lớp đầu, trung bình |Δmean|:", float((bn0.running_mean - bn1.running_mean).abs().mean()),
          " tỉ lệ var M3FD/gốc TB:", float((bn1.running_var / bn0.running_var).mean()))

    jobs = {"m3fd": [], "fmb": []}
    for n in cham_:
        y, ir = doc(M3 / "Vis" / f"{n}.png", M3 / "Ir" / f"{n}.png")
        jobs["m3fd"] += [(n, "goc", ghep(net, y, ir), ir, y), (n, "adabn", ghep(net_m, y, ir), ir, y)]
    for r in test.itertuples():
        y, ir = doc(FMB / r.vis_path, FMB / r.ir_path)
        jobs["fmb"] += [(r.image_id, "goc", ghep(net, y, ir), ir, y), (r.image_id, "adabn", ghep(net_f, y, ir), ir, y)]

    with ProcessPoolExecutor(4, initializer=_init) as ex:
        out = {k: pd.DataFrame(list(ex.map(cham, v, chunksize=4))) for k, v in jobs.items()}
    for k, d in out.items():
        d.to_csv(HERE / f"kq_e6_{k}.csv", index=False)

    km = tong_hop(out["m3fd"], {n: n for n in cham_}, "M3FD (BN ước lượng trên 150 ảnh khác)")
    khoi = (test.image_id.astype(int).diff() != 1).cumsum()
    kf = tong_hop(out["fmb"], dict(zip(test.image_id, khoi)), "FMB test 273 (kiểm tra hợp lệ, BN ước lượng trên dev)")

    hop_le = all(abs(kf[k][0]) < 1 for k in ["SCD", "MS_SSIM_sum", "Q_ABF"])
    dat = km["SCD"][1][0] > 0 and km["Q_ABF"][1][0] > 0 and km["MS_SSIM_sum"][1][1] >= 0
    print(f"\nKiểm tra hợp lệ trên FMB (|Δ| < 1%): {'CÓ' if hop_le else 'KHÔNG'}")
    print(f"Kết luận E6 theo tiêu chí: {'ĐẠT' if dat and hop_le else ('KHÔNG KẾT LUẬN' if not hop_le else 'TRƯỢT')}")


if __name__ == "__main__":
    main()
