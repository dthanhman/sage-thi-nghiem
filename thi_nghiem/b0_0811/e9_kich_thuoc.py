"""E9 (kiểm G3 – Student có "quen" kích thước lúc train không?).
Student được train trên crop 192x256 Ở ĐÚNG TỈ LỆ ẢNH GỐC (train.py crop_size, random_crop chỉ cắt, không thu nhỏ).
Chạy Student trên 273 ảnh test FMB theo 5 cách, chấm SCD, MS-SSIM(tổng), Q_ABF trên ảnh 800x600:
  full  : cả ảnh (như bình thường)
  tile  : cắt ô 192x256 (pad phản xạ lên 768x1024), chạy từng ô rồi ghép  -> đúng điều kiện lúc train
  s0.5, s0.75, s1.5 : thu/phóng ảnh vào, chạy, đưa về 800x600
Tiêu chí đặt trước: G3 có bằng chứng nếu `tile` hơn `full` ở >= 2/3 metric với CI95 > 0
(hoặc một tỉ lệ khác 1.0 hơn `full` như vậy). Ngược lại: G3 không có bằng chứng."""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from chay import FMB, MAN, SAGE, PR6, HERE
from tong_hop import boot_idx, ci, SEED

CACH = ["full", "tile", "s0.5", "s0.75", "s1.5"]
QUYET = ["SCD", "MS_SSIM_sum", "Q_ABF"]


def init():
    sys.path.insert(0, str(PR6))


def cham(job):
    from src.evaluation.metrics import fusion_ms_ssim, q_abf, sum_of_correlation_differences
    iid, c, f, ir, y = job
    f, ir, y = (a.astype(np.float64) for a in (f, ir, y))
    return {"image_id": iid, "cach": c, "SCD": sum_of_correlation_differences(f, ir, y),
            "MS_SSIM_sum": fusion_ms_ssim(f, ir, y, data_range=255.0, aggregation="sum")["MS_SSIM_sum"],
            "Q_ABF": q_abf(f, ir, y)}


def main():
    sys.path.insert(0, str(SAGE))
    from model_sub.model import Network
    net = Network()
    net.load_state_dict(torch.load(SAGE / "checkpoints/checkpoints.pt", map_location="cpu", weights_only=True))
    net.cuda().eval()
    t2 = lambda z: torch.from_numpy(np.ascontiguousarray(z)).float().div(255)[None, None].cuda()

    def chay(y, ir):
        with torch.inference_mode():
            return net(t2(y), t2(ir))[0].clamp(0, 1)[0, 0].cpu().numpy() * 255

    def chay_cach(y, ir, c):
        H, W = y.shape
        if c == "full":
            o = chay(y, ir)
        elif c == "tile":
            yp = cv2.copyMakeBorder(y, 0, 768 - H, 0, 1024 - W, cv2.BORDER_REFLECT_101)
            ip = cv2.copyMakeBorder(ir, 0, 768 - H, 0, 1024 - W, cv2.BORDER_REFLECT_101)
            o = np.zeros(yp.shape, np.float64)
            for i in range(0, 768, 192):
                for j in range(0, 1024, 256):
                    o[i:i + 192, j:j + 256] = chay(yp[i:i + 192, j:j + 256], ip[i:i + 192, j:j + 256])
            o = o[:H, :W]
        else:
            s = float(c[1:])
            sz = (round(W * s / 8) * 8, round(H * s / 8) * 8)
            o = chay(cv2.resize(y, sz, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC),
                     cv2.resize(ir, sz, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC))
            o = cv2.resize(o, (W, H), interpolation=cv2.INTER_CUBIC if s < 1 else cv2.INTER_AREA)
        return np.clip(o, 0, 255).round().astype(np.uint8)

    anh = pd.read_csv(HERE / "kq_anh.csv", dtype={"image_id": str})
    te = pd.read_csv(MAN, dtype={"image_id": str}).set_index("image_id")
    anh = anh[~anh.leak].reset_index(drop=True)
    jobs, psnr = [], {c: [] for c in CACH}
    for iid in anh.image_id:
        r = te.loc[iid]
        vis = cv2.imread(str(FMB / r.vis_path)); y = cv2.cvtColor(vis, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        ir = cv2.cvtColor(cv2.imread(str(FMB / r.ir_path)), cv2.COLOR_BGR2GRAY)
        out = {c: chay_cach(y, ir, c) for c in CACH}
        for c in CACH:
            psnr[c].append(cv2.PSNR(out[c], out["full"]) if c != "full" else np.inf)
            jobs.append((iid, c, out[c], ir, y))
    print("PSNR so với full (dB, trung vị):", {c: round(float(np.median(v)), 1) for c, v in psnr.items() if c != "full"})
    with ProcessPoolExecutor(4, initializer=init) as ex:
        kq = pd.DataFrame(list(ex.map(cham, jobs, chunksize=8)))
    kq.to_csv(HERE / "kq_e9.csv", index=False)
    bang = kq.pivot_table(index="cach", values=QUYET, aggfunc="mean").loc[CACH]
    print("\nTrung bình 273 ảnh:\n", bang.round(4).to_string())
    khoi = anh.khoi.to_numpy(); rng = np.random.default_rng(SEED); idxs = list(boot_idx(khoi, rng))
    base = kq[kq.cach == "full"].set_index("image_id").loc[anh.image_id]
    co = False
    print("\nChênh % so với full [CI95]:")
    for c in CACH[1:]:
        x = kq[kq.cach == c].set_index("image_id").loc[anh.image_id]
        dong, hon = [], 0
        for m in QUYET:
            d = (x[m].to_numpy() - base[m].to_numpy())
            f = lambda i: d[i].mean() / base[m].to_numpy()[i].mean() * 100
            v, cc = f(np.arange(len(d))), ci([f(i) for i in idxs])
            hon += cc[0] > 0
            dong.append(f"{m} {v:+.2f} [{cc[0]:+.2f}; {cc[1]:+.2f}]")
        co |= hon >= 2
        print(f"  {c:6s} " + " | ".join(dong))
    print("\nKết luận theo tiêu chí:", "G3 CÓ bằng chứng" if co else "G3 KHÔNG có bằng chứng")


if __name__ == "__main__":
    main()
