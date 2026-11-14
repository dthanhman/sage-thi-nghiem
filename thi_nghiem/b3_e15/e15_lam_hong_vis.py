"""E15 – Khi nào ghép có ích: làm hỏng VIS test, ghép lại, chấm bằng các model SegFormer-B3 đã train (không train lại).

Câu hỏi: VIS càng hỏng thì lợi ích của ghép (SAGE, tb) so với VIS có tăng không? SAGE có hơn trộn trung bình (tb)
và hơn chỉ dùng IR ở chỗ VIS hỏng không?

Cách làm (chỉ làm hỏng VIS, IR giữ nguyên; 280 ảnh test, báo trên 273 ảnh sạch):
  toi   : ảnh tối + nhiễu. Đổi sang tuyến tính (^2.2), nhân độ phơi sáng a, nhiễu Poisson (P=2000 photon)
          + nhiễu đọc Gauss (sigma 0,002), đổi lại sRGB. Mức 1/2/3: a = 0,30 / 0,12 / 0,05
  suong : sương/khói đều. x' = x*t + 0,85*(1-t). Mức 1/2/3: t = 0,60 / 0,35 / 0,15
  sach  : mức 0, tính lại để đối chiếu với test_nham_last.npz của Kaggle (kiểm code)
Mỗi model chấm đúng loại đầu vào nó đã học: vis <- VIS hỏng; sage <- SAGE(VIS hỏng, IR); tb <- (Y VIS hỏng + IR)/2.
Model ir: đầu vào không đổi -> lấy lại ma trận nhầm lẫn đã có. Nhiễu cố định theo id ảnh (tái lập được).

TIÊU CHÍ ĐẶT TRƯỚC (Mân duyệt 14/11, ghi trước khi chạy), mIoU_co_mat, 273 ảnh, CI95 bootstrap theo khối:
  C1 ghép có ích khi VIS hỏng: ở mức 3 (cả toi lẫn suong) sage−vis ≥ +5 và CI > 0 ở mọi seed, và sage−vis
     tăng dần theo mức (0<1<2<3, theo TB seed).  Đạt cả hai = ĐẠT; chỉ một = LƯNG CHỪNG; không = KHÔNG.
  C2 SAGE có giá trị riêng so với trộn trung bình: ở mức 3, sage−tb > 0 và CI > 0 ở mọi seed = ĐẠT (SAGE hơn);
     tb−sage > 0 và CI > 0 ở mọi seed = GAP (cách trộn của SAGE không hơn trộn TB đúng ở chỗ cần IR);
     còn lại = LƯNG CHỪNG.
  C3 ghép còn tận dụng VIS hỏng: ở mức 3, sage−ir > 0 và CI > 0 ở mọi seed = ĐẠT; ir−sage > 0 và CI > 0 = KHÔNG
     (VIS hỏng nặng thì dùng IR một mình tốt hơn); còn lại = LƯNG CHỪNG.
Lưu ý: model vis/sage/tb đều học trên ảnh sạch -> so sage−vis thiên về phía ghép; so sánh chính là sage với tb, ir.

Chạy (env DoAn, GPU):
  python e15_lam_hong_vis.py --thu          # 5 ảnh, 1 seed, đo tốc độ
  python e15_lam_hong_vis.py                # chạy thật (tự bỏ qua phần đã có), rồi in báo cáo -> e15.txt
  python e15_lam_hong_vis.py --chi_bao_cao  # chỉ in lại báo cáo
Ra: e15/conf/s{seed}_{bt}_{dk}.npz (ma trận nhầm lẫn 15x15 từng ảnh), e15/anh_mau/*.jpg, e15.txt
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
TL = HERE.parent
sys.path.insert(0, str(TL / "b3"))
from chung import MAN, MEAN, STD, STUDENT, cac_miou, doc_nhan, duong, nham  # noqa: E402

B0 = TL.parent / "luu_tru" / "tim_gap_b0_0811"
sys.path.insert(0, str(B0))
from tong_hop import boot_idx, ci, SEED as SEED_BOOT  # noqa: E402

OUT = HERE / "e15"
KAG = TL / "_kaggle_out"
DK = ["sach"] + [f"{k}{m}" for k in ("toi", "suong") for m in (1, 2, 3)]
A_TOI = {1: 0.30, 2: 0.12, 3: 0.05}
T_SUONG = {1: 0.60, 2: 0.35, 3: 0.15}
BT = ["vis", "sage", "tb"]
ANH_MAU = ["00089", "00518"]  # khói (ca đã biết), phố đêm


def tim_run(bt, s):
    """last.pt của biến thể bt, seed s (tìm trong mọi thư mục tải từ Kaggle)."""
    ds = sorted(KAG.rglob(f"{bt}_b3_s{s}_1220_15lop/last.pt"))
    return ds[0].parent if ds else None


def lam_hong(vis, dk, iid):
    """vis BGR uint8 -> ảnh hỏng theo điều kiện dk."""
    if dk == "sach":
        return vis
    x = vis.astype(np.float64) / 255
    rng = np.random.default_rng(int(iid))
    if dk.startswith("toi"):
        a, P = A_TOI[int(dk[-1])], 2000.0
        lin = x ** 2.2 * a
        lin = rng.poisson(lin * P) / P + rng.normal(0, 0.002, lin.shape)
        x = np.clip(lin, 0, 1) ** (1 / 2.2)
    else:
        t = T_SUONG[int(dk[-1])]
        x = x * t + 0.85 * (1 - t)
    return (np.clip(x, 0, 1) * 255).round().astype(np.uint8)


class GhepSAGE:
    def __init__(self, dev):
        sys.path.insert(0, str(STUDENT))
        from model import Network
        self.net = Network()
        self.net.load_state_dict(torch.load(STUDENT / "checkpoints.pt", map_location="cpu", weights_only=True),
                                 strict=True)
        self.net.to(dev).eval()
        self.dev = dev

    @torch.inference_mode()
    def __call__(self, y, ir):
        t = lambda z: torch.from_numpy(z).float().div(255)[None, None].to(self.dev)
        return (self.net(t(y), t(ir))[0].clamp(0, 1) * 255).round().byte().cpu().numpy()[0, 0]


def dau_vao(bt, vis_h, ir, sage):
    """Ảnh BGR đưa vào model bt, giống hệt xuat_anh.py nhưng từ VIS đã hỏng."""
    if bt == "vis":
        return vis_h
    ycc = cv2.cvtColor(vis_h, cv2.COLOR_BGR2YCrCb)
    y = ycc[:, :, 0]
    yf = ((y.astype(np.float64) + ir) / 2).round().astype(np.uint8) if bt == "tb" else sage(y, ir)
    return cv2.cvtColor(np.dstack([yf, ycc[:, :, 1:]]), cv2.COLOR_YCrCb2BGR)


def chay(seeds, gioi_han):
    from train_seg import du_doan, tao_model
    dev = "cuda"
    df = pd.read_csv(MAN / "test.csv", dtype={"image_id": str})
    if gioi_han:
        df = df.head(gioi_han)
    sage = GhepSAGE(dev)
    (OUT / "conf").mkdir(parents=True, exist_ok=True)
    (OUT / "anh_mau").mkdir(parents=True, exist_ok=True)
    nhan = [doc_nhan(duong(r.label_path), lop0=True) for r in df.itertuples()]
    vis0 = [cv2.imread(str(duong(r.vis_path))) for r in df.itertuples()]
    irs = [cv2.cvtColor(cv2.imread(str(duong(r.ir_path))), cv2.COLOR_BGR2GRAY) for r in df.itertuples()]
    for dk in DK:
        viec = [(s, bt) for s in seeds for bt in BT
                if gioi_han or not (OUT / "conf" / f"s{s}_{bt}_{dk}.npz").exists()]
        if not viec:
            continue
        t0 = time.time()
        vis_h = [lam_hong(v, dk, r.image_id) for v, r in zip(vis0, df.itertuples())]
        vao = {bt: [dau_vao(bt, v, ir, sage) for v, ir in zip(vis_h, irs)] for bt in {b for _, b in viec}}
        del vis_h
        print(f"[{dk}] tạo ảnh đầu vào {time.time() - t0:.0f} s", flush=True)
        for s, bt in viec:
            model = tao_model("b3", 15)
            model.load_state_dict(torch.load(tim_run(bt, s) / "last.pt", map_location="cpu", weights_only=True))
            model.to(dev).eval()
            t0 = time.time()
            conf = np.zeros((len(df), 15, 15), np.int64)
            for i, r in enumerate(df.itertuples()):
                img = vao[bt][i]
                if s == seeds[0] and r.image_id in ANH_MAU:
                    cv2.imwrite(str(OUT / "anh_mau" / f"{r.image_id}_{dk}_{bt}.jpg"), img,
                                [cv2.IMWRITE_JPEG_QUALITY, 85])
                x = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - MEAN) / STD
                x = torch.from_numpy(x.transpose(2, 0, 1).copy())[None].to(dev)
                p = du_doan(model, x).argmax(1)[0].cpu().numpy()
                conf[i] = nham(nhan[i], p, 15)
            dt = time.time() - t0
            print(f"seed {s} {bt:4s} {dk:6s}: {len(df)} ảnh {dt:.0f} s ({dt / len(df):.2f} s/ảnh) "
                  f"| mIoU_co_mat (mọi ảnh) {cac_miou(conf.sum(0))[0]['mIoU_co_mat']:.2f}", flush=True)
            if not gioi_han:
                np.savez_compressed(OUT / "conf" / f"s{s}_{bt}_{dk}.npz", conf=conf,
                                    image_id=df.image_id.to_numpy().astype(str))
            del model
            torch.cuda.empty_cache()


def bao_cao(seeds):
    anh = pd.read_csv(B0 / "kq_anh.csv", dtype={"image_id": str})
    sach = ~anh.leak.to_numpy()
    khoi = anh.khoi[sach].to_numpy()
    m = lambda c: cac_miou(c)[0]["mIoU_co_mat"]
    dong = []
    in_ = lambda *a: (print(*a), dong.append(" ".join(str(x) for x in a)))
    in_(__doc__.split("Chạy (env")[0].strip())
    in_(f"\nSeed: {seeds}. mIoU_co_mat trên {sach.sum()} ảnh sạch. * = CI95 không chứa 0.\n")

    # Kiểm code: mức sạch tính lại phải khớp số Kaggle
    for s in seeds:
        for bt in BT + ["ir"]:
            z0 = np.load(tim_run(bt, s) / "test_nham_last.npz")
            assert list(z0["image_id"].astype(str)) == list(anh.image_id)
            if bt != "ir":
                c = np.load(OUT / "conf" / f"s{s}_{bt}_sach.npz")["conf"]
                in_(f"kiểm: seed {s} {bt:4s} sạch tính lại {m(c[sach].sum(0)):.2f} | Kaggle {m(z0['conf'][sach].sum(0)):.2f}")

    CAP = [("sage", "vis"), ("tb", "vis"), ("sage", "tb"), ("sage", "ir"), ("tb", "ir")]
    R = []  # seed, dk, cap, gia_tri, lo, hi
    MI = []
    for s in seeds:
        cir = np.load(tim_run("ir", s) / "test_nham_last.npz")["conf"][sach]
        for dk in DK:
            C = {bt: np.load(OUT / "conf" / f"s{s}_{bt}_{dk}.npz")["conf"][sach] for bt in BT}
            C["ir"] = cir
            for b, c in C.items():
                MI.append((s, dk, b, m(c.sum(0))))
            rng = np.random.default_rng(SEED_BOOT)
            ds = {cp: [] for cp in CAP}
            for idx in boot_idx(khoi, rng):
                v = {b: m(c[idx].sum(0)) for b, c in C.items()}
                for a, b in CAP:
                    ds[(a, b)].append(v[a] - v[b])
            for a, b in CAP:
                lo, hi = ci(ds[(a, b)])
                R.append((s, dk, f"{a}−{b}", m(C[a].sum(0)) - m(C[b].sum(0)), lo, hi))
    mi = pd.DataFrame(MI, columns=["seed", "dk", "bt", "miou"])
    t = mi.pivot_table(index="dk", columns="bt", values="miou", aggfunc="mean").loc[DK, ["vis", "ir", "sage", "tb"]]
    in_("\n== mIoU (TB các seed) ==\n" + t.round(2).to_string())
    r = pd.DataFrame(R, columns=["seed", "dk", "cap", "gia_tri", "lo", "hi"])
    r["o"] = r.apply(lambda x: f"{x.gia_tri:+.2f}{'*' if x.lo > 0 or x.hi < 0 else ' '} [{x.lo:+.1f};{x.hi:+.1f}]", axis=1)
    for s in seeds:
        in_(f"\n== Hiệu (seed {s}) ==\n" + r[r.seed == s].pivot(index="dk", columns="cap", values="o")
            .loc[DK, [f"{a}−{b}" for a, b in CAP]].to_string())

    # Tiêu chí
    def lay(dk, cap):
        return r[(r.dk == dk) & (r.cap == cap)]
    in_("\n== Tiêu chí ==")
    for k in ("toi", "suong"):
        g = lay(f"{k}3", "sage−vis")
        c1a = bool(((g.gia_tri >= 5) & (g.lo > 0)).all())
        tb = [r[(r.dk == d) & (r.cap == "sage−vis")].gia_tri.mean() for d in ["sach", f"{k}1", f"{k}2", f"{k}3"]]
        c1b = bool(np.all(np.diff(tb) > 0))
        in_(f"[{k}] C1 sage−vis mức 0..3 (TB seed) {' '.join(f'{v:+.2f}' for v in tb)} | mức 3 ≥+5 & CI>0: {c1a} "
            f"| tăng dần: {c1b} -> {'ĐẠT' if c1a and c1b else 'LƯNG CHỪNG' if c1a or c1b else 'KHÔNG'}")
        g = lay(f"{k}3", "sage−tb")
        c2 = "ĐẠT (SAGE hơn tb)" if (g.lo > 0).all() else "GAP (tb ≥ SAGE)" if (g.hi < 0).all() else "LƯNG CHỪNG"
        in_(f"[{k}] C2 sage−tb mức 3: {' '.join(g.o)} -> {c2}")
        g = lay(f"{k}3", "sage−ir")
        c3 = "ĐẠT" if (g.lo > 0).all() else "KHÔNG (IR một mình tốt hơn)" if (g.hi < 0).all() else "LƯNG CHỪNG"
        in_(f"[{k}] C3 sage−ir mức 3: {' '.join(g.o)} -> {c3}")
    (HERE / "e15.txt").write_text("\n".join(dong), encoding="utf-8")
    mi.to_csv(HERE / "kq_e15_miou.csv", index=False)
    r.drop(columns="o").to_csv(HERE / "kq_e15_hieu.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thu", action="store_true", help="5 ảnh, 1 seed, không lưu")
    ap.add_argument("--chi_bao_cao", action="store_true")
    a = ap.parse_args()
    seeds = [s for s in range(10) if all(tim_run(b, s) for b in BT + ["ir"])]
    print("Seed có đủ vis/sage/tb/ir:", seeds, flush=True)
    if a.thu:
        chay(seeds[:1], 5)
        return
    if not a.chi_bao_cao:
        chay(seeds, 0)
    bao_cao(seeds)


if __name__ == "__main__":
    main()
