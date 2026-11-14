"""E15e – Khử nhiễu VIS trước khi ghép có thu hẹp khoảng cách tb − SAGE khi VIS tối không? (kiểm nhân quả cho E15d)

Bối cảnh: E15 toi3: tb − sage = +2,4*/+2,8* (seed 0/1). E15d: SAGE giữ nhiễu VIS gấp ~1,8 lần tb (vệt 2,2 lần),
nghi do loss gradient max(|∇VIS|, |∇IR|) của Student -> nhiễu VIS bị giữ như chi tiết.
Nếu nhiễu là nguyên nhân chính: khử nhiễu VIS rồi mới ghép -> SAGE hồi phục nhiều hơn tb -> khoảng cách thu hẹp.

Cách làm:
  - Làm tối VIS test giống hệt E15 (lam_hong, nhiễu cố định theo id), mức toi3 (chính) và toi2 (phụ). IR giữ nguyên.
  - Khử nhiễu: OpenCV fastNlMeansDenoisingColored (NLM), h = hColor chọn TRÊN ẢNH TRAIN (30 ảnh cách đều,
    làm tối cùng cách) bằng MSE nhỏ nhất so với VIS sạch – không nhìn test, không nhìn mIoU. Lưu ở e15e_h.json.
  - Cùng VIS đã khử nhiễu đưa vào CẢ SAGE lẫn tb (công bằng). Chấm bằng model B3 cùng loại (sage <- ảnh SAGE,
    tb <- ảnh tb), seed 0, 1, 2. 273 ảnh sạch, mIoU_co_mat.

TIÊU CHÍ ĐẶT TRƯỚC (ghi 14/11, trước khi chạy), CI95 bootstrap theo khối:
  Gọi g_raw = mIoU(tb) − mIoU(sage) trên VIS tối thô; g_dn = như vậy trên VIS đã khử nhiễu; Δ = g_raw − g_dn.
  K1 (chính, toi3):
    XÁC NHẬN  : Δ > 0 và CI > 0 ở cả 3 seed, Δ gộp 3 seed ≥ +1,0, VÀ g_dn gộp có CI chứa 0 hoặc < 0
                -> nhiễu VIS là nguyên nhân chính khiến SAGE thua tb khi tối.
    MỘT PHẦN  : Δ > 0 và CI > 0 ở cả 3 seed, Δ gộp ≥ +1,0, nhưng g_dn gộp vẫn > 0 có ý nghĩa
                -> nhiễu giải thích một phần.
    BÁC       : Δ gộp < +0,5 hoặc CI của Δ chứa 0 ở ≥ 2 seed -> nhiễu không phải nguyên nhân chính.
    còn lại   : LƯNG CHỪNG.
  Kèm (không dùng để kết luận): toi2; mIoU từng ô; PSNR VIS thô / đã khử so với VIS sạch.
Giới hạn: làm tối tổng hợp; model phân đoạn học trên ảnh sạch (như nhau cho sage và tb).

Chạy (env DoAn):
  python e15e_khu_nhieu.py --chon_h   # CPU, chọn h trên ảnh train -> e15e_h.json
  python e15e_khu_nhieu.py            # GPU: tạo ảnh, chấm (tự bỏ qua phần đã có), in báo cáo -> e15e.txt
  python e15e_khu_nhieu.py --chi_bao_cao
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e15_lam_hong_vis import B0, OUT, GhepSAGE, dau_vao, lam_hong, tim_run  # noqa: E402
from chung import MAN, MEAN, STD, cac_miou, doc_nhan, duong, nham  # noqa: E402
from tong_hop import boot_idx, ci, SEED  # noqa: E402

SEEDS = [0, 1, 2]
DK = ["toi3", "toi2"]
BT = ["sage", "tb"]
CACH = ["raw", "dn"]
CONF = OUT / "conf_e15e"
FH = HERE / "e15e_h.json"
H_LUOI = [3, 5, 8, 10, 13, 16, 20, 25, 30, 40]


def toi_khong_nhieu(vis, dk):
    """Ảnh tối cùng mức với lam_hong nhưng không có nhiễu – đích để chọn h."""
    a = {"toi1": 0.30, "toi2": 0.12, "toi3": 0.05}[dk]
    x = ((vis.astype(np.float64) / 255) ** 2.2 * a) ** (1 / 2.2)
    return (np.clip(x, 0, 1) * 255).round().astype(np.uint8)


def khu(img, h):
    return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)


def chon_h():
    df = pd.read_csv(MAN / "train.csv", dtype={"image_id": str})
    df = df.iloc[np.linspace(0, len(df) - 1, 30).round().astype(int)]
    kq = {}
    for dk in DK:
        mse = {h: [] for h in H_LUOI}
        tho = []
        for r in df.itertuples():
            v = cv2.imread(str(duong(r.vis_path)))
            vh = lam_hong(v, dk, r.image_id)
            ref = toi_khong_nhieu(v, dk).astype(np.float64)
            tho.append(np.mean((vh.astype(np.float64) - ref) ** 2))
            for h in H_LUOI:
                mse[h].append(np.mean((khu(vh, h).astype(np.float64) - ref) ** 2))
        tb = {h: float(np.mean(m)) for h, m in mse.items()}
        tho = float(np.mean(tho))
        best = min(tb, key=tb.get)
        kq[dk] = {"h": best, "mse_theo_h": tb, "mse_tho": tho}
        print(f"{dk}: h tốt nhất {best} | MSE thô {tho:.1f} | " + " ".join(f"h{h}:{m:.1f}" for h, m in tb.items()),
              flush=True)
    FH.write_text(json.dumps({"anh_train": list(df.image_id), **kq}, indent=1, ensure_ascii=False), encoding="utf-8")


def ten(s, bt, dk, c):
    return CONF / f"s{s}_{bt}_{dk}_{c}.npz"


def chay():
    from train_seg import du_doan, tao_model
    hcfg = json.loads(FH.read_text(encoding="utf-8"))
    df = pd.read_csv(MAN / "test.csv", dtype={"image_id": str})
    CONF.mkdir(parents=True, exist_ok=True)
    sage = GhepSAGE("cuda")
    nhan = [doc_nhan(duong(r.label_path), lop0=True) for r in df.itertuples()]
    vis0 = [cv2.imread(str(duong(r.vis_path))) for r in df.itertuples()]
    irs = [cv2.cvtColor(cv2.imread(str(duong(r.ir_path))), cv2.COLOR_BGR2GRAY) for r in df.itertuples()]
    psnr = {}
    for dk in DK:
        for c in CACH:
            # raw seed 0/1 đã có ở E15 -> chép lại, khỏi chạy
            for s in SEEDS:
                for bt in BT:
                    cu = OUT / "conf" / f"s{s}_{bt}_{dk}.npz"
                    if c == "raw" and cu.exists() and not ten(s, bt, dk, c).exists():
                        np.savez_compressed(ten(s, bt, dk, c), conf=np.load(cu)["conf"])
            viec = [(s, bt) for s in SEEDS for bt in BT if not ten(s, bt, dk, c).exists()]
            if not viec and c == "raw":
                continue
            t0 = time.time()
            vh = [lam_hong(v, dk, r.image_id) for v, r in zip(vis0, df.itertuples())]
            if c == "dn":
                vh = [khu(v, hcfg[dk]["h"]) for v in vh]
            psnr[f"{dk}_{c}"] = float(np.mean([cv2.PSNR(a, b) for a, b in zip(vh, vis0)]))
            if not viec:
                continue
            vao = {bt: [dau_vao(bt, v, ir, sage) for v, ir in zip(vh, irs)] for bt in {b for _, b in viec}}
            for iid in ("00089", "00518"):
                i = list(df.image_id).index(iid)
                (OUT / "anh_mau").mkdir(exist_ok=True)
                for bt, im in [("vis", vh[i])] + [(b, vao[b][i]) for b in vao]:
                    cv2.imwrite(str(OUT / "anh_mau" / f"e15e_{iid}_{dk}_{c}_{bt}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 88])
            del vh
            print(f"[{dk} {c}] tạo ảnh {time.time() - t0:.0f} s", flush=True)
            for s, bt in viec:
                model = tao_model("b3", 15)
                model.load_state_dict(torch.load(tim_run(bt, s) / "last.pt", map_location="cpu", weights_only=True))
                model.to("cuda").eval()
                conf = np.zeros((len(df), 15, 15), np.int64)
                for i, img in enumerate(vao[bt]):
                    x = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - MEAN) / STD
                    x = torch.from_numpy(x.transpose(2, 0, 1).copy())[None].cuda()
                    conf[i] = nham(nhan[i], du_doan(model, x).argmax(1)[0].cpu().numpy(), 15)
                np.savez_compressed(ten(s, bt, dk, c), conf=conf)
                print(f"  seed {s} {bt:4s} {dk} {c}: mIoU (280) {cac_miou(conf.sum(0))[0]['mIoU_co_mat']:.2f}", flush=True)
                del model
                torch.cuda.empty_cache()
    if psnr:
        (HERE / "e15e_psnr.json").write_text(json.dumps(psnr, indent=1), encoding="utf-8")


def bao_cao():
    anh = pd.read_csv(B0 / "kq_anh.csv", dtype={"image_id": str})
    sach = ~anh.leak.to_numpy()
    khoi = anh.khoi[sach].to_numpy()
    mi = lambda c: cac_miou(c)[0]["mIoU_co_mat"]
    out = []
    p = lambda t="": (out.append(t), print(t))
    p(__doc__.split("Chạy (env")[0].strip() + "\n")
    hcfg = json.loads(FH.read_text(encoding="utf-8"))
    p("h NLM chọn trên ảnh train: " + ", ".join(f"{dk} h={hcfg[dk]['h']} (MSE {hcfg[dk]['mse_tho']:.0f} -> "
                                                 f"{hcfg[dk]['mse_theo_h'][str(hcfg[dk]['h'])]:.0f})" for dk in DK))
    fp = HERE / "e15e_psnr.json"
    if fp.exists():
        p("PSNR VIS so với VIS sạch (test 280): " + ", ".join(f"{k} {v:.1f} dB" for k, v in json.loads(fp.read_text()).items()))
    C = {(s, bt, dk, c): np.load(ten(s, bt, dk, c))["conf"][sach] for s in SEEDS for bt in BT for dk in DK for c in CACH}

    for dk in DK:
        p(f"\n== {dk}: mIoU 273 ảnh (TB 3 seed; trong ngoặc từng seed) ==")
        for c in CACH:
            p(f"  {c:3s}  " + "  ".join(
                f"{bt}: {np.mean([mi(C[(s, bt, dk, c)].sum(0)) for s in SEEDS]):.2f} "
                f"({'/'.join(f'{mi(C[(s, bt, dk, c)].sum(0)):.1f}' for s in SEEDS)})" for bt in BT))

        rng = np.random.default_rng(SEED)
        bs = {k: [] for k in [(s, x) for s in SEEDS + ["gop"] for x in ("raw", "dn", "delta")]}
        for idx in boot_idx(khoi, rng):
            v = {k: mi(C[k][idx].sum(0)) for k in C if k[2] == dk}
            g = {(s, c): v[(s, "tb", dk, c)] - v[(s, "sage", dk, c)] for s in SEEDS for c in CACH}
            for s in SEEDS:
                bs[(s, "raw")].append(g[(s, "raw")])
                bs[(s, "dn")].append(g[(s, "dn")])
                bs[(s, "delta")].append(g[(s, "raw")] - g[(s, "dn")])
            for x, f in [("raw", lambda s: g[(s, "raw")]), ("dn", lambda s: g[(s, "dn")]),
                         ("delta", lambda s: g[(s, "raw")] - g[(s, "dn")])]:
                bs[("gop", x)].append(np.mean([f(s) for s in SEEDS]))
        G = {(s, c): mi(C[(s, "tb", dk, c)].sum(0)) - mi(C[(s, "sage", dk, c)].sum(0)) for s in SEEDS for c in CACH}
        goc = {}
        for s in SEEDS:
            goc[(s, "raw")], goc[(s, "dn")] = G[(s, "raw")], G[(s, "dn")]
            goc[(s, "delta")] = G[(s, "raw")] - G[(s, "dn")]
        for x in ("raw", "dn", "delta"):
            goc[("gop", x)] = np.mean([goc[(s, x)] for s in SEEDS])
        kq = {k: (goc[k], *ci(bs[k])) for k in bs}
        o = lambda k: f"{kq[k][0]:+.2f}{'*' if kq[k][1] > 0 or kq[k][2] < 0 else ' '} [{kq[k][1]:+.2f}; {kq[k][2]:+.2f}]"
        p(f"  {'':6s} {'g_raw = tb−sage thô':28s} {'g_dn = tb−sage khử nhiễu':28s} Δ = g_raw − g_dn")
        for s in SEEDS + ["gop"]:
            p(f"  {('seed ' + str(s)) if s != 'gop' else 'gộp 3':6s} {o((s, 'raw')):28s} {o((s, 'dn')):28s} {o((s, 'delta'))}")
        if dk == "toi3":
            d_ok = all(kq[(s, "delta")][0] > 0 and kq[(s, "delta")][1] > 0 for s in SEEDS)
            d_gop = kq[("gop", "delta")][0]
            dn_lo = kq[("gop", "dn")][1]
            n_chua0 = sum(kq[(s, "delta")][1] <= 0 <= kq[(s, "delta")][2] for s in SEEDS)
            if d_ok and d_gop >= 1.0 and dn_lo <= 0:
                kl = "XÁC NHẬN – nhiễu VIS là nguyên nhân chính khiến SAGE thua tb khi tối"
            elif d_ok and d_gop >= 1.0:
                kl = "MỘT PHẦN – khử nhiễu thu hẹp khoảng cách nhưng tb vẫn hơn SAGE có ý nghĩa"
            elif d_gop < 0.5 or n_chua0 >= 2:
                kl = "BÁC – nhiễu không phải nguyên nhân chính"
            else:
                kl = "LƯNG CHỪNG"
            p(f"\n== K1 (toi3): Δ>0 & CI>0 cả 3 seed: {d_ok}; Δ gộp {d_gop:+.2f}; g_dn gộp CI dưới {dn_lo:+.2f} -> {kl}")
    (HERE / "e15e.txt").write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chon_h", action="store_true")
    ap.add_argument("--chi_bao_cao", action="store_true")
    a = ap.parse_args()
    if a.chon_h:
        chon_h()
    else:
        if not a.chi_bao_cao:
            chay()
        bao_cao()
