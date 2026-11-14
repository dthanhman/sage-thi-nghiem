"""E15d – Ảnh SAGE khi VIS tối+nhiễu kém ảnh tb (E15c: −2,4*) có phải do SAGE tạo nhiễu/vệt không? (273 ảnh, GPU vài phút)

Ý tưởng: ở VÙNG PHẲNG (nơi cả VIS sạch lẫn IR đều gần như không có chi tiết) mọi dao động trong ảnh ghép là nhiễu/vệt.
  - Vùng phẳng: độ lệch chuẩn cục bộ (cửa sổ 9x9) của Y VIS SẠCH < 2,5 VÀ của IR < 2,5 (thang 0–255), co mask 5 px.
    Xác định từ ảnh nguồn sạch -> như nhau cho mọi phương pháp, mọi mức.
  - Thành phần cao tần HP = Y − GaussBlur(Y, sigma 4). Đo phương sai HP trong vùng phẳng.
  - Nhiễu THÊM VÀO = var_HP(mức tối) − var_HP(sạch), cộng gộp mọi ảnh (tính riêng SAGE, tb). Trừ mức sạch để bỏ
    phần kết cấu thật còn sót trong vùng phẳng.
  - Tham khảo tuyến tính: tb = (Yv + IR)/2 -> nhiễu VIS bị chia đôi. E15b: SAGE lấy IR ~0,74 ở nền khi toi3 -> nếu SAGE trộn
    tuyến tính thì nhiễu phải ÍT hơn tb. Nhiều hơn tb = SAGE khuếch đại nhiễu (phi tuyến).
  - Khám phá: tách 2 dải: mịn (Y − Gauss sigma 1) và vệt (Gauss sigma 1 − Gauss sigma 4).

TIÊU CHÍ ĐẶT TRƯỚC (ghi 14/11, trước khi chạy), CI95 bootstrap theo khối:
  N1 (chính) toi3: R = sqrt(nhiễu thêm SAGE / nhiễu thêm tb).
     R ≥ 1,5 và CI > 1           -> SAGE nhiễu/vệt hơn tb rõ (ủng hộ "ảnh SAGE tối kém do nhiễu/vệt")
     R ≤ 1,1 hoặc CI chứa 1      -> BÁC: ảnh SAGE tối kém không phải do nhiễu nhiều hơn
     còn lại                     -> LƯNG CHỪNG
     Kèm R ở toi1, toi2 (xem có tăng theo mức không).
  N2 (liên hệ với mIoU, theo ảnh, toi3): d = mIoU_ảnh(tb) − mIoU_ảnh(SAGE), cùng model phân đoạn "sage" seed 0
     (lấy từ e15/conf_cheo của E15c). Spearman rho giữa log(R_ảnh) và d.
     rho ≥ 0,2 và CI > 0 -> nhiễu giải thích một phần chỗ thua; CI chứa 0 -> không thấy liên hệ theo ảnh
     (mIoU từng ảnh nhiễu nên N2 chỉ là phụ).
Ra: e15d.txt, kq_e15d_anh.csv, e15/anh_mau/e15d_{id}.jpg (Y SAGE | Y tb | |HP| SAGE | |HP| tb ở toi3, cùng thang).
Chạy: python e15d_nhieu.py   (thêm --chi_bao_cao để in lại từ kq_e15d_anh.csv)
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e15_lam_hong_vis import B0, OUT, GhepSAGE, lam_hong  # noqa: E402
from chung import MAN, duong  # noqa: E402
from tong_hop import boot_idx, ci, SEED  # noqa: E402



def miou_anh(conf):
    """mIoU từng ảnh (%) – chép từ luu_tru/tim_gap_b0_0811/seg/e20_nen_ir.py (bản 15 lớp, lớp 0 = unlabeled)."""
    c = conf[:, 1:, :].astype(float)
    tp = np.stack([c[:, k, k + 1] for k in range(14)], 1)
    g, p = c.sum(2), c[:, :, 1:].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(g > 0, tp / (g + p - tp), np.nan)
    return np.nanmean(iou, 1) * 100


DK = ["sach", "toi1", "toi2", "toi3"]
BT = ["sage", "tb"]
DAI = ["hp", "min", "vet"]  # hp = sigma 4 (chính), min = < sigma 1, vet = sigma 1–4
NGUONG, CUA_SO, CO = 2.5, 9, 5
MIN_PX = 2000
CSV = HERE / "kq_e15d_anh.csv"
ANH_HINH = ["00089", "00518"]


def std_cuc_bo(y):
    y = y.astype(np.float32)
    m = cv2.blur(y, (CUA_SO, CUA_SO))
    return np.sqrt(np.maximum(cv2.blur(y * y, (CUA_SO, CUA_SO)) - m * m, 0))


def cac_dai(y):
    y = y.astype(np.float32)
    g1, g4 = cv2.GaussianBlur(y, (0, 0), 1), cv2.GaussianBlur(y, (0, 0), 4)
    return {"hp": y - g4, "min": y - g1, "vet": g1 - g4}


def do():
    anh = pd.read_csv(B0 / "kq_anh.csv", dtype={"image_id": str})
    man = pd.read_csv(MAN / "test.csv", dtype={"image_id": str})
    assert (anh.image_id.to_numpy() == man.image_id.to_numpy()).all(), "thứ tự kq_anh khác test.csv"
    man = man.set_index("image_id")
    sage = GhepSAGE("cuda" if torch.cuda.is_available() else "cpu")
    (OUT / "anh_mau").mkdir(parents=True, exist_ok=True)
    hang = []
    for i, r0 in enumerate(anh.itertuples()):
        if r0.leak:
            continue
        iid = r0.image_id
        r = man.loc[iid]
        vis = cv2.imread(str(duong(r.vis_path)))
        ir = cv2.cvtColor(cv2.imread(str(duong(r.ir_path))), cv2.COLOR_BGR2GRAY)
        y0 = cv2.cvtColor(vis, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        phang = (std_cuc_bo(y0) < NGUONG) & (std_cuc_bo(ir) < NGUONG)
        phang = cv2.erode(phang.astype(np.uint8), np.ones((CO, CO), np.uint8)).astype(bool)
        d = {"image_id": iid, "khoi": r0.khoi, "px_phang": int(phang.sum())}
        for dk in DK:
            yv8 = cv2.cvtColor(lam_hong(vis, dk, iid), cv2.COLOR_BGR2YCrCb)[:, :, 0]
            out = {"sage": sage(yv8, ir), "tb": ((yv8.astype(np.float64) + ir) / 2).round().astype(np.uint8),
                   "vis": yv8}
            for b, y in out.items():
                for k, h in cac_dai(y).items():
                    d[f"var_{k}_{b}_{dk}"] = float(h[phang].var()) if phang.any() else np.nan
            if dk == "toi3" and iid in ANH_HINH:
                hp = {b: np.abs(cac_dai(out[b])["hp"]) for b in BT}
                th = max(np.percentile(hp[b], 99) for b in BT)
                tm = lambda a: cv2.applyColorMap((np.clip(a / th, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
                o = [cv2.cvtColor(out["sage"], cv2.COLOR_GRAY2BGR), cv2.cvtColor(out["tb"], cv2.COLOR_GRAY2BGR),
                     tm(hp["sage"]), tm(hp["tb"])]
                o = [cv2.resize(a, (400, 300)) for a in o]
                for a, t in zip(o, ("Y SAGE toi3", "Y tb toi3", "|HP| SAGE", "|HP| tb")):
                    cv2.putText(a, t, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imwrite(str(OUT / "anh_mau" / f"e15d_{iid}.jpg"),
                            np.vstack([np.hstack(o[:2]), np.hstack(o[2:])]), [cv2.IMWRITE_JPEG_QUALITY, 88])
        hang.append(d)
        if len(hang) % 50 == 0:
            print(f"  {len(hang)} ảnh", flush=True)
    pd.DataFrame(hang).to_csv(CSV, index=False)


def bao_cao():
    df = pd.read_csv(CSV, dtype={"image_id": str})
    ok = (df.px_phang >= MIN_PX).to_numpy()
    out = []
    p = lambda t="": (out.append(t), print(t))
    p(__doc__.split("Ra:")[0].strip() + "\n")
    p(f"Vùng phẳng: {ok.sum()}/{len(df)} ảnh có ≥ {MIN_PX} px (trung vị {df.px_phang.median():.0f} px/ảnh, "
      f"{df.px_phang.median() / 480000 * 100:.1f}% ảnh); chỉ dùng các ảnh này.\n")
    D = df[ok].reset_index(drop=True)
    khoi = D.khoi.to_numpy()
    w = D.px_phang.to_numpy(float)

    def gop(col, idx):  # phương sai gộp, trọng số theo số pixel phẳng
        return (D[col].to_numpy()[idx] * w[idx]).sum() / w[idx].sum()

    def them(b, dk, k, idx):
        return gop(f"var_{k}_{b}_{dk}", idx) - gop(f"var_{k}_{b}_sach", idx)

    tat = np.arange(len(D))
    p("== Độ lệch chuẩn cao tần trong vùng phẳng (gộp, thang 0–255) – sqrt(var)")
    for k in DAI:
        p(f"  dải {k}:   " + "  ".join(f"{dk:>6s}" for dk in DK))
        for b in ["vis", "sage", "tb"]:
            p(f"    {b:5s}   " + "  ".join(f"{np.sqrt(gop(f'var_{k}_{b}_{dk}', tat)):6.2f}" for dk in DK))
    p()

    p("== Tỉ lệ nhiễu THÊM VÀO R = sqrt(thêm SAGE / thêm tb)   (>1: SAGE nhiễu hơn tb)")
    kq = {}
    for k in DAI:
        for dk in DK[1:]:
            rv = lambda idx: np.sqrt(max(them("sage", dk, k, idx), 0) / them("tb", dk, k, idx))
            rng = np.random.default_rng(SEED)
            c = ci([rv(bi) for bi in boot_idx(khoi, rng)])
            kq[(k, dk)] = (rv(tat), c)
            p(f"  dải {k:3s} {dk}: R = {rv(tat):.2f} [{c[0]:.2f}; {c[1]:.2f}]   (tham khảo VIS/tb: "
              f"{np.sqrt(them('vis', dk, k, tat) / them('tb', dk, k, tat)):.2f})")
    p()

    R, c = kq[("hp", "toi3")]
    tt = ("SAGE nhiễu/vệt hơn tb rõ (ủng hộ giả thuyết)" if R >= 1.5 and c[0] > 1
          else "BÁC: không phải do nhiễu nhiều hơn" if R <= 1.1 or c[0] <= 1 <= c[1] else "LƯNG CHỪNG")
    p(f"N1 toi3 dải hp: R = {R:.2f} [{c[0]:.2f}; {c[1]:.2f}] -> {tt}")

    # N2: liên hệ theo ảnh với mIoU (model sage seed 0, E15c)
    anh = pd.read_csv(B0 / "kq_anh.csv", dtype={"image_id": str})
    sach = ~anh.leak.to_numpy()
    cs = np.load(OUT / "conf_cheo" / "s0_sage_msage_toi3.npz")["conf"][sach]
    ct = np.load(OUT / "conf_cheo" / "s0_tb_msage_toi3.npz")["conf"][sach]
    assert (anh.image_id[sach].to_numpy() == df.image_id.to_numpy()).all()
    dm = (miou_anh(ct) - miou_anh(cs))[ok]
    ri = np.sqrt(np.maximum(D.var_hp_sage_toi3 - D.var_hp_sage_sach, 1e-6)
                 / np.maximum(D.var_hp_tb_toi3 - D.var_hp_tb_sach, 1e-6)).to_numpy()
    lr = np.log(ri)
    m = np.isfinite(dm) & np.isfinite(lr)
    rho = spearmanr(lr[m], dm[m])[0]
    rng = np.random.default_rng(SEED)
    rs = []
    for bi in boot_idx(khoi, rng):
        mm = m[bi]
        rs.append(spearmanr(lr[bi][mm], dm[bi][mm])[0])
    c2 = ci(rs)
    tt2 = ("nhiễu giải thích một phần chỗ thua" if rho >= 0.2 and c2[0] > 0
           else "không thấy liên hệ theo ảnh" if c2[0] <= 0 <= c2[1] else "LƯNG CHỪNG")
    p(f"N2 toi3: {m.sum()} ảnh, TB d = {np.nanmean(dm[m]):+.2f}; R theo ảnh trung vị {np.median(ri):.2f} "
      f"(P10 {np.percentile(ri, 10):.2f}, P90 {np.percentile(ri, 90):.2f}); "
      f"rho(log R, d) = {rho:+.2f} [{c2[0]:+.2f}; {c2[1]:+.2f}] -> {tt2}")
    q = np.quantile(lr[m], [1 / 3, 2 / 3])
    for ten, sel in (("R thấp", lr[m] < q[0]), ("R giữa", (lr[m] >= q[0]) & (lr[m] < q[1])), ("R cao", lr[m] >= q[1])):
        p(f"     {ten}: {sel.sum()} ảnh, TB d = {dm[m][sel].mean():+.2f}")
    (HERE / "e15d.txt").write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    if "--chi_bao_cao" not in sys.argv:
        do()
    bao_cao()
