"""Tổng hợp E0/E1/E2 từ kq_*.csv/npz. Bootstrap theo khối id liền nhau. Ghi ra ket_qua.md."""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
B, SEED = 2000, 0
METRIC = ["EN", "SD", "SF", "AG", "SCD", "MS_SSIM_sum", "Q_ABF", "NIQE"]  # BRISQUE PR6 nghi lỗi, bỏ
QUYET = ["SCD", "MS_SSIM_sum", "Q_ABF"]
USM = ["sage_usm0.25", "sage_usm0.5", "sage_usm1.0"]


def md(df):
    """DataFrame -> bảng markdown (không cần tabulate)."""
    h = "| " + " | ".join([str(df.index.name or "")] + [str(c) for c in df.columns]) + " |"
    rows = ["| " + " | ".join([str(i)] + ["" if pd.isna(v) else f"{v:g}" for v in r]) + " |"
            for i, r in zip(df.index, df.to_numpy())]
    return "\n".join([h, "|" + "---|" * (len(df.columns) + 1)] + rows)


def miou(c):
    """c: (15,15) GT x pred, bỏ hàng 0. mIoU % trên lớp có trong GT."""
    c = c[1:]
    tp = c[np.arange(14), np.arange(1, 15)]
    g, p = c.sum(1), c[:, 1:].sum(0)
    ok = g > 0
    iou = np.full(14, np.nan)
    iou[ok] = tp[ok] / (g[ok] + p[ok] - tp[ok])
    return np.nanmean(iou) * 100, iou * 100


def boot_idx(khoi, rng):
    """Sinh B bộ chỉ số ảnh, bốc lại theo khối."""
    ks = np.unique(khoi)
    nhom = [np.flatnonzero(khoi == k) for k in ks]
    for _ in range(B):
        yield np.concatenate([nhom[i] for i in rng.integers(len(ks), size=len(ks))])


def ci(v):
    return np.percentile(v, [2.5, 97.5])


def main():
    anh = pd.read_csv(HERE / "kq_anh.csv", dtype={"image_id": str})
    m = pd.read_csv(HERE / "kq_metric.csv", dtype={"image_id": str})
    z = np.load(HERE / "kq_seg.npz", allow_pickle=True)
    BT = list(z["bien_the"])
    conf = z["conf"]
    assert list(z["image_id"]) == list(anh.image_id)
    L = []
    p = L.append

    hop_le = m[METRIC].notna().all(axis=1).groupby(m.bien_the).sum().astype(int).to_dict()
    p(f"# Kết quả E0/E1/E2 – FMB test\n\nSố ảnh hợp lệ mỗi biến thể: {hop_le} (cần 280).\n")
    p("Proxy mIoU = SegFormer-B2 Cityscapes zero-shot, **không phải mIoU chính thức**. "
      "CI95 = bootstrap theo khối id (B=2000). BRISQUE của PR #6 ra ≈0 trên hầu hết ảnh nên bỏ.\n")

    for ten, chon in [("273 ảnh (bỏ 7 ảnh rò) – dùng để kết luận", ~anh.leak.to_numpy()),
                      ("280 ảnh", np.ones(len(anh), bool))]:
        ids = anh.image_id[chon].to_numpy()
        khoi = anh.khoi[chon].to_numpy()
        cf = conf[:, chon]
        mm = m[m.image_id.isin(ids)]
        p(f"\n## {ten}\n\n### E0 – bảng nền\n")
        tb = mm.groupby("bien_the")[METRIC].mean().reindex(BT)
        tb["mIoU_proxy"] = [miou(cf[j].sum(0))[0] for j in range(len(BT))]
        p(md(tb.round(3)) + "\n")
        p("↑ tốt hơn: EN, SD, SF, AG, SCD, MS-SSIM (tổng 2 nguồn, tối đa 2), Q_ABF, mIoU. ↓ tốt hơn: NIQE.\n"
          "Với `vis`, MS_SSIM_VIS = 1 và với `ir`, MS_SSIM_IR = 1 theo định nghĩa.\n")

        # IoU từng lớp
        names = ["road", "sidewalk", "building", "t_light", "t_sign", "veg", "sky", "person", "car",
                 "truck", "bus", "moto", "bicycle", "pole"]
        iou = pd.DataFrame([miou(cf[j].sum(0))[1] for j in range(len(BT))], index=BT, columns=names)
        p("IoU proxy từng lớp (%):\n\n" + md(iou.dropna(axis=1, how="all").round(1)) + "\n")

        # chênh theo cặp + bootstrap
        piv = {k: mm.pivot(index="image_id", columns="bien_the", values=k).loc[ids] for k in QUYET}
        rng = np.random.default_rng(SEED)
        idxs = list(boot_idx(khoi, rng))

        def d_metric(k, a, b):  # chênh tương đối % của trung bình
            x, y = piv[k][a].to_numpy(), piv[k][b].to_numpy()
            f = lambda i: (x[i].mean() / y[i].mean() - 1) * 100
            return f(np.arange(len(x))), ci([f(i) for i in idxs])

        def d_miou(a, b):  # chênh tuyệt đối, điểm %
            ja, jb = BT.index(a), BT.index(b)
            f = lambda i: miou(cf[ja, i].sum(0))[0] - miou(cf[jb, i].sum(0))[0]
            return f(np.arange(cf.shape[1])), ci([f(i) for i in idxs])

        p("\n### E1 – USM so với SAGE (chênh theo cặp)\n")
        p("| biến thể | ΔSCD % | ΔMS-SSIM % | ΔQ_ABF % | ΔmIoU proxy (điểm) | đạt 4 tiêu chí? |\n|---|---|---|---|---|---|")
        dat_e1 = []
        for u in USM:
            r = {k: d_metric(k, u, "sage") for k in QUYET}
            dm = d_miou(u, "sage")
            ok = (r["SCD"][0] >= 0.5 and r["SCD"][1][0] > 0 and r["MS_SSIM_sum"][0] >= 0.9
                  and r["MS_SSIM_sum"][1][0] > 0 and r["Q_ABF"][1][0] > 0 and dm[0] >= -0.5)
            dat_e1.append(ok)
            s = lambda t: f"{t[0]:+.2f} [{t[1][0]:+.2f}; {t[1][1]:+.2f}]"
            p(f"| {u} | {s(r['SCD'])} | {s(r['MS_SSIM_sum'])} | {s(r['Q_ABF'])} | {s(dm)} | {'CÓ' if ok else 'không'} |")

        p("\n### E2 – proxy mIoU, chênh theo cặp (điểm %)\n")
        p("| so sánh | Δ | CI95 | kết luận |\n|---|---|---|---|")
        kl = {}
        for a, b in [("sage", "vis"), ("sage", "tb"), ("sage", "ir"), ("tb", "vis"), ("ir", "vis")]:
            d, c = d_miou(a, b)
            kl[(a, b)] = "giúp" if c[0] > 0 else ("kém" if c[1] < 0 else "không thấy khác")
            p(f"| {a} − {b} | {d:+.2f} | [{c[0]:+.2f}; {c[1]:+.2f}] | {a} {kl[(a, b)]} |")

        if chon.sum() == 273:
            vis_m = miou(cf[BT.index("vis")].sum(0))[0]
            p("\n### Kết luận theo tiêu chí đặt trước (273 ảnh)\n")
            p(f"- Proxy hợp lệ (mIoU vis ≥ 20%): {'CÓ' if vis_m >= 20 else 'KHÔNG'} ({vis_m:.1f}%).")
            p(f"- E1: {'USM bù được → Gap A coi như chết' if any(dat_e1) else 'USM không bù đủ → Gap A chưa chết (xem metric nào trượt)'}.")
            p(f"- E2: SAGE so với VIS: **{kl[('sage', 'vis')]}**; SAGE so với ghép TB: **{kl[('sage', 'tb')]}**.")

    (HERE / "ket_qua.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
