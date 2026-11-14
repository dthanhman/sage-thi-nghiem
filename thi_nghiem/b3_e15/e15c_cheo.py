"""E15c – Kết quả tối của E15 là do ảnh SAGE kém, hay do model phân đoạn "sage" lạ ảnh? (seed 0, ~15 phút GPU)

Chấm chéo: ảnh đầu vào {sage, tb} (mức sach, toi3) × model B3 {vis, ir, sage, tb} (seed 0). 273 ảnh sạch.
E15b: khi tối, SAGE lấy IR ở nền 0,74 (sạch chỉ 0,20) -> ảnh SAGE tối giống IR hơn ảnh SAGE sạch mà model "sage" đã học.

TIÊU CHÍ ĐẶT TRƯỚC (ghi 14/11, trước khi chạy), mIoU_co_mat, CI95 bootstrap theo khối:
  H1 (lệch phân phối): trên ảnh SAGE toi3, model tốt nhất KHÁC "sage" hơn model "sage" ≥ +2 và CI > 0
      -> ĐÚNG: kết quả tối của E15 là do model lạ ảnh, không dùng làm gap.
      Nếu model "sage" vẫn tốt nhất hoặc hơn < +2 -> SAI: ảnh SAGE tối thật sự kém.
  H2 (so ảnh, mỗi ảnh lấy model tốt nhất cho nó): max_model(SAGE toi3) − max_model(tb toi3)
      CI < 0 -> ảnh SAGE tối kém ảnh tb (gap thật); CI chứa 0 -> ngang nhau; CI > 0 -> SAGE hơn.
      (Lưu ý: chọn max trên test là lạc quan như nhau cho cả hai.)
Kèm: hình anh_mau/luoi_{id}.jpg (VIS hỏng | SAGE | tb | IR ở sach / toi3 / suong3) để nhìn vùng người khi sương.
Chạy: python e15c_cheo.py -> e15c.txt
"""
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
from train_seg import du_doan, tao_model  # noqa: E402

S = 0
VAO = ["sage", "tb"]
MODEL = ["vis", "ir", "sage", "tb"]
DK = ["sach", "toi3"]
CONF = OUT / "conf_cheo"


def chay():
    df = pd.read_csv(MAN / "test.csv", dtype={"image_id": str})
    sage = GhepSAGE("cuda")
    CONF.mkdir(parents=True, exist_ok=True)
    nhan = [doc_nhan(duong(r.label_path), lop0=True) for r in df.itertuples()]
    vis0 = [cv2.imread(str(duong(r.vis_path))) for r in df.itertuples()]
    irs = [cv2.cvtColor(cv2.imread(str(duong(r.ir_path))), cv2.COLOR_BGR2GRAY) for r in df.itertuples()]
    for dk in DK:
        viec = [(v, m) for v in VAO for m in MODEL if not (CONF / f"s{S}_{v}_m{m}_{dk}.npz").exists()]
        if not viec:
            continue
        vh = [lam_hong(v, dk, r.image_id) for v, r in zip(vis0, df.itertuples())]
        vao = {v: [dau_vao(v, a, ir, sage) for a, ir in zip(vh, irs)] for v in VAO}
        del vh
        for m in MODEL:
            ds = [v for v, mm in viec if mm == m]
            if not ds:
                continue
            model = tao_model("b3", 15)
            model.load_state_dict(torch.load(tim_run(m, S) / "last.pt", map_location="cpu", weights_only=True))
            model.to("cuda").eval()
            for v in ds:
                t0 = time.time()
                conf = np.zeros((len(df), 15, 15), np.int64)
                for i, img in enumerate(vao[v]):
                    x = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - MEAN) / STD
                    x = torch.from_numpy(x.transpose(2, 0, 1).copy())[None].cuda()
                    conf[i] = nham(nhan[i], du_doan(model, x).argmax(1)[0].cpu().numpy(), 15)
                np.savez_compressed(CONF / f"s{S}_{v}_m{m}_{dk}.npz", conf=conf)
                print(f"ảnh {v:4s} model {m:4s} {dk}: {time.time() - t0:.0f} s | "
                      f"mIoU {cac_miou(conf.sum(0))[0]['mIoU_co_mat']:.2f}", flush=True)
            del model
            torch.cuda.empty_cache()


def luoi():
    """Ghép ảnh mẫu (đã lưu ở E15, seed 0) thành lưới để nhìn."""
    df = pd.read_csv(MAN / "test.csv", dtype={"image_id": str}).set_index("image_id")
    for iid in ("00089", "00518"):
        ir = cv2.imread(str(duong(df.loc[iid].ir_path)))
        hang = []
        for dk in ("sach", "toi3", "suong3"):
            o = [cv2.imread(str(OUT / "anh_mau" / f"{iid}_{dk}_{b}.jpg")) for b in ("vis", "sage", "tb")] + [ir]
            o = [cv2.resize(a, (400, 300)) for a in o]
            for a, t in zip(o, (f"VIS {dk}", "SAGE", "tb", "IR")):
                cv2.putText(a, t, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            hang.append(np.hstack(o))
        cv2.imwrite(str(OUT / "anh_mau" / f"luoi_{iid}.jpg"), np.vstack(hang), [cv2.IMWRITE_JPEG_QUALITY, 88])


def bao_cao():
    anh = pd.read_csv(B0 / "kq_anh.csv", dtype={"image_id": str})
    sach = ~anh.leak.to_numpy()
    khoi = anh.khoi[sach].to_numpy()
    mi = lambda c: cac_miou(c)[0]["mIoU_co_mat"]
    out = []
    p = lambda t="": (out.append(t), print(t))
    p(__doc__.split("Chạy:")[0].strip() + "\n")
    C = {}
    for dk in DK:
        for v in VAO:
            for m in MODEL:
                f = CONF / f"s{S}_{v}_m{m}_{dk}.npz"
                C[(v, m, dk)] = np.load(f)["conf"][sach]
    for dk in DK:
        t = pd.DataFrame({f"model {m}": [mi(C[(v, m, dk)].sum(0)) for v in VAO] for m in MODEL},
                         index=[f"ảnh {v}" for v in VAO])
        p(f"== mIoU, mức {dk} (hàng = ảnh đầu vào, cột = model; đường chéo sage/sage, tb/tb = số E15)\n"
          + t.round(2).to_string() + "\n")

    ten = {v: max(MODEL, key=lambda m: mi(C[(v, m, "toi3")].sum(0))) for v in VAO}
    rng = np.random.default_rng(SEED)
    d1, d2 = [], []
    for bi in boot_idx(khoi, rng):
        v = {k: mi(c[bi].sum(0)) for k, c in C.items() if k[2] == "toi3"}
        d1.append(v[("sage", ten["sage"], "toi3")] - v[("sage", "sage", "toi3")])
        d2.append(v[("sage", ten["sage"], "toi3")] - v[("tb", ten["tb"], "toi3")])
    g1 = mi(C[("sage", ten["sage"], "toi3")].sum(0)) - mi(C[("sage", "sage", "toi3")].sum(0))
    g2 = mi(C[("sage", ten["sage"], "toi3")].sum(0)) - mi(C[("tb", ten["tb"], "toi3")].sum(0))
    c1, c2 = ci(d1), ci(d2)
    p("== Tiêu chí (toi3, seed 0)")
    p(f"H1 ảnh SAGE: model tốt nhất = {ten['sage']}; hơn model sage {g1:+.2f} [{c1[0]:+.2f}; {c1[1]:+.2f}] -> "
      + ("ĐÚNG: model sage lạ ảnh" if ten["sage"] != "sage" and g1 >= 2 and c1[0] > 0 else "SAI: ảnh SAGE tối thật sự kém"))
    p(f"H2 ảnh SAGE (model {ten['sage']}) − ảnh tb (model {ten['tb']}) = {g2:+.2f} [{c2[0]:+.2f}; {c2[1]:+.2f}] -> "
      + ("ảnh SAGE kém hơn tb" if c2[1] < 0 else "ảnh SAGE hơn tb" if c2[0] > 0 else "ngang nhau"))
    (HERE / "e15c.txt").write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    chay()
    luoi()
    bao_cao()
