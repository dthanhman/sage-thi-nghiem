"""Kiểm manifest với ảnh FMB thật, và (nếu có) 12 last.pt với artifact_manifest.json.
Chạy: python kiem/kiem_md5.py --fmb <FMB chứa train/Visible, test/Visible> [--ckpt <thư mục chứa <run_id>/last.pt>]"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

B3 = Path(__file__).resolve().parents[1]


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while k := f.read(1 << 20):
            h.update(k)
    return h.hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument("--fmb")
ap.add_argument("--ckpt")
a = ap.parse_args()
R = {k: list(csv.DictReader(open(B3 / f"manifests/{k}.csv", encoding="utf-8-sig"))) for k in ("train", "dev", "test")}
ids = {k: {r["image_id"] for r in v} for k, v in R.items()}
print("số dòng / ID khác nhau:", {k: (len(R[k]), len(ids[k])) for k in R})
print("giao train-dev, train-test, dev-test:", len(ids["train"] & ids["dev"]), len(ids["train"] & ids["test"]), len(ids["dev"] & ids["test"]))
print("leak:", [r["image_id"] for r in R["test"] if r["leak"] == "True"])
moi = sum(R.values(), [])
print("số ảnh VIS trùng hệt (md5):", len(moi) - len({r["vis_md5"] for r in moi}))
if a.fmb:
    thieu = lech = 0
    for r in moi:
        for k in ("vis", "ir", "label"):
            p = Path(a.fmb) / r[f"{k}_path"]
            if not p.exists():
                thieu += 1
            elif md5(p) != r[f"{k}_md5"]:
                lech += 1
    print(f"ảnh: {len(moi)} bộ VIS/IR/nhãn, thiếu {thieu} file, lệch md5 {lech} file")
if a.ckpt:
    runs = json.loads((B3 / "artifact_manifest.json").read_text(encoding="utf-8"))["runs"]
    sai = [r["run_id"] for r in runs if not (p := Path(a.ckpt) / r["run_id"] / "last.pt").exists() or md5(p) != r["checkpoint_md5"]]
    print(f"checkpoint: khớp {len(runs) - len(sai)}/{len(runs)}", "lệch/thiếu: " + ", ".join(sai) if sai else "")
