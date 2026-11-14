"""Bước 2: train SegFormer trên ảnh (đã ghép) của FMB train, chọn checkpoint tốt nhất trên dev.

Không đụng tới test. Mỗi biến thể ảnh train một model riêng (giống giao thức bài SAGE).

Ví dụ:
  python train_seg.py --bien_the sage                    # MiT-B0, mặc định 100 epoch
  python train_seg.py --bien_the vis --model b1
  python train_seg.py --bien_the sage --epochs 1 --eval_moi 50   # chạy thử
"""
import argparse
import functools
import json
import math
import os
import random
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from chung import ANH, FMB, IGNORE, MAN, MEAN, NC, RUNS, STD, doc_nhan, miou, nham, thu_muc

MODEL = {"b0": "nvidia/mit-b0", "b1": "nvidia/mit-b1", "b2": "nvidia/mit-b2", "b3": "nvidia/mit-b3"}


class FMBSeg(Dataset):
    def __init__(self, split, bien_the, train, crop=512, scale=(0.5, 2.0)):
        self.df = pd.read_csv(MAN / f"{split}.csv", dtype={"image_id": str})
        self.bt, self.train, self.crop, self.scale = bien_the, train, crop, scale

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = cv2.imread(str(ANH / self.bt / thu_muc(r) / f"{r.image_id}.png"))
        assert img is not None, f"thiếu ảnh {self.bt}/{r.image_id}, chạy xuat_anh.py trước"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lab = doc_nhan(FMB / r.label_path)
        if self.train:
            s = random.uniform(*self.scale)
            h, w = int(img.shape[0] * s + 0.5), int(img.shape[1] * s + 0.5)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            lab = cv2.resize(lab.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int64)
            ph, pw = max(self.crop - h, 0), max(self.crop - w, 0)
            if ph or pw:
                img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
                lab = cv2.copyMakeBorder(lab.astype(np.uint8), 0, ph, 0, pw, cv2.BORDER_CONSTANT,
                                         value=IGNORE).astype(np.int64)
            y0 = random.randint(0, img.shape[0] - self.crop)
            x0 = random.randint(0, img.shape[1] - self.crop)
            img = img[y0:y0 + self.crop, x0:x0 + self.crop]
            lab = lab[y0:y0 + self.crop, x0:x0 + self.crop]
            if random.random() < 0.5:
                img, lab = img[:, ::-1], lab[:, ::-1]
        x = (img.astype(np.float32) / 255 - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), torch.from_numpy(lab.copy())


def _seed_worker(w, seed):
    random.seed(seed * 1000 + w)
    np.random.seed(seed * 1000 + w)


def tao_model(ten):
    from transformers import SegformerForSemanticSegmentation
    return SegformerForSemanticSegmentation.from_pretrained(
        MODEL[ten], num_labels=NC, id2label={i: str(i) for i in range(NC)},
        label2id={str(i): i for i in range(NC)}, ignore_mismatched_sizes=True)


@torch.inference_mode()
def du_doan(model, x):
    with torch.autocast("cuda", dtype=torch.float16):
        lg = model(pixel_values=x).logits
    return F.interpolate(lg.float(), size=x.shape[-2:], mode="bilinear", align_corners=False)


def danh_gia(model, loader):
    model.eval()
    c = np.zeros((NC, NC), np.int64)
    for x, y in loader:
        p = du_doan(model, x.cuda()).argmax(1).cpu().numpy()
        for pi, yi in zip(p, y.numpy()):
            c += nham(yi, pi)
    model.train()
    return miou(c)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bien_the", required=True)
    ap.add_argument("--model", default="b0", choices=list(MODEL))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--eval_moi", type=int, default=500, help="đánh giá dev mỗi bao nhiêu iter")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tiep", action="store_true", help="train tiếp từ resume.pt nếu có")
    ap.add_argument("--dung_sau_ep", type=int, default=0, help="chỉ để thử: tự dừng sau N epoch (giả lập bị ngắt)")
    a = ap.parse_args()

    if sys.platform == "win32":  # giữ máy không ngủ khi đang train; hết hiệu lực khi process thoát
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    run = RUNS / f"{a.bien_the}_{a.model}_s{a.seed}"
    run.mkdir(parents=True, exist_ok=True)

    tr = FMBSeg("train", a.bien_the, True, a.crop)
    dev = FMBSeg("dev", a.bien_the, False)
    g = torch.Generator().manual_seed(a.seed)
    lt = DataLoader(tr, a.batch, shuffle=True, drop_last=True, num_workers=a.workers,
                    pin_memory=True, persistent_workers=a.workers > 0, generator=g,
                    worker_init_fn=functools.partial(_seed_worker, seed=a.seed))
    ld = DataLoader(dev, 2, num_workers=a.workers)

    model = tao_model(a.model).cuda().train()
    enc = [p for n, p in model.named_parameters() if n.startswith("segformer.")]
    head = [p for n, p in model.named_parameters() if not n.startswith("segformer.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": a.lr}, {"params": head, "lr": a.lr * 10}],
                            lr=a.lr, weight_decay=0.01)
    tong = a.epochs * len(lt)
    warm = min(1500, tong // 10)
    # warmup tuyến tính rồi giảm poly (power 1), giống mmseg/SegFormer
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: (it + 1) / warm if it < warm else max(0.0, 1 - (it - warm) / max(1, tong - warm)))
    scaler = torch.amp.GradScaler("cuda")

    cfg = vars(a) | {"iters": tong, "warmup": warm, "n_train": len(tr), "n_dev": len(dev),
                     "init": MODEL[a.model], "loss": "CE, bỏ nhãn 0", "optimizer": "AdamW wd 0.01, head lr x10",
                     "aug": "scale 0.5-2.0, random crop, hflip", "torch": torch.__version__}
    cfg.pop("tiep", None)
    cfg.pop("dung_sau_ep", None)

    best, it, t0, dl, ep0, da_chay = -1.0, 0, time.time(), [], 0, 0.0
    rp = run / "resume.pt"
    if a.tiep and rp.exists():
        # train tiếp từ cuối epoch đã lưu (không khớp từng bit: worker augmentation khởi động lại RNG)
        ck = torch.load(rp, map_location="cpu", weights_only=False)
        bo = lambda c: {k: v for k, v in c.items() if k != "workers"}  # đổi số worker vẫn train tiếp được
        if bo(ck["cfg"]) != bo(cfg):
            raise RuntimeError(f"cấu hình khác lần trước, không train tiếp được:\n{ck['cfg']}\n{cfg}")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sch.load_state_dict(ck["sch"])
        scaler.load_state_dict(ck["scaler"])
        best, it, ep0, dl, da_chay = ck["best"], ck["it"], ck["ep"], ck["dl"], ck["giay"]
        g.set_state(ck["rng_gen"])
        torch.set_rng_state(ck["rng_torch"])
        torch.cuda.set_rng_state(ck["rng_cuda"])
        random.setstate(ck["rng_py"])
        np.random.set_state(ck["rng_np"])
        log = open(run / "log.csv", "a", encoding="utf-8")
        print(f"{run.name}: TRAIN TIẾP từ epoch {ep0}, iter {it}/{tong}, dev tốt nhất {best:.2f}", flush=True)
    else:
        (run / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        log = open(run / "log.csv", "w", encoding="utf-8")
        log.write("iter,epoch,loss,lr,dev_miou,giay\n")
        print(f"{run.name}: {len(tr)} train, {len(dev)} dev, {tong} iter", flush=True)
    t0 -= da_chay  # cột "giay" tính cả thời gian các lần chạy trước

    for ep in range(ep0, a.epochs):
        for x, y in lt:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                lg = model(pixel_values=x).logits
            lg = F.interpolate(lg.float(), size=y.shape[-2:], mode="bilinear", align_corners=False)
            loss = F.cross_entropy(lg, y, ignore_index=IGNORE)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sch.step()
            it += 1
            dl.append(loss.item())
            if it % a.eval_moi == 0 or it == tong:
                m = danh_gia(model, ld)
                l = float(np.mean(dl))
                dl = []
                log.write(f"{it},{ep + 1},{l:.4f},{sch.get_last_lr()[0]:.2e},{m:.2f},{time.time() - t0:.0f}\n")
                log.flush()
                tot = m > best
                if tot:
                    best = m
                    torch.save(model.state_dict(), run / "best.pt")
                print(f"iter {it}/{tong} ep {ep + 1} loss {l:.3f} dev mIoU {m:.2f}"
                      f"{' *' if tot else ''}  {time.time() - t0:.0f}s", flush=True)
            if not math.isfinite(loss.item()):
                raise RuntimeError("loss NaN/inf")
        # lưu trạng thái cuối mỗi epoch (~1 phút/lần) để train tiếp nếu bị ngắt
        tmp = run / "resume.tmp"
        torch.save({"cfg": cfg, "model": model.state_dict(), "opt": opt.state_dict(), "sch": sch.state_dict(),
                    "scaler": scaler.state_dict(), "best": best, "it": it, "ep": ep + 1, "dl": dl,
                    "giay": time.time() - t0, "rng_gen": g.get_state(), "rng_torch": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state(), "rng_py": random.getstate(),
                    "rng_np": np.random.get_state()}, tmp)
        os.replace(tmp, rp)
        if a.dung_sau_ep and ep + 1 - ep0 >= a.dung_sau_ep:
            print(f"Dừng thử sau epoch {ep + 1} (resume.pt đã lưu)", flush=True)
            return
    torch.save(model.state_dict(), run / "last.pt")
    cfg["best_dev_miou"] = best
    (run / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Xong. dev mIoU tốt nhất {best:.2f}. Checkpoint: {run / 'best.pt'}")


if __name__ == "__main__":
    main()
