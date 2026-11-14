# So running_mean/var của BN đầu tiên với giá trị tính lại trên crop 192x256 từ mỗi bộ dữ liệu
import numpy as np, cv2, glob, sys
rng = np.random.default_rng(0)
sd = np.load(sys.argv[1], allow_pickle=True).item()
W, b = sd['module1.0.weight'], sd['module1.0.bias']
rm, rv = sd['module1.1.running_mean'], sd['module1.1.running_var']
def tinh(vis, ir, ten, n_crop=4):
    outs = []
    for v, i in zip(vis, ir):
        y = cv2.cvtColor(cv2.imread(v), cv2.COLOR_BGR2YCrCb)[...,0].astype(np.float32)/255
        r = cv2.imread(i, 0).astype(np.float32)/255
        for _ in range(n_crop):
            t, l = rng.integers(0, y.shape[0]-192), rng.integers(0, y.shape[1]-256)
            yc, rc = y[t:t+192, l:l+256], r[t:t+192, l:l+256]
            o = np.stack([cv2.filter2D(yc, -1, W[c,0], borderType=cv2.BORDER_CONSTANT) +
                          cv2.filter2D(rc, -1, W[c,1], borderType=cv2.BORDER_CONSTANT) + b[c] for c in range(32)])
            outs.append(o.reshape(32, -1))
    o = np.concatenate(outs, 1)
    m, va = o.mean(1), o.var(1)
    em = np.abs(m - rm).mean() / np.abs(rm).mean()
    ev = np.median(va / rv)
    print(f'{ten:24s} lech mean TB={em:6.1%}   ty le var (tinh/luu) trung vi={ev:.2f}  [P10 {np.percentile(va/rv,10):.2f}, P90 {np.percentile(va/rv,90):.2f}]')
F = 'D:/CS406/DoAn/Sage/xulianh/data/raw/FMB'
vis, ir = sorted(glob.glob(f'{F}/train/Visible/*.png')), sorted(glob.glob(f'{F}/train/Infrared/*.png'))
idx = rng.choice(len(vis), 200, replace=False)
tinh([vis[k] for k in idx], [ir[k] for k in idx], 'FMB train (200 anh)')
vis, ir = sorted(glob.glob(f'{F}/test/Visible/*.png')), sorted(glob.glob(f'{F}/test/Infrared/*.png'))
tinh(vis[:100], ir[:100], 'FMB test (100 anh)')
M = 'D:/CS406/DoAn/Sage/thu_viec3/SAGE_IVIF/data/test'
tinh(sorted(glob.glob(f'{M}/Vis/*.png')), sorted(glob.glob(f'{M}/Ir/*.png')), 'M3FD 6 anh mau', n_crop=30)
