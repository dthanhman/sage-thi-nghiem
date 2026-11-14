# Giải ngược E[Y], E[IR] của dữ liệu train từ BN đầu tiên của Student
import numpy as np, cv2, glob, sys
sd = np.load(sys.argv[1], allow_pickle=True).item()
W = sd['module1.0.weight']; b = sd['module1.0.bias']; rm = sd['module1.1.running_mean']
A = np.stack([W[:,0].sum((1,2)), W[:,1].sum((1,2))], 1)   # 32x2
x, res, *_ = np.linalg.lstsq(A, rm - b, rcond=None)
pred = A @ x + b
print(f'Giai nguoc: E[Y]={x[0]:.3f}  E[IR]={x[1]:.3f}  | sai so khop R2={1-((rm-pred)**2).sum()/((rm-rm.mean())**2).sum():.4f}')
def thong_ke(vis, ir, ten):
    ys, irs = [], []
    for v, i in zip(vis, ir):
        ys.append(cv2.cvtColor(cv2.imread(v), cv2.COLOR_BGR2YCrCb)[...,0].mean()/255)
        irs.append(cv2.imread(i, 0).mean()/255)
    ys, irs = np.array(ys), np.array(irs)
    print(f'{ten:22s} n={len(ys):4d}  E[Y]={ys.mean():.3f} (P10-P90 {np.percentile(ys,10):.2f}-{np.percentile(ys,90):.2f})  E[IR]={irs.mean():.3f} (P10-P90 {np.percentile(irs,10):.2f}-{np.percentile(irs,90):.2f})')
F = 'D:/CS406/DoAn/Sage/xulianh/data/raw/FMB'
for sp in ['train', 'test']:
    thong_ke(sorted(glob.glob(f'{F}/{sp}/Visible/*.png')), sorted(glob.glob(f'{F}/{sp}/Infrared/*.png')), f'FMB {sp}')
M = 'D:/CS406/DoAn/Sage/thu_viec3/SAGE_IVIF/data/test'
thong_ke(sorted(glob.glob(f'{M}/Vis/*.png')), sorted(glob.glob(f'{M}/Ir/*.png')), 'M3FD (6 anh mau repo)')
