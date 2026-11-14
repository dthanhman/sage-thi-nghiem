"""Kiểm tra: (1) model tái tạo ảnh mẫu của SAGE; (2) phase/ECC có dịch đúng chiều trên ảnh thật."""
import sys, cv2, numpy as np, torch
sys.argv=['x']
import run_thu as R
from pathlib import Path
h=R.HERE/'SAGE_IVIF'
ir=cv2.cvtColor(cv2.imread(str(h/'data/test/Ir/00000.png')),cv2.COLOR_BGR2GRAY)
yc=cv2.cvtColor(cv2.imread(str(h/'data/test/Vis/00000.png')),cv2.COLOR_BGR2YCrCb)
ref=cv2.cvtColor(cv2.imread(str(h/'result/test/00000.png')),cv2.COLOR_BGR2YCrCb)[:,:,0]
net=R.Network();net.load_state_dict(torch.load(h/'checkpoints/checkpoints.pt',map_location='cpu',weights_only=True));net.cuda().eval()
t=lambda a:torch.from_numpy(a).float().div(255)[None,None].cuda()
with torch.inference_mode(): f=net(t(yc[:,:,0]),t(ir))[0].clamp(0,1)[0,0].cpu().numpy()
print('kich thuoc',f.shape,ref.shape)
mse=((f*255-ref.astype(float))**2).mean();print('PSNR so voi anh mau SAGE: %.1f dB'%(10*np.log10(255**2/mse)))
# ảnh giống hệt: ms-ssim phải bằng 1
k=R.gauss('cuda');print('ms_ssim(a,a)=',R.ms_ssim(t(ir),t(ir),k))
# chiều dịch: IR dịch rồi căn phải ra lượng ngược lại, dùng chính ảnh làm 2 nguồn
e=R.canh(ir);w=cv2.createHanningWindow((ir.shape[1],ir.shape[0]),cv2.CV_32F)
for dx,dy in [(5,0),(0,-5),(10,0),(2,0)]:
    s=R.dich(ir,dx,dy);es=R.canh(s)
    print('dich',(dx,dy),'can phase',np.round(R.can_phase(e,es,w)[:2],2),'can ecc',np.round(R.can_ecc(e,es)[:2],2))
