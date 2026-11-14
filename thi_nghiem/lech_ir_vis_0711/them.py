import numpy as np, pandas as pd, cv2, torch, sys
import run_thu as R
df=pd.read_csv('ket_qua.csv',dtype={'image_id':str})
p=lambda c,d,m='ms_ssim_tong':df[(df.cach==c)&(df.d==d)].set_index('image_id')[m].sort_index()
b=p('none',0)
print('--- 3 cách tính hồi phục ở MS-SSIM tổng ---')
for c in ('phase','ecc'):
  for d in (5,10):
    n,s=p('none',d),p(c,d)
    a=(s-n).median()/(b-n).median()*100
    ca=(s-n).mean()/(b-n).mean()*100
    e=(s.median()-n.median())/(b.median()-n.median())*100
    print(c,d,'median hieu/median mat %.1f | mean/mean %.1f | theo median muc %.1f'%(a,ca,e))
print('--- so với chính cách đó ở d=0 ---')
for c in ('none','ecc','phase'):
  for d in (5,10):
    x=(p(c,d)-p(c,0))/p(c,0)*100
    print(c,d,'median %.2f%%'%x.median())
# ảnh tiêu biểu: gần median mức giảm d=10
x=((p('none',10)-b)/b*100); iid=(x-x.median()).abs().idxmin()
rows=pd.read_csv('dev.csv',dtype={'image_id':str}).set_index('image_id');r=rows.loc[iid]
root=R.HERE.parent/'xulianh/data/raw/FMB'
y=cv2.cvtColor(cv2.imread(str(root/r.vis_path)),cv2.COLOR_BGR2YCrCb)[:,:,0]
ir=cv2.cvtColor(cv2.imread(str(root/r.ir_path)),cv2.COLOR_BGR2GRAY)
net=R.Network();net.load_state_dict(torch.load('SAGE_IVIF/checkpoints/checkpoints.pt',map_location='cpu',weights_only=True));net.cuda().eval()
t=lambda a:torch.from_numpy(a).float().div(255)[None,None].cuda()
hx,hy=R.huong(iid);print(iid,'huong',hx,hy,'giam %.2f%%'%x[iid])
def run(i):
    with torch.inference_mode(): return (net(t(y),t(i))[0].clamp(0,1)[0,0].cpu().numpy()*255).astype(np.uint8)
ird=R.dich(ir,hx*10,hy*10)
sx,sy,_=R.can_ecc(R.canh(y),R.canh(ird)); ie=R.ap_dich(ird,sx,sy)
f0,f10,fe=run(ir),run(ird),run(ie)
d=cv2.boxFilter(np.abs(f0.astype(float)-f10),-1,(200,150)); yy,xx=np.unravel_index(d[100:500,120:680].argmax(),(400,560)); yy+=100;xx+=120
c=lambda a:cv2.resize(a[yy-75:yy+75,xx-100:xx+100],None,fx=3,fy=3,interpolation=cv2.INTER_NEAREST)
top=np.hstack([c(y),c(ir)]);bot=np.hstack([c(f0),c(f10)]);row3=np.hstack([c(fe),np.abs(c(f10).astype(int)-c(f0)).astype(np.uint8)*3])
cv2.imwrite('xem_bong_doi.png',np.vstack([top,bot,row3]))
print('crop tam',xx,yy)
