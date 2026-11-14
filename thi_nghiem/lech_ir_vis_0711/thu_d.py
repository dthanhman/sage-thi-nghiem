"""Thăm dò phương án D trên dev: SAGE có ích ở nhóm độ sáng nào, so với 2 đối chứng đơn giản."""
import cv2, numpy as np, pandas as pd, torch
import run_thu as R
dev=pd.read_csv('dev.csv',dtype={'image_id':str})
root=R.HERE.parent/'xulianh/data/raw/FMB'
net=R.Network();net.load_state_dict(torch.load('SAGE_IVIF/checkpoints/checkpoints.pt',map_location='cpu',weights_only=True));net.cuda().eval()
k=R.gauss('cuda');t=lambda a:torch.from_numpy(a).float().div(255)[None,None].cuda()
def en(f):
    h=torch.histc(f*255,256,0,255);p=h/h.sum();p=p[p>0];return float(-(p*p.log2()).sum())
rows=[]
for r in dev.itertuples():
    y=cv2.cvtColor(cv2.imread(str(root/r.vis_path)),cv2.COLOR_BGR2YCrCb)[:,:,0]
    ir=cv2.cvtColor(cv2.imread(str(root/r.ir_path)),cv2.COLOR_BGR2GRAY)
    ty,ti=t(y),t(ir)
    with torch.inference_mode(): fs=net(ty,ti)[0].clamp(0,1)
    for ten,f in (('SAGE',fs),('TB',(ty+ti)/2),('MAX',torch.max(ty,ti))):
        m=R.cham(f,ty,ti,k)
        rows.append({'image_id':r.image_id,'vis_mean':float(y.mean()),'ir_mean':float(ir.mean()),'cach':ten,'scd':m['scd'],'ms':m['ms_ssim_tong'],'en':en(f),'sd':float(f.std()*255)})
d=pd.DataFrame(rows);d.to_csv('ket_qua_d.csv',index=False)
d['nhom']=pd.cut(d.vis_mean,[0,60,90,130,256],labels=['<60','60-90','90-130','>=130'])
print('so anh',d[d.cach=='SAGE'].groupby('nhom',observed=False).size().to_dict())
for m in ('scd','ms','en','sd'):
    print('\n',m);print(d.pivot_table(index='nhom',columns='cach',values=m,aggfunc='median',observed=False)[['SAGE','TB','MAX']].round(3))
p=d.pivot_table(index='image_id',columns='cach',values='scd').join(d[d.cach=='SAGE'].set_index('image_id')[['vis_mean']])
p['khoi']=[next(k for k,(a,b) in enumerate([(52,84),(91,120),(172,206),(591,630),(1383,1419)]) if a<=int(i)<=b) for i in p.index]
p['nhom']=pd.cut(p.vis_mean,[0,60,90,130,256],labels=['<60','60-90','90-130','>=130'])
print('\nSCD: % ảnh SAGE cao hơn cả TB và MAX');print((((p.SAGE>p.TB)&(p.SAGE>p.MAX)).groupby(p.nhom,observed=False).mean()*100).round(0).to_dict())
print('Nhom theo khoi:');print(pd.crosstab(p.khoi,p.nhom))
