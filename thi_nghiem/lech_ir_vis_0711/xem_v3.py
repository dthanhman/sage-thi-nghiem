import numpy as np, pandas as pd
d=pd.read_csv('kiem_v2.csv',dtype={'image_id':str}).set_index('image_id')
r=pd.read_csv('ket_qua.csv',dtype={'image_id':str})
cc=lambda a,b:np.corrcoef(a,b)[0,1]
print('[A] Hai bộ đo độc lập có đồng ý không ở d=0? (ECC vs phase, chỉ lấy phase <=12px)')
o=np.hypot(d.phase_sx,d.phase_sy)<=12
print(' n',o.sum(),'corr sx %.2f sy %.2f'%(cc(d.ecc_sx[o],d.phase_sx[o]),cc(d.ecc_sy[o],d.phase_sy[o])),'|chenh| med %.2f P90 %.2f'%(np.hypot(d.ecc_sx-d.phase_sx,d.ecc_sy-d.phase_sy)[o].median(),np.hypot(d.ecc_sx-d.phase_sx,d.ecc_sy-d.phase_sy)[o].quantile(.9)))
print('[B] Nhãn đo IR->VIS = dinh_ir - dinh_vis, so với ECC/phase')
lx,ly=d.ir_bx-d.vis_bx,d.ir_by-d.vis_by
print(' |nhan| med %.2f; corr ECC sx %.2f sy %.2f; corr phase sx %.2f sy %.2f'%(np.hypot(lx,ly).median(),cc(lx,d.ecc_sx),cc(ly,d.ecc_sy),cc(lx[o],d.phase_sx[o]),cc(ly[o],d.phase_sy[o])))
print('[C] Lệch thật của camera phải đổi từ từ theo khung liền nhau. So chênh giữa 2 khung liền kề với 2 khung ngẫu nhiên trong cùng khối')
rng=np.random.default_rng(42)
for c in ('ecc','phase'):
    a=[];b=[]
    use=d if c=='ecc' else d[o]
    for k,g in use.groupby('khoi'):
        g=g.sort_index(); ids=g.index.astype(int).values
        x=g[c+'_sx'].values;y=g[c+'_sy'].values
        for i in range(len(g)-1):
            if ids[i+1]-ids[i]==1:
                a.append(np.hypot(x[i]-x[i+1],y[i]-y[i+1]))
                j=rng.permutation(len(g))[:2]
                b.append(np.hypot(x[j[0]]-x[j[1]],y[j[0]]-y[j[1]]))
    print(' ',c,'lien ke med %.2f | ngau nhien trong khoi med %.2f (n=%d)'%(np.median(a),np.median(b),len(a)))
print('[D] Phần giảm vượt đối chứng, theo cặp từng ảnh: %SAGE - %TB')
g=lambda c,x:r[(r.cach==c)&(r.d==x)].set_index('image_id').ms_ssim_tong.reindex(d.index)
for x in (5,10):
    ps=(g('none',x)-g('none',0))/g('none',0)*100
    pt=(d[f'tb_goc_{x}']-d.tb_goc)/d.tb_goc*100
    e=ps-pt
    B=[np.median(np.concatenate([e[d.khoi==k].values for k in rng.choice(5,5)])) for _ in range(5000)]
    print(' d=%d median E %+.2f, SAGE kém hơn TB ở %.0f%% ảnh, CI95 theo khoi %s'%(x,e.median(),(e<0).mean()*100,np.percentile(B,[2.5,97.5]).round(2)))
    print('   theo khoi',e.groupby(d.khoi).median().round(2).tolist())
