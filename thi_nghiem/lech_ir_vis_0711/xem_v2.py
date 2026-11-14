import numpy as np, pandas as pd
d=pd.read_csv('kiem_v2.csv',dtype={'image_id':str}); n=len(d)
r=pd.read_csv('ket_qua.csv',dtype={'image_id':str})
print('n',n)
print('\n[1] Nhãn khớp nguồn nào? đỉnh quét')
for s in ('vis','ir'):
    m=np.hypot(d[s+'_bx'],d[s+'_by'])
    print(s,'dinh median dx,dy',d[s+'_bx'].median(),d[s+'_by'].median(),'|lech| med',m.median(),'P90',m.quantile(.9),'<=1px %.0f%%'%((m<=1).mean()*100),'do nhon',round(d[s+'_tp'].median(),3))
print('Tuong quan dinh IR-nhan voi ECC: sx %.2f sy %.2f; phase: sx %.2f sy %.2f'%(np.corrcoef(d.ir_bx,d.ecc_sx)[0,1],np.corrcoef(d.ir_by,d.ecc_sy)[0,1],np.corrcoef(d.ir_bx,d.phase_sx.clip(-20,20))[0,1],np.corrcoef(d.ir_by,d.phase_sy.clip(-20,20))[0,1]))
print('\n[2] Trọng tài nhãn ở d=0: cạnh IR nằm trên biên nhãn (cao hơn = khớp hơn)')
for c in ('ecc','phase'):
    a=(d['b_'+c]-d.b_goc)/d.b_goc*100; b=(d['b_'+c+'_nguoc']-d.b_goc)/d.b_goc*100
    print(c,'dich theo est: med %+.2f%%, anh tang %.0f%% | dich NGUOC: med %+.2f%%, anh tang %.0f%%'%(a.median(),(a>0).mean()*100,b.median(),(b>0).mean()*100))
    big=np.hypot(d[c+'_sx'],d[c+'_sy'])>=2
    print('   rieng |dich|>=2px (n=%d): theo est med %+.2f%%, tang %.0f%%; nguoc med %+.2f%%, tang %.0f%%'%(big.sum(),a[big].median(),(a[big]>0).mean()*100,b[big].median(),(b[big]>0).mean()*100))
    print('   theo so voi nguoc: anh est>nguoc %.0f%%'%((d['b_'+c]>d['b_'+c+'_nguoc']).mean()*100))
print('\n[3] Nội suy ở d=0, ECC (MS-SSIM tổng so với IR gốc)')
q=r[(r.cach=='ecc')&(r.d==0)].set_index('image_id').ms_ssim_tong.reindex(d.image_id).values
p=lambda v:((v-d.goc_tong)/d.goc_tong*100).median()
print('ecc le(subpx) %+.2f%% | ecc tron px %+.2f%% | phase tron %+.2f%% | ecc chuan %+.2f%%'%(p(d.ecc_le_tong),p(d.ecc_tron_tong),p(d.phase_tron_tong),p(q)))
print('\n[4] Đối chứng ghép trung bình, cùng cách chấm IR gốc')
for x in (5,10):
    print(x,'TB %+.2f%%'%(((d[f'tb_goc_{x}']-d.tb_goc)/d.tb_goc*100).median()),'| SAGE so voi IR dua vao %+.2f%%'%(((d[f'sage_vao_{x}']-d.sage_vao_0)/d.sage_vao_0*100).median()))
print('\n[5] Theo khối dev: none d=5 và ecc so voi chinh no')
d=d.set_index('image_id')
g=lambda c,x:r[(r.cach==c)&(r.d==x)].set_index('image_id').ms_ssim_tong
pn=(g('none',5)-g('none',0))/g('none',0)*100; pe=(g('ecc',5)-g('ecc',0))/g('ecc',0)*100
print(pd.DataFrame({'n':pn.groupby(d.khoi).size(),'none_d5':pn.groupby(d.khoi).median().round(2),'ecc_d5':pe.groupby(d.khoi).median().round(2),'sang':d.groupby('khoi').vis_mean.mean().round(0)}))
rng=np.random.default_rng(42); ks=sorted(d.khoi.unique()); B=[]
for _ in range(5000):
    B.append(np.median(np.concatenate([pn[d.khoi==k].values for k in rng.choice(ks,len(ks))])))
print('bootstrap theo khoi none d5 CI95',np.percentile(B,[2.5,97.5]).round(2))
