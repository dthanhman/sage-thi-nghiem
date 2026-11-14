"""Phép căn ở d=0: IR gốc bị dịch bao nhiêu, và điểm với từng nguồn đổi ra sao.
Cũng đo độ ổn định: est(d) - est(0) có khớp lượng dịch thật không."""
import sys, numpy as np, pandas as pd
df=pd.read_csv(sys.argv[1] if len(sys.argv)>1 else 'ket_qua.csv',dtype={'image_id':str})
g=df[(df.cach=='none')&(df.d==0)].set_index('image_id')
for c in ('phase','ecc'):
    s=df[(df.cach==c)&(df.d==0)].set_index('image_id')
    lech=np.hypot(s.est_sx,s.est_sy)
    print(f'[{c} d=0] |dich| median {lech.median():.2f} P90 {lech.quantile(.9):.2f}; dx med {s.est_sx.median():.2f} dy med {s.est_sy.median():.2f}')
    for m in ('ms_ssim_vis','ms_ssim_ir','ms_ssim_tong','scd'):
        d=(s[m]-g[m].reindex(s.index))
        print(f'   {m:13} median delta {d.median():+.4f} (%{(d/g[m].reindex(s.index)).median()*100:+.2f}), ảnh tăng {(d>0).mean()*100:.0f}%')
    # chia theo độ lớn phép dịch
    b=pd.cut(lech,[-.01,1,3,6,1e4],labels=['<=1','1-3','3-6','>6'])
    t=(s.ms_ssim_tong-g.ms_ssim_tong.reindex(s.index))/g.ms_ssim_tong.reindex(s.index)*100
    print(pd.DataFrame({'n':t.groupby(b,observed=False).size(),'med%':t.groupby(b,observed=False).median().round(2)}).T.to_string())
print('\n--- ổn định: |[est(d)-est(0)] - dịch thật| ---')
for c in ('phase','ecc'):
    z=df[(df.cach==c)&(df.d==0)].set_index('image_id')
    for d in (2,5,10):
        s=df[(df.cach==c)&(df.d==d)].set_index('image_id')
        e=np.hypot(s.est_sx-z.est_sx.reindex(s.index)-s.true_sx, s.est_sy-z.est_sy.reindex(s.index)-s.true_sy)
        print(f'{c:5} d={d:2} median {e.median():.2f} P90 {e.quantile(.9):.2f} >2px {(e>2).mean()*100:.1f}%')
# IR sau căn so với IR gốc nếu căn hoàn hảo: điểm cao nhất có thể ở d>0 là bao nhiêu?
for c in ('phase','ecc'):
    o=df[(df.cach==c)&(df.d==0)].ms_ssim_tong.median(); 
    print(c,'median tong d0',round(o,4),'d5',round(df[(df.cach==c)&(df.d==5)].ms_ssim_tong.median(),4),'d10',round(df[(df.cach==c)&(df.d==10)].ms_ssim_tong.median(),4))
