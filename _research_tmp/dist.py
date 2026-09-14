import glob, os
import numpy as np, pydicom
AAPM = r"D:/图像相关论文/LDCT/data/aapm"
def rd(pid, dose):
    d = os.path.join(AAPM, f"{dose}_3mm", pid, f"{dose}_3mm"); e=[]
    for f in glob.glob(os.path.join(d,"*.IMA")):
        h=pydicom.dcmread(f, stop_before_pixels=True); e.append((float(h.ImagePositionPatient[2]),f))
    e.sort(key=lambda t:t[0]); o=[]
    for _,f in e:
        ds=pydicom.dcmread(f); a=ds.pixel_array.astype(np.float32)
        o.append(a*float(getattr(ds,"RescaleSlope",1.0))+float(getattr(ds,"RescaleIntercept",0.0)))
    return np.stack(o)
for pid in ["L506","L333"]:
    q=rd(pid,"quarter"); f=rd(pid,"full")
    lo,hi=-160.0,240.0; R=400.0
    qn=(np.clip(q,lo,hi)-lo)/R; fn=(np.clip(f,lo,hi)-lo)/R
    se=((qn-fn)**2).reshape(q.shape[0],-1).sum(1)
    ps=10*np.log10(q.shape[1]*q.shape[2]/se)
    body=(q>-500).reshape(q.shape[0],-1).mean(1)   # 身体覆盖比例
    print(f"{pid} n={q.shape[0]}")
    print("  per-slice PSNR 分位: min %.2f p10 %.2f p25 %.2f 中位 %.2f p75 %.2f p90 %.2f max %.2f"
          % (ps.min(),*np.percentile(ps,[10,25,50,75,90]),ps.max()))
    print("  均值 %.4f | >35dB 的切片数 %d | body>5%% 的切片数 %d"
          % (ps.mean(), int((ps>35).sum()), int((body>0.05).sum())))
    m=body>0.05
    print("  仅 body>5%% 切片: PSNR %.4f (n=%d)" % (ps[m].mean(), int(m.sum())))
    m2=body>0.20
    print("  仅 body>20%% 切片: PSNR %.4f (n=%d)" % (ps[m2].mean(), int(m2.sum())))
