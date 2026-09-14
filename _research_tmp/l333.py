import glob, os
import numpy as np, pydicom
from skimage.metrics import structural_similarity as SSIM
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
    ss=[SSIM(fn[k],qn[k],data_range=1.0,gaussian_weights=True,sigma=1.5,
             use_sample_covariance=False,win_size=11) for k in range(q.shape[0])]
    print(f"{pid}: n={q.shape[0]} HU=[{q.min():.0f},{q.max():.0f}] "
          f"PSNR(per-slice)={ps.mean():.4f} SSIM={np.mean(ss):.4f}", flush=True)
