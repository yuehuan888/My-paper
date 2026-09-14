"""临时核验：identity 基线的聚合方式与 HU 窗对 PSNR 的影响。只读，不落盘。"""
import glob, os, sys
import numpy as np
import pydicom

AAPM = r"D:/图像相关论文/LDCT/data/aapm"


def read_patient(pid, dose):
    d = os.path.join(AAPM, f"{dose}_3mm", pid, f"{dose}_3mm")
    ent = []
    for f in glob.glob(os.path.join(d, "*.IMA")):
        h = pydicom.dcmread(f, stop_before_pixels=True)
        ent.append((float(h.ImagePositionPatient[2]), f))
    ent.sort(key=lambda t: t[0])
    out = []
    for _, f in ent:
        ds = pydicom.dcmread(f)
        a = ds.pixel_array.astype(np.float32)
        out.append(a * float(getattr(ds, "RescaleSlope", 1.0))
                   + float(getattr(ds, "RescaleIntercept", 0.0)))
    return np.stack(out)


PIDS = ["L506", "L067"]
data = {}
for p in PIDS:
    q = read_patient(p, "quarter")
    f = read_patient(p, "full")
    print(f"{p}: quarter {q.shape} full {f.shape} "
          f"HU=[{min(q.min(), f.min()):.0f},{max(q.max(), f.max()):.0f}]", flush=True)
    data[p] = (q, f)

print()
for lo, hi in [(-1000.0, 1000.0), (-160.0, 240.0), (-1024.0, 3071.0), (-1024.0, 1185.0)]:
    R = hi - lo
    per_slice, per_pat, n = [], [], {}
    se_tot, cnt_tot = 0.0, 0
    for p in PIDS:
        q, f = data[p]
        qc = np.clip(q, lo, hi); fc = np.clip(f, lo, hi)
        se = ((qc - fc) ** 2).reshape(q.shape[0], -1).sum(1)
        c = q.shape[1] * q.shape[2]
        ps = 10 * np.log10(R * R * c / se)          # 逐切片 PSNR
        per_slice.extend(ps.tolist())
        per_pat.append((ps.mean(), q.shape[0]))      # 逐患者均值 + 权重
        n[p] = q.shape[0]
        se_tot += se.sum(); cnt_tot += c * q.shape[0]
    w = np.array([x[1] for x in per_pat], float); w /= w.sum()
    pat_mean = float(np.average([x[0] for x in per_pat], weights=w))
    pooled = float(10 * np.log10(R * R * cnt_tot / se_tot))
    print(f"窗 [{lo:.0f},{hi:.0f}] R={R:.0f}  切片数={n}  "
          f"逐切片平均={np.mean(per_slice):.4f}  逐患者加权={pat_mean:.4f}  "
          f"全局汇总MSE={pooled:.4f}  (汇总-切片平均={pooled-np.mean(per_slice):+.4f})")
