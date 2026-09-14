"""临时核验 2：各患者 HU 分布 + 逐切片平均口径下的 identity 基线换算表。只读。"""
import glob, os
import numpy as np
import pydicom
from skimage.metrics import structural_similarity as SSIM

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
print("=== 每患者 HU 分布（quarter 剂量，全部像素）===")
for p in PIDS:
    q = read_patient(p, "quarter"); f = read_patient(p, "full")
    data[p] = (q, f)
    allpx = np.concatenate([q.ravel(), f.ravel()])
    print(f"{p}: n={q.shape[0]}片  观测[{q.min():.0f},{q.max():.0f}] 真值[{f.min():.0f},{f.max():.0f}]")
    for th in [240, 400, 1000, 1185, 1500, 2000, 3000]:
        c = int((allpx > th).sum())
        print(f"    >{th:>5} HU : {c:>9} px  ({c / allpx.size * 100:8.5f}%)")
    for th in [-160, -1000, -1024]:
        c = int((allpx < th).sum())
        print(f"    <{th:>5} HU : {c:>9} px  ({c / allpx.size * 100:8.5f}%)")

print()
print("=== identity 基线（输出=输入），逐切片 PSNR/SSIM 后平均，L506+L067 共 435 片 ===")
WINS = [(-160, 240), (-300, 300), (-1000, 400), (-1000, 1000),
        (-1024, 1185), (-1024, 1500), (-1000, 2000), (-1024, 3071)]
print(f"{'窗':<20}{'R':>7}{'PSNR(逐片均)':>14}{'SSIM(逐片均)':>14}{'观测裁剪%':>11}")
base = None
for lo, hi in WINS:
    R = float(hi - lo)
    ps, ss, cl = [], [], []
    for p in PIDS:
        q, f = data[p]
        qc = np.clip(q, lo, hi); fc = np.clip(f, lo, hi)
        qn = (qc - lo) / R; fn = (fc - lo) / R
        se = ((qn - fn) ** 2).reshape(q.shape[0], -1).sum(1)
        ps.extend((10 * np.log10(q.shape[1] * q.shape[2] / se)).tolist())
        for k in range(q.shape[0]):
            ss.append(SSIM(fn[k], qn[k], data_range=1.0, gaussian_weights=True,
                           sigma=1.5, use_sample_covariance=False, win_size=11))
        cl.append(float(((q < lo) | (q > hi)).mean()))
    P, S, C = float(np.mean(ps)), float(np.mean(ss)), float(np.mean(cl))
    if (lo, hi) == (-160, 240):
        base = P
    name = "[" + str(int(lo)) + "," + str(int(hi)) + "]"
    print("%-18s%7.0f%14.4f%14.4f%10.2f%%" % (name, R, P, S, C * 100))
    if (lo, hi) in [(-160.0, 240.0), (-1000.0, 1000.0)]:
        for p in PIDS:
            q, f = data[p]
            R2 = float(hi - lo)
            qc = np.clip(q, lo, hi); fc = np.clip(f, lo, hi)
            qn = (qc - lo) / R2; fn = (fc - lo) / R2
            se = ((qn - fn) ** 2).reshape(q.shape[0], -1).sum(1)
            ps2 = 10 * np.log10(q.shape[1] * q.shape[2] / se)
            ss2 = [SSIM(fn[k], qn[k], data_range=1.0, gaussian_weights=True,
                        sigma=1.5, use_sample_covariance=False, win_size=11)
                   for k in range(q.shape[0])]
            print("      %s only (%d 片): PSNR %.4f  SSIM %.4f"
                  % (p, q.shape[0], np.mean(ps2), np.mean(ss2)))
print()
print("相对软组织窗 [-160,240] 的 PSNR 抬升（= 20*log10(R/400)）：")
for lo, hi in WINS:
    R = float(hi - lo)
    print(f"  [{lo:.0f},{hi:.0f}] 理论 {20 * np.log10(R / 400):+7.3f} dB")
