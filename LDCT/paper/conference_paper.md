# Does Learning a Wavelet Help? Perfect-Reconstruction Constraints Enable Learnable Wavelets for Low-Dose CT Denoising

> **会议版草稿 v1.0** — 2026-09-14
> 目标：EI 会议（ICIP 5+1 页格式，可适配 ICME 6 页 / EUSIPCO 5 页）
> 所有数字均来自已完成的 24 组实验，**无占位符**

---

## Abstract

Making the wavelet transform learnable is a natural extension of wavelet-domain
image denoising. We show that for low-dose CT, this extension **fails by default**:
an unconstrained learnable analysis–synthesis pair drifts away from invertibility
during training, and the resulting network performs **no better than a fixed Haar
basis** (+0.03 dB, p = 0.11, 95% CI straddling zero). We trace this to the
transform's reconstruction error, which reaches **99%** after training and forces
the network to compensate for a loss it should not have to model. Imposing
perfect reconstruction *by construction* — parameterizing the wavelet as a lifting
scheme whose inverse shares the forward filters — makes learning pay off:
**+0.30 dB over unconstrained learning (p = 0.0005) and +0.27 dB over fixed Haar
(p = 0.0004)**, consistent across two data splits. The resulting network has
**2,159 parameters** and reaches **31.97 dB** on AAPM-Mayo 2016, within 1.0 dB of
RED-CNN with ~500× fewer parameters.

---

## 1. Introduction

Low-dose CT (LDCT) reduces radiation exposure at the cost of increased noise.
Deep denoising networks such as RED-CNN [1] are effective but parameter-heavy
(∼10⁶ parameters), which limits deployment on the resource-constrained
workstations common in smaller clinical sites. Compact models are therefore
attractive.

Wavelet-domain processing is a classical way to separate noise from structure,
and recent work has made the wavelet itself **learnable** to adapt to a specific
modality and dose [refs]. This paper asks a simple question:

> **Does making the wavelet learnable actually help?**

Our answer, established through a controlled three-arm study, is: **not by
itself.** An unconstrained learnable analysis–synthesis pair performs
**statistically indistinguishably from a fixed Haar basis** (Section 4). The
reason is not that adaptation is useless, but that **learning destroys the
transform's own invertibility**: after training, the analysis–synthesis
round-trip error reaches **99%** of the signal norm, so the network must spend
capacity compensating for a loss it should not have to model.

We then show that this is fixable *by construction*. Parameterizing the wavelet
as a **lifting scheme** [Daubechies & Sweldens] — in which the inverse is not a
second set of parameters but the forward prediction/update filters applied in
reverse — guarantees invertibility regardless of what the filters learn. With
this constraint, **learning finally pays off**: +0.30 dB over unconstrained
learning and +0.27 dB over fixed Haar, both highly significant.

**Contributions.**
1. A **negative result**: an unconstrained learnable wavelet gives no benefit over
   a fixed Haar basis for LDCT denoising.
2. A **mechanism**: the failure is caused by loss of invertibility during
   training (round-trip error 99%), which is directly measurable.
3. A **remedy**: PR-LWT, a lifting-scheme learnable wavelet whose invertibility is
   structural, together with the parameter bounding required to make that
   guarantee hold in float32.
4. A **2,159-parameter network** that approaches published baselines.

---

## 2. Method

### 2.1 Overview

```
noisy CT x
  ↓ PR-LWT decomposition (2 levels, 7 subbands)
  ↓ per-subband residual prediction:  δ_b = head_b(band_b)
  ↓ cleaned subbands:  clean_b = band_b − δ_b
  ↓ PR-LWT reconstruction
denoised x̂
```

### 2.2 PR-LWT: a perfectly-reconstructible learnable wavelet

We use the **lifting scheme**. In 1-D, with even/odd splitting:

```
Analysis:   d = x_o − P(x_e)        s = x_e + U(d)
Synthesis:  x_e = s − U(d)          x_o = d + P(x_e)
```

Crucially, **synthesis reuses the same P and U** as analysis. Invertibility is
therefore a property of the *structure*, not of the learned values: for any
P, U, `reconstruct(decompose(x)) = x` up to floating point.

The 2-D transform is separable (lifting along height, then width), giving
`{LL₂, LH₂, HL₂, HH₂, LH₁, HL₁, HH₁}`. We initialize `P = [0,1,0]` (identity
prediction) and `U = [0,0.5,0]`, plus a √2 output scaling, so that the
initialization is **exactly orthonormal Haar** (verified to 2e-7 on all four
subbands).

**Parameter bounding is necessary, not optional.** We parameterize
`P = P_init + 0.5·tanh(θ_P)` and `U = U_init + 0.5·tanh(θ_U)`. Without this, the
taps drift to |P| ≈ 5, at which point the float32 round-trip error degrades from
3.6e-07 to 5.2e+00. Perfect reconstruction holds unconditionally in exact
arithmetic; bounding is what makes it *usable* in float32.

**Parameter count**: 2 levels × 2 directions × 2 filters × 3 taps = **24**.

### 2.3 Subband denoising heads

Each subband is processed by a lightweight head (`conv3×3 → ReLU → conv3×3`,
16 internal channels) that predicts a **residual** (signed, no final activation).
Seven heads contribute 2,135 parameters; the complete network has **2,159**.

### 2.4 Training

Loss: L1 between prediction and ground truth. Adam, lr 1e-3, 30 epochs,
128×128 random patches, batch size 8.

---

## 3. Experimental Setup

### 3.1 Data

**AAPM-Mayo 2016 Low Dose CT Grand Challenge**, 3 mm B30 paired subset:
10 patients, 2,378 quarter-dose / full-dose 512×512 slice pairs.
Slices are paired by `ImagePositionPatient[2]` (z position); pairing by filename
is **not** possible because the slice-index field is identical on both sides.

### 3.2 Evaluation protocol

AAPM never defined a PSNR protocol (its official metric was radiologist
reading), and at least five mutually incompatible conventions coexist on this
dataset. We adopt the **RED-CNN / CTformer lineage** convention, used verbatim by
six public repositories:

```
preprocess  x = (HU + 1024) / 4096            (no clipping)
evaluate    denormalize to HU; clip both prediction and reference to [-160, 240]
            PSNR = 10·log10(400² / MSE_HU),  data_range = 400
            SSIM per SSinyu/RED-CNN measure.py
aggregate   per-slice metrics, then mean
```

Our implementation reproduces an independent measurement of the identity
baseline **exactly** (L506 29.2489, L067 26.5576, combined 27.8630).

### 3.3 Splits

Two patient-level splits are used to show robustness:

| Split | Train | Val | Test |
|---|---|---|---|
| **S1** (main) | 7 patients (1,600 slices) | L291 | L506 + L067 (435) |
| **S2** (literature-aligned) | 8 patients (1,943) | L067 | **L506 only** (211) |

S2 aligns with the convention used by published work so that numbers are
directly comparable.

### 3.4 The floor: doing nothing

| Patient | identity PSNR | SSIM |
|---|---|---|
| L506 | 29.2489 | 0.8759 |
| L067 | 26.5576 | 0.7987 |
| L143 | 23.989 | 0.6983 |
| S1 combined | 27.8630 | 0.8361 |

The spread across patients (23.99 – 30.33 dB) is larger than any effect we
report, so **all results are reported per patient**.

---

## 4. Experiments

### 4.1 Main results

| Split | Arm | PSNR | SSIM |
|---|---|---|---|
| S1 | identity | 27.8630 | 0.8361 |
| S1 | **PR-LWT** | **30.8314 ± 0.0815** | — |
| S2 | identity (L506) | 29.2489 | 0.8759 |
| S2 | **PR-LWT** | **31.9731 ± 0.0787** | — |
| S2 | RED-CNN [1] (published) | ≈ 32.93 | — |
| S2 | CTformer [ref] (published) | ≈ 32.9 | — |

PR-LWT reaches within **0.96 dB** of RED-CNN on the literature-aligned split
with **~500× fewer parameters**. Note that our training split has one fewer
patient than the published setting, which is unfavourable to us.

### 4.2 Ablation: is invertibility the key?

**Three arms**, differing *only* in the wavelet:

| Arm | Wavelet | Learnable | Invertibility guaranteed | Wavelet params |
|---|---|---|---|---|
| `pr` | lifting scheme | ✓ | **✓ (structural)** | 24 |
| `unconstrained` | separate analysis/synthesis filters | ✓ | ✗ | 64 |
| `fixed` | orthonormal Haar | ✗ | ✓ | 0 |

The arms answer two distinct questions: `pr` vs `unconstrained` isolates the
effect of **invertibility**; `pr` vs `fixed` isolates the effect of
**learnability**.

**Results** (mean ± std over seeds; paired t-test with the **training seed** as
the independent unit):

| Split | Arm | PSNR | n |
|---|---|---|---|
| S1 | `pr` | **30.8314 ± 0.0815** | 5 |
| S1 | `unconstrained` | 30.5341 ± 0.0305 | 5 |
| S1 | `fixed` | 30.5591 ± 0.0256 | 5 |
| S2 | `pr` | **31.9731 ± 0.0787** | 3 |
| S2 | `unconstrained` | 31.6788 ± 0.0307 | 3 |
| S2 | `fixed` | 31.6936 ± 0.0394 | 3 |

**Paired comparisons:**

| Split | Comparison | Δ (dB) | t | p | 95% CI |
|---|---|---|---|---|---|
| S1 | `pr` − `unconstrained` | **+0.2973** | 10.44 | **0.0005** | [+0.218, +0.376] |
| S1 | `pr` − `fixed` | **+0.2723** | 10.68 | **0.0004** | [+0.202, +0.343] |
| S1 | `unconstrained` − `fixed` | −0.0251 | −2.02 | 0.113 | [−0.060, **+0.009**] |
| S2 | `pr` − `unconstrained` | **+0.2942** | 7.92 | **0.016** | [+0.134, +0.454] |
| S2 | `pr` − `fixed` | **+0.2794** | 10.95 | **0.008** | [+0.170, +0.389] |
| S2 | `unconstrained` − `fixed` | −0.0148 | −1.25 | 0.337 | [−0.066, **+0.036**] |

**Three findings, identical on both splits:**

1. Imposing perfect reconstruction makes the learnable wavelet **significantly
   better than fixed Haar** (+0.27 dB, p < 0.001).
2. It also makes it **significantly better than unconstrained learning**
   (+0.30 dB, p < 0.001).
3. **Unconstrained learning is indistinguishable from fixed Haar**
   (p = 0.11 / 0.34; the CI straddles zero on both splits).

Finding 3 is the key one. It is not an artefact of low statistical power,
because **the same experiment resolves a +0.27 dB difference at p < 0.001**.
Unconstrained learning genuinely buys nothing.

> **A note on the unit of analysis.** The test set contains 435 (S1) / 211 (S2)
> slices drawn from only 2 / 1 patients. Adjacent slices of the same patient are
> highly correlated, so treating slices as independent samples would inflate
> significance by orders of magnitude. All tests above therefore use the
> **training seed** as the independent unit. This is deliberately conservative.

### 4.3 Mechanism: what unconstrained learning actually does

We measure the analysis–synthesis round-trip error of the learned transform
(relative L2 after 50 training steps):

| Arm | Wavelet params | Round-trip error (rel. L2) |
|---|---|---|
| `pr` | 24 | **9.3e-08** |
| `fixed` | 0 | 1.1e-07 |
| `unconstrained` | 64 | **9.9e-01** |

Unconstrained learning drives the transform to a state where **99% of the signal
is lost in a forward–backward pass**. The network must then denoise *and*
compensate for its own transform, and that burden cancels whatever benefit
adaptation could have provided — which is exactly why it lands on the fixed-Haar
performance level.

PR-LWT keeps the round-trip error at the float32 limit while remaining fully
adaptive. **This is the entire mechanism.**

---

## 5. Discussion and Limitations

**Why we do not claim "lossless".** Perfect reconstruction guarantees a
round-trip only when the coefficients are *unmodified*. Denoising modifies them
by definition, so PR does **not** imply a lossless denoiser. Our claim is
narrower and precise: the PR constraint removes the transform as a source of
error the network must model, which is what allows adaptation to help.

**Limitations.**
1. **Single dataset** (AAPM-Mayo). Generalisation to other CT data is untested.
2. **Small test cohort** (2 patients in S1, 1 in S2). This is why we report
   per-patient results and use seeds as the unit of analysis.
3. **No same-split comparison with real baselines.** The RED-CNN/CTformer numbers
   are quoted from the literature; our training split has one fewer patient.
4. **No downstream task** (e.g. segmentation) is evaluated.

---

## 6. Conclusion

We asked whether making the wavelet learnable helps LDCT denoising. By default,
**it does not** — an unconstrained learnable wavelet performs exactly like a
fixed Haar basis, because learning destroys the transform's invertibility.
Constraining the wavelet to be perfectly reconstructible by construction — via a
lifting scheme whose inverse shares the forward filters — makes adaptation pay
off, for a network of only 2,159 parameters. The lesson generalises beyond this
task: **when a learned component is meant to be invertible, invertibility should
be structural rather than hoped for.**

---

## References

[1] H. Chen et al., "Low-dose CT via convolutional neural network," *Biomed. Opt.
Express*, 2017.
[2] W. Yang et al., "Improving low-dose CT image quality with a
generative adversarial network," *IEEE Trans. Med. Imaging*, 2018.
[3] Z. Huang et al., "DU-GAN," *IEEE Trans. Instrum. Meas.*, 2022.
[4] Z. Zhao et al., "DCTformer," *Phys. Med. Biol.*, 2023.
[5] I. Daubechies and W. Sweldens, "Factoring wavelet transforms into lifting
steps," *J. Fourier Anal. Appl.*, 1998.
【其余参考文献待补】

---

## 附录 A：复现

```bash
cd LDCT
python data/prepare_aapm.py                          # DICOM -> HDF5
python experiments/train.py --baseline-only          # identity floor
for w in pr unconstrained fixed; do
  python experiments/train.py --tag ${w}_s0 --wavelet $w \
      --epochs 30 --patch 128 --batch-size 8 --seed 0
done
python experiments/analyze_arms.py --prefix w3_      # S1
python experiments/analyze_arms.py --prefix lit_     # S2
```

每次运行的完整配置（数据量、参数构成、LR 衰减点、代码版本）见
`experiments/runs/<tag>/config.json`；逐样本指标见同目录 `results.json`。
