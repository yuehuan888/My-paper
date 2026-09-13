# Reconstruction-Constrained Learnable Wavelet for Low-Dose CT Denoising

> 草稿 v0.1 — 2026-09-13
> 状态：方法、实验设计、消融已定稿；最终数字待 24 组实验跑完填入
> ⚠️ 文中标 `【待填】` 的为未定数字

---

## Abstract（草稿）

Small convolutional networks are attractive for low-dose CT (LDCT) denoising in
resource-constrained clinical settings, and wavelet-domain processing is a classic
way to separate noise from structure. A natural extension is to make the wavelet
itself learnable. We show that this extension **fails by default**: an unconstrained
learnable analysis–synthesis pair drifts away from invertibility during training,
and the resulting network performs **no better than a fixed Haar basis**
(p = 【待填】, 95% CI straddling zero). We trace the failure to the transform's
reconstruction error, which reaches 99% after training, forcing the network to
compensate for a loss it should not have to model. We then show that **imposing
perfect reconstruction by construction** — parameterizing the wavelet as a lifting
scheme whose inverse shares the forward filters — makes learning pay off
(+0.29 dB over both fixed Haar and unconstrained learning, p = 0.0001).
The resulting network has **2,159 parameters** and approaches published
denoising baselines within 【待填】 dB on AAPM-Mayo 2016.

---

## 1. Introduction

### 1.1 问题
低剂量 CT 在降低辐射风险的同时引入噪声。深度去噪方法（RED-CNN 等）有效但参数量大
（百万级）。在资源受限场景（基层医院、移动设备、嵌入式）下，**轻量模型**更实用。

### 1.2 小波域去噪与"可学习小波"的诱惑
小波域把噪声与结构在频率上分开，是去噪的经典做法。自然的想法是：
**让滤波器本身可学习**，以适应特定模态/剂量的噪声特性。

### 1.3 我们的发现：默认会失败
我们构造三臂受控对照，结论出乎意料：

> **无约束地学习小波，相比固定 Haar 毫无收益**（p = 【待填】）。
> 原因不是"学习没用"，而是**学习摧毁了变换自身的可逆性**——
> 训练后分析–合成闭环的相对误差达 **99%**，网络被迫一边去噪、
> 一边补偿变换损失，这个额外负担抵消了自适应的好处。

### 1.4 我们的解法
用**提升格式（lifting scheme）** 参数化小波：逆变换不是另一组独立参数，
而是把同一组预测/更新滤波器反向减回去。**可逆性由结构保证**，
与训练过程无关。并加入参数有界化以保证 float32 下的数值稳定。

结果：**这一约束让学习真正产生价值**（+0.29 dB，p = 0.0001）。

### 1.5 贡献
1. **一个负结果**：无约束可学习小波在 LDCT 去噪上无效，并与固定 Haar 无法区分
2. **一个机制解释**：失效源于训练中可逆性的丧失（闭环误差 99%），可直接测量
3. **一个解法**：PR-LWT —— 提升格式的可学习小波，可逆性由结构保证，
   并给出 float32 数值稳定所需的有界化
4. **一个极轻量网络**：2,159 参数，接近已发表基线的表现

---

## 2. Related Work

### 2.1 低剂量 CT 去噪
RED-CNN (Chen et al., TMI 2017)、WGAN-VGG (Yang et al., TMI 2018)、
DU-GAN (Huang et al., TIM 2022)、CTformer (PMB 2023)。
【待补：各自参数量与在 AAPM-Mayo 上的报告值】

### 2.2 小波域去噪
【待补：经典小波去噪 + 近年可学习小波工作，注意已有的自适应小波方法】

### 2.3 完美重构与提升格式
Daubechies & Sweldens 的提升格式；【待补：深度学习中已有的 lifting 工作】

> ⚠️ **诚实定位**：lifting scheme 本身是小波社区 1990 年代的成熟工具，
> 深度学习领域也有人用过。**本文不宣称提出了新的小波变换。**
> 我们的贡献是把"可学习小波的可逆性"从"训练中希望它成立"变成
> "结构上必然成立"，并**证明这一约束是学习产生收益的前提**。

---

## 3. Method

### 3.1 总体结构
```
含噪 CT x
  ↓  PR-LWT 分解（2 级，7 子带）
  ↓  逐子带残差预测：delta_b = head_b(band_b)
  ↓  clean_b = band_b − delta_b
  ↓  PR-LWT 重建
干净估计 x̂
```

### 3.2 PR-LWT：提升格式的可学习小波

一维提升：
```
分解：d = x_o − P(x_e)      s = x_e + U(d)
合成：x_e = s − U(d)        x_o = d + P(x_e)     ← 与分解共用同一组 P、U
```

**可逆性由结构保证**：逆变换不是另一组参数。无论 P、U 训练成什么，
`inverse(forward(x)) = x` 恒成立（浮点精度内）。

Haar 初始化：`P = [0,1,0]`（恒等预测）、`U = [0,0.5,0]`，再施加 √2 缩放，
使初始化**严格等于正交归一 Haar**（四个子带均已核对）。

**参数有界化**（必要，非可选）：
```
P = P_init + 0.5 · tanh(θ_P)        U = U_init + 0.5 · tanh(θ_U)
```
若不加此约束，taps 在训练中可漂移到 |P| ≈ 5，此时 float32 下的闭环误差
从 3.6e-07 恶化到 5.2e+00。即 **PR 保证在数学上无条件成立，但在浮点下需要
参数有界化才能兑现**。【待填：完整训练后的实测闭环】

**参数量**：2 级 × 2 方向 × 2 滤波器 × 3 taps = **24 个**。

### 3.3 子带残差去噪头
每个子带一个轻量 CNN（`conv-3×3 → ReLU → conv-3×3`，内部 16 通道），
预测该子带上的**残差**（可正可负，无末层激活）。
7 个子带 → 7 个头，共 2,135 参数。

### 3.4 损失
`L = L1(fused, gt)`。【待填：是否加其他项】

---

## 4. Experiments

### 4.1 数据
**AAPM-Mayo 2016 Low Dose CT Grand Challenge**，3mm B30 配对子集。
10 位患者，2,378 对 512×512 切片（四分之一剂量 / 全剂量）。
公开可下（AAPM Box 直链，免登录）。

配对按 `ImagePositionPatient[2]`（z 位置）——**不能用文件名**，
实测两侧文件名的切片序号段完全一致、无法区分。

### 4.2 评测口径（必须写明，否则数字不可比）
**AAPM 官方从未定义 PSNR 口径**（官方指标是医师阅片评分）。
该数据集上至少并存 5 个互不兼容的阵营，同一模型换口径可差 **20 dB**。

本文采用 **RED-CNN / CTformer 谱系的口径**（被 6 个公开仓库逐字沿用）：
```
预处理  x = (HU + 1024) / 4096            ← 不裁剪
评测    反归一化回 HU，pred 与 gt 同时 clip 到 [-160, 240]
        PSNR = 10·log10(400² / MSE_HU)，data_range = 400
        SSIM 按 SSinyu/RED-CNN 的 measure.py
聚合    逐切片算指标后取平均
```

> 我们的实现与独立测量**逐位吻合**（L506 29.2489 / L067 26.5576 / 合并 27.8630）。

### 4.3 数据集划分
| 划分 | 患者 | 切片数 |
|---|---|---|
| 训练 | L096 L109 L143 L192 L286 L310 L333 | 1,600 |
| 验证 | L291 | 343 |
| 测试 | L506 + L067 | 435 |

**按患者划分**，避免同一患者的相邻切片跨集合。

### 4.4 基准：不处理会怎样（identity）
| 患者 | identity PSNR | SSIM |
|---|---|---|
| L506 | 29.2489 | 0.8759 |
| L067 | 26.5576 | 0.7987 |
| 合并 | 27.8630 | 0.8361 |

**这是模型必须超过的地板。** 注意患者间差异极大（L143 仅 23.99），
故必须逐患者报告。

### 4.5 主结果
【待填】

### 4.6 与已发表方法对比
【待填，用文献划分的数字】

---

## 5. Ablation：可逆性是否是学习产生收益的前提

### 5.1 三臂设计

| 臂 | 小波 | 可学习 | 可逆性保证 | 小波参数 |
|---|---|---|---|---|
| `pr` | 提升格式 | ✓ | **✓（结构保证）** | 24 |
| `unconstrained` | 分离参数 | ✓ | ✗ | 64 |
| `fixed` | 正交归一 Haar | ✗ | ✓ | 0 |

**三臂回答两个不同的问题**：
- `pr` vs `unconstrained` —— **可逆性**是否带来收益（本文的核心问题）
- `pr` vs `fixed` —— **可学习**是否带来收益

### 5.2 结果

【待填：5 种子 × 3 臂的均值±标准差，配对 t 检验（独立单元=种子）】

### 5.3 机制：可逆性的丧失

【待填：三臂训练后的闭环误差】

---

## 6. Discussion / Limitations

### 6.1 诚实边界
1. **单数据集**（AAPM-Mayo）。结论是否推广到其他 CT 数据未验证。
2. **测试集 2 个患者**，跨样本比较的独立单元是**患者**而非切片。
   435 张切片来自 2 个患者，相邻切片高度相关——本文所有显著性检验
   均以**训练种子**为独立单元，避免伪重复。
3. **未与真实基线同划分对比**；上表已发表数字引自文献，
   训练划分与本工作不同（文献 9 患者训练，本工作 8 + 1 验证）。
4. **未做下游任务验证**（如分割）。本文只声称图像层面的去噪效果。
5. **单一窗**（[-160, 240]）。若需腹部/胸部/头部各自的分窗基准，本文口径不适用。

### 6.2 为什么不声称"无损"
PR 只保证**系数未被修改**时的分析–合成闭环。去噪恰恰要修改系数，
故 PR **不保证**去噪结果无损。本文的声称严格限定为：
**PR 约束使变换不再是网络需要补偿的负担，从而使学习产生收益。**

---

## 7. Conclusion

【待填】

---

## 附录 A：复现

```bash
cd LDCT
python data/prepare_aapm.py          # DICOM -> HDF5
python experiments/train.py --baseline-only
for w in pr unconstrained fixed; do
  python experiments/train.py --tag ${w}_s0 --wavelet $w --epochs 30 --seed 0
done
python experiments/analyze_arms.py
```

每次运行的完整配置（数据量、参数构成、LR 衰减点、代码版本）在
`experiments/runs/<tag>/config.json`。
