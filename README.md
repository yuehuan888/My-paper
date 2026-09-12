# MGF-Net

**多尺度门控频域融合网络（Multi-scale Gated Frequency Fusion Network）** —— 面向医学图像融合（CT-MRI），主打**轻量化**与**频域自适应**。

> 🚧 **项目状态：开发中（Phase A）**
> 当前代码为 v2 基线实现。正在按 [`发表路线与实验计划_2026-09-12.md`](发表路线与实验计划_2026-09-12.md) 重构评价协议与核心模块，
> **仓库中暂不包含可引用的实验结论**。论文中报告的数字请以下一版 README 为准。

---

## 方法概述

MGF-Net 在**小波频域**中完成跨模态融合，而非在像素域直接融合。整体流程：

```
CT ──┐
     ├─► 多级频域分解 ──► 逐子带门控融合 ──► 跨频段协调 ──► 逆变换 ──► 边缘精炼 ──► 融合图
MRI ─┘   (7 子带)        (CMGF)             (CrossBand)              (EAR)
```

| 模块 | 说明 | 位置 |
|---|---|---|
| **多级频域分解** | 2 级小波分解，得到 7 个子带 `{LL2, LH2, HL2, HH2, LH1, HL1, HH1}`，滤波器参数可端到端学习 | `models/dwt_layer.py` |
| **跨模态门控融合 (CMGF)** | 每个子带独立的轻量融合块，以 CT 为基底、学习注入 MRI 细节的权重（残差式，非凸组合） | `models/gated_fusion.py` |
| **跨频段协调** | 轻量 SE 式注意力，建模子带间的相关性 | `models/gated_fusion.py` |
| **边缘感知精炼 (EAR)** | 由两源图与初始融合图提取梯度图，引导残差精炼，增强骨骼与组织边界 | `models/edge_refine.py` |

**设计动机**：医学图像具有明确的频域分工——CT 骨骼边缘以高频为主、MRI 软组织纹理偏中频、功能影像（PET/SPECT）信息集中在低频。在频域中按子带分别决策融合策略，比在像素域做全局平均/取最大更符合数据特性。

---

## 快速开始

### 环境

```bash
pip install -r MGF-Net/requirements.txt
```

开发环境：Python 3.10 / PyTorch 2.6 + CUDA 12.4。
本项目的设计目标之一是**可在消费级 GPU 上训练**（实测 RTX 3050 Laptop 4GB 可跑通）。

### 数据准备

见 [`MGF-Net/data/README.md`](MGF-Net/data/README.md)。数据集不随仓库分发。

### 训练

```bash
cd MGF-Net
python train.py
```

### 测试

```bash
cd MGF-Net
python test.py
```

配置项集中在 `MGF-Net/config.py`。

---

## 仓库结构

```
.
├── MGF-Net/                          # 代码
│   ├── models/
│   │   ├── mgf_net.py                # 主模型（wavelet / gate_type 双开关）
│   │   ├── lifting_dwt.py            # ⭐ PR-LWT：可逆性由结构保证的可学习小波
│   │   ├── dwt_layer.py              # 旧的分离式小波（保留作对照）
│   │   ├── gated_fusion.py           # 跨模态门控融合 + 跨频段注意力
│   │   └── edge_refine.py            # 边缘感知精炼
│   ├── utils/
│   │   └── metrics.py                # ⭐ 10 项标准融合指标
│   ├── tests/
│   │   ├── test_metrics_invariants.py  # 指标 L0 不变量测试
│   │   └── test_lifting_pr.py          # PR-LWT 完美重构验证
│   ├── experiments/
│   │   ├── run_experiment.py         # ⭐ 单实验运行器（固定种子、验证集选模）
│   │   ├── aggregate.py              # 跨种子汇总
│   │   └── inspect_gate.py           # 门控权重分布检查
│   ├── data/
│   │   ├── dataset.py                # 数据加载（显式 ID / 路径对）
│   │   ├── build_manifest.py         # ⭐ 清单与病例级划分构建
│   │   ├── manifest_ct_mri.csv       # 184 对逐图清单
│   │   └── README.md                 # 数据集获取说明
│   ├── splits/
│   │   ├── ct_mri_case_v1.json       # ⭐ 病例级划分 + 切片级随机对照
│   │   └── ct_mri_screen_v1.json     # 早期筛查划分（已过时）
│   ├── evaluate.py                   # 批量评价 CLI
│   ├── diagnose_step1.py             # 审计脚本
│   ├── losses.py / config.py
│   ├── train.py / test.py            # 旧入口（已被 experiments/ 取代）
│   ├── compare_baselines.py / generate_figures.py
│   ├── requirements.txt
│   └── 工作进展报告_MGF-Net.md        # 历史文档（数字与代码不符，见实验记录 §9.2）
│
├── MGF-Net_实验记录_2026-09-12.md     # ⭐⭐ 本轮完整记录，从这里开始读
├── MGF-Net_审计报告_Step1_2026-09-12.md
├── MGF-Net_实验报告_门控对照_2026-09-12.md
├── 对比协议调研_2026-09-12.md         # 领域对比协议（107-agent 调研）
├── 发表路线与实验计划_2026-09-12.md    # 路线图（已被修订版替代）
├── 发表路线与实验计划_2026-09-12_修订版.md
├── 前沿调研报告_图像融合2024-2026.md
├── 前沿调研报告_图像融合2024-2026_复核补充版.md
├── 图像融合顶刊论文及源码汇总_2022-2026.md
├── 图像融合汇总表_勘误与补全_2026-09-12.md
│
├── LICENSE
└── README.md
```

> 📄 根目录的调研文档只包含**书目信息与分析**（作者、标题、会议/期刊、DOI、方法评述），
> **不包含任何论文 PDF 原文**。

---

## 路线图

完整计划见 [`发表路线与实验计划_2026-09-12.md`](发表路线与实验计划_2026-09-12.md)。

| 阶段 | 内容 | 状态 |
|---|---|---|
| Phase A | 标准评价协议、可追溯的数据清单与病例级划分 | ✅ 完成 |
| Phase B | 核心模块重构（PR-LWT）+ 门控修正 + 消融实验 | 🚧 进行中 |
| Phase C | 真实基线对比、论文撰写与投稿 | ⏳ 待开始 |

### 已有结果（详见 `MGF-Net_实验记录_2026-09-12.md`）

- **门控饱和**：原式 `tanh(g)·0.4` 使 99.13% 的像素贴在 +0.4 上限，门控退化为常数。
  改为中性 sigmoid 门控后权重铺满 [0.007, 0.977]，10 项指标赢 9 项。
- **PR-LWT**：变换闭环相对误差由 `6.09e-01` 降至 `1.40e-07`，参数量还少 40 个。
  并发现"有 PR 保证 ≠ 数值稳定"，需对提升滤波器参数做有界化。
- **数据泄漏**：ASFE-Fusion 数据集的 ID 含病例结构（184 张 = 10 病例 × 16–21 层）。
  复现领域惯例的切片级随机划分后，10 个病例中 9 个跨训练/测试集。

> ⚠️ 以上均为**机制层**验证。**融合质量是否改善尚未验证**——
> PR 只保证"系数未被修改"时的闭环，而融合恰恰要修改系数。

---

## 关于第三方材料

本仓库**不包含**以下内容，请自行获取：

- **数据集** —— 见 `MGF-Net/data/README.md`
- **对比方法的代码与预训练权重** —— 请访问 U2Fusion、SwinFusion、CDDFuse、SeAFusion、EMFusion 等方法的官方仓库
- **参考文献原文** —— 调研文档中仅保留书目信息（作者、标题、会议/期刊、DOI），不含 PDF 文件

---

## 引用

论文发表后补充。

---

## 致谢

本项目的对比实验基于以下开源工作，一并致谢：U2Fusion、EMFusion、SwinFusion、CDDFuse、SeAFusion。
文献调研部分受益于 [Awesome-Image-Fusion](https://github.com/GeoVectorMatrix/Awesome-Image-Fusion) 等汇总项目。

## 许可

代码部分采用 MIT 许可，见 [`LICENSE`](LICENSE)。
`docs/` 下的调研文档采用 CC BY 4.0。
