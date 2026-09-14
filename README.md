# 可逆小波域低剂量 CT 去噪（PR-LWT）

**让可学习小波的可逆性由结构保证，而不是寄希望于它自己成立。**

> 📄 论文初稿已完稿，目标 EI 会议（ICIP / EUSIPCO / ICME），投稿准备中。
> 全部数字来自已完成的实验，无占位符。

---

## 一句话结论

在小波域去噪里让变换本身可学习，**默认是无效的**——无约束的可学习分析/合成滤波器对
在训练中会丢掉可逆性，最终表现与固定 Haar 基**无法区分**（p = 0.11）。
把可逆性**由结构保证**（提升格式，逆变换复用正向滤波器）之后，学习才产生收益（+0.27 dB，p = 0.0004）。

**而且这是因果，不只是相关**：向本来可逆的 PR 臂**人为注入**一个受控的闭环误差，
它会单调退化到固定 Haar 水平（n = 5 种子，每档 p ≤ 0.007）。

---

## 核心实验

### 三臂对照（S1 划分，5 种子；S2 划分，3 种子）

| 臂 | 小波 | 可学习 | 可逆性保证 | 小波参数 | S1 PSNR |
|---|---|---|---|---|---|
| `pr` | 提升格式 | ✓ | **✓（结构保证）** | 24 | **30.8314 ± 0.0815** |
| `unconstrained` | 分离正/逆滤波器 | ✓ | ✗ | 64 | 30.5341 ± 0.0305 |
| `fixed` | 正交归一 Haar | ✗ | ✓ | 0 | 30.5591 ± 0.0256 |

关键配对检验（独立单元 = **训练种子**，非测试切片）：

| 比较 | Δ (dB) | p |
|---|---|---|
| `pr` − `unconstrained` | **+0.2973** | **0.0005** |
| `pr` − `fixed` | **+0.2723** | **0.0004** |
| `unconstrained` − `fixed` | −0.0251 | 0.113（**无法区分**） |

### 干预实验（因果）

让 **PR 臂**的合成步使用 `(1−ε)·U` 而分解仍用 `U`，隔离复现无约束臂的缺陷：

| 注入 ε | 实测闭环 | 测试 PSNR | 配对 p |
|---|---|---|---|
| 0（对照） | 1.6e-07 | 30.8314 | — |
| 0.111 | 5.5% | 30.7142 | 0.0067 |
| 0.222 | 8.7% | **30.5500** | 0.0010 |
| 0.444 | 11.8% | 30.2850 | <0.0001 |

**8.7% 误差处落到固定 Haar 水平（30.5591）** —— 一个"什么都没学到"的臂的表现。

### 同划分基线

| 划分 | 方法 | 参数量 | PSNR |
|---|---|---|---|
| S1 | PR-LWT（本文） | **2,159** | 30.8314 |
| S1 | RED-CNN（**我们同协议复现**） | 1,848,865 | 31.8310 |
| S2 | PR-LWT（本文） | **2,159** | 31.9731 |
| S2 | RED-CNN（**我们同协议复现**） | 1,848,865 | **32.9471** |
| S2 | RED-CNN（文献报告值） | ~10⁶ | ≈ 32.93 |

> 我们在 L506 上复现的 RED-CNN 是 **32.9471**，与文献的 32.93 相差 **0.017 dB** ——
> 这独立验证了实现与评测口径。**PR-LWT 在 PSNR 上并未击败 RED-CNN**（差约 1.0 dB），
> 但参数量少 **857 倍**。本文的主张是*什么让可学习变换产生收益*，不是刷 SOTA。

---

## 仓库结构

```
.
├── LDCT/                              # ⭐ 当前项目
│   ├── models/
│   │   ├── lifting_dwt.py             # ⭐ PR-LWT：提升格式，正逆共享 P/U
│   │   ├── denoiser.py                # 子带残差去噪网络（2,159 参数）
│   │   ├── legacy_dwt.py              # 旧式分离参数小波（unconstrained 对照臂）
│   │   └── redcnn.py                  # RED-CNN 基线
│   ├── utils/metrics.py               # 口径 A：RED-CNN/CTformer 谱系
│   ├── data/
│   │   ├── dataset.py, prepare_aapm.py
│   │   └── download.sh                # 数据集获取（不入库，见 .gitignore）
│   ├── experiments/
│   │   ├── train.py                   # ⭐ 训练/评测主入口（逐轮落盘 + --resume）
│   │   ├── analyze_arms.py            # 三臂统计（配对 t 检验 + CI）
│   │   ├── analyze_intervention.py    # ⭐ 干预实验的剂量-响应
│   │   ├── calibrate_mismatch.py      # 标定注入强度 ε
│   │   ├── verify_roundtrip.py        # 闭环误差测量（含从 checkpoint 复测）
│   │   ├── identity_floors.py         # 逐患者 identity 地板
│   │   ├── make_figures.py            # 论文图表
│   │   ├── run_redcnn_baselines.ps1   # 同划分 RED-CNN 基线
│   │   ├── run_intervention.ps1       # 干预实验（3 档 ε × 5 种子）
│   │   └── runs/<tag>/                # 每次运行的 config / results / DONE
│   ├── tests/
│   │   ├── test_lifting_pr.py         # PR 完美重构验证（含奇偶尺寸、AMP）
│   │   └── test_eval_batch_equiv.py   # 验证评测批大小不改变数值
│   └── paper/
│       ├── conference_paper.md        # ⭐ 英文稿（投稿用）
│       ├── conference_paper_zh.md     # 中文对照版（仅供审阅）
│       └── latex/main.tex             # IEEEtran 版（tectonic main.tex 编译）
│
├── EI会议调研_2026-09-14.md             # 投哪个会、截稿、EI 依据
├── LDCT投稿路线_2026-09-13.md           # 期刊路线（备用）
├── LDCT_三臂对照结果_2026-09-13.md       # ⚠️ 其中「99%」已过时，见下文
├── LDCT_HU口径核查_2026-09-13.md
├── 替代方向调研_2026-09-13.md
├── PR-LWT决断报告_2026-09-12.md
└── LICENSE
```

> ⚠️ **历史文档的已知错误**：`LDCT_三臂对照结果_2026-09-13.md` 与 `LDCT投稿路线_2026-09-13.md`
> 中写的「闭环误差 **99%**」**是错的**，实测为 **12.5–14.5%**；`LDCT投稿路线` 把 **WE-UNet**
> 列为 LDCT 同题材论文也不对（它是**胸片**辐射剂量论文）。论文正文已更正，历史文档未回改。

---

## 复现

```bash
cd LDCT
PY=D:/DeveloperTools/miniconda/envs/mgfnet/python.exe

# 数据（需先下载 AAPM-Mayo，见 data/download.sh 与 data/README）
$PY data/prepare_aapm.py

# identity 地板
$PY experiments/train.py --baseline-only

# 三臂对照
for w in pr unconstrained fixed; do
  $PY experiments/train.py --tag w3_${w}_s0 --wavelet $w \
     --epochs 30 --patch 128 --batch-size 8 --seed 0 --val-interval 5
done
$PY experiments/analyze_arms.py --prefix w3_        # S1
$PY experiments/analyze_arms.py --prefix lit_       # S2

# 干预实验（因果）
$PY experiments/calibrate_mismatch.py               # 标定 ε
powershell -File experiments/run_intervention.ps1    # 3 档 × 5 种子
$PY experiments/analyze_intervention.py

# RED-CNN 同划分基线
powershell -File experiments/run_redcnn_baselines.ps1

# 图表与论文
$PY experiments/make_figures.py
cd paper/latex && tectonic main.tex                  # 需装 tectonic
```

**判断"跑完了没"看 `experiments/runs/<tag>/DONE`；看进度看同目录 `progress.json`。**
任何一次运行被打断，用 `--resume` 接着跑，不必从零开始。

---

## 环境与注意事项

- Python 3.10 / PyTorch 2.5.1+cu121 / CUDA；实测 **RTX 3050 Laptop 4GB** 可跑通
- **评测批大小必须为 1**（大模型）：cuDNN 对 `ConvTranspose2d` 在 512² 且 batch ≥ 2 时
  会选一个 workspace 约 7 GB 的算法，4GB 卡装不下 → 溢出到 WDDM 共享内存 → 耗时跳 49 倍。
  已用 `tests/test_eval_batch_equiv.py` 验证批大小**不改变数值**。
- 本机的 `.ps1` 脚本**必须纯 ASCII**：PowerShell 5.1 读无 BOM 的 .ps1 会按 GBK 解码，
  中文全角标点会吃掉引号导致脚本静默失效。

---

## 诚实边界

1. **单一数据集**（AAPM-Mayo）。跨库泛化未验证 —— LoDoPaB-CT 我们考察后放弃：
   它是**重建**基准（observation 是 sinogram），不先做 FBP 无法用作零样本去噪测试。
2. **测试队列小**（S1 两位患者，S2 一位）。故全部逐患者报告、以种子为分析单元。
3. **同划分基线只覆盖 RED-CNN**；CTformer 等仍引自文献。
4. **注入的缺陷比真实的更简单**：只扰动合成侧，而无约束臂是分析/合成两组滤波器
   各自漂移。干预确立的是**同量级下的充分性**，不是精确复现。
5. **未做下游任务验证**（如分割）。

---

## 定位说明

提升格式（lifting scheme）本身是小波社区 1990 年代的成熟工具，
**LINN（EUSIPCO 2021）与 WINNet（IEEE TIP 2022）已经**用提升格式构建可逆去噪网络，
且 WINNet 已在保持完美重构的前提下学习滤波器 taps。

**本文不宣称提出了新的小波或新的架构。** 本文的贡献是那个**缺失的对照**：
先前的可逆小波去噪方法*假设*了结构性可逆、从未检验其缺失，
因此观察不到"没有它会怎样"。让可逆性变成**可选项**、固定其余一切并测量后果，
是本文做的事。

---

## 第三方材料

本仓库**不包含**：数据集（体积大，公开可下）、他人论文 PDF、第三方方法源码。
调研文档只保留书目信息（作者、标题、会议/期刊、DOI）。

## 许可

代码采用 MIT 许可，见 [`LICENSE`](LICENSE)。
