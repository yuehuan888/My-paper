# LDCT-Denoise：低剂量 CT 去噪

> **立项日期**：2026-09-13
> **前身**：`MGF-Net`（医学图像融合）—— 见 `../MGF-Net_项目状态_2026-09-13.md`
> **决策依据**：`../替代方向调研_2026-09-13.md`、`../MGF-Net_γ扫描与指标退化_2026-09-13.md`

---

## 一、为什么换到这个方向

原方向（CT-MRI 图像融合）遇到三个**结构性**问题，不是调参能解决的：

| 问题 | 实测证据 |
|---|---|
| **指标退化** | MI 的最优解是 α=0.10（几乎照抄 MRI）、VIF 的最优解是纯 MRI。**它们不衡量融合能力** |
| **无唯一真值** | 融合没有 ground truth，只能与源图比；零参数像素平均在 MI/CC/PSNR/VIF 上击败训练模型 |
| **数据太小** | Harvard/AANLIB 仅 10 个病例、184 对图；测试最多 2 个病例，统计功效近零 |

**低剂量 CT 去噪从根上避开这三点**：

| 维度 | 融合 | LDCT 去噪 |
|---|---|---|
| 真值 | ❌ 无 | ✅ **同患者实测的全剂量重建（数值张量）** |
| 指标 | 源参考型，可被"照抄单源"刷高 | PSNR / SSIM，**任何人可复算** |
| 数据 | 10 例 | ~800 患者 / 约 4 万切片 |
| 划分 | ❌ 领域无标准，无公开清单 | ✅ 官方 train/val/test 划分 + 患者 ID 清单 |

**技术资产可复用**：`MGF-Net/models/lifting_dwt.py`（PR-LWT 完美重构可学习小波）
可 1:1 迁移——"小波域去噪 + **可逆性由结构保证** + 极低参数量"是一个完整的叙事。

---

## 二、数据集

**LoDoPaB-CT**（Zenodo 3384092，**CC-BY-4.0**）

| 项 | 值 |
|---|---|
| 规模 | 约 40,000 切片 / 约 800 患者 |
| 图像 | 362×362，float32，值域 [0,1] |
| ground truth | 同患者**全剂量 FBP 重建** |
| 划分 | 官方 train / validation / test + `patient_ids_rand_*.csv`（患者级） |
| 总大小 | **55 GB**（train 46 / val 4.5 / test 4.6） |
| 官方评测 | `dival` 库（PyPI，MIT）自带下载/校验/评测/榜单 |

⚠️ **下载是本项目的实际瓶颈**：本机经代理访问 Zenodo 实测约 1–2 MB/s，
全量需 6+ 小时。故下载在后台串行进行（test → val → train）。

**为什么不用 dival 建环境**：`dival` 会把 numpy 从 2.2.6 降到 2.1.3 并引入
`odl`/`hyperopt`/`sklearn`。为避免破坏 `mgfnet` 环境，使用独立环境，
PSNR/SSIM 自行实现（该任务指标无歧义，不存在融合那样的口径争议）。

---

## 三、任务定义

```
输入  x : 低剂量 CT 重建  (362×362, [0,1])
目标  y : 全剂量 CT 重建  (同患者，同几何)
指标  PSNR / SSIM
```

**与融合的关键区别**：这里有**唯一的 ground truth**，模型的输出可以直接与
真实全剂量 CT 逐像素比较。不存在"贴近哪个源"的选择空间。

---

## 四、方法：PR-LWT 小波域去噪

```
x (含噪)
  ↓  PR-LWT 分解（2 级，7 子带，可逆性由结构保证）
  ↓  逐子带残差预测：delta_b = CNN_b(band_b)
  ↓  clean_b = band_b − delta_b
  ↓  PR-LWT 重建（与分解共享同一组 P/U）
x̂ (干净)
```

**为什么用小波域 + 完美重构**：

1. 小波域把噪声与结构在频率上分开（低频结构 / 高频噪声），是去噪的经典做法
2. **PR-LWT 的正逆变换共享同一组参数**，可逆性由结构保证——
   网络只需学习"改哪些系数"，不必补偿变换自身的损失
3. 该模块已实测：闭环相对误差 1.40e-07（对比旧式分离参数实现的 60.9%），
   且参数有界化保证 float32 数值稳定（`tests/test_lifting_pr.py`）

---

## 五、目录结构

```
LDCT/
├── README.md              本文件
├── data/
│   ├── download.sh        并行分段下载器（带断点续传）
│   ├── prepare.py         从 zip 建立索引（待数据到位后按实际格式实现）
│   ├── dataset.py         PyTorch Dataset
│   └── raw/               下载的 zip（不入库）
├── models/
│   ├── lifting_dwt.py     PR-LWT（自 MGF-Net 迁移）
│   └── denoiser.py        去噪网络
├── utils/
│   └── metrics.py         PSNR / SSIM（自行实现，任务无歧义）
└── experiments/
    └── ...
```

---

## 六、目标刊物

| 刊物 | 说明 |
|---|---|
| Biomedical Signal Processing and Control (BSPC) | 2025–2026 持续刊出去噪类工作；订阅路线版面费 0 |
| Medical Physics | 数据集本体即发表此刊，有天然连续性 |
| Physics in Medicine & Biology (PMB) | 方向对口 |
| Biomedical Optics Express | 门槛明显更低 |

---

## 七、状态（2026-09-13）

| 项 | 状态 |
|---|---|
| 方向决策与记录 | ✅ |
| 数据下载 | 🔄 进行中（55 GB，实测约 1.3 MB/s，预计 10+ 小时） |
| 数据格式确认 | 🟡 ground_truth 已实测确认；observation 待下载后确认 |
| 指标模块 + L0 测试 | ✅ `utils/metrics.py`，**8/8 通过** |
| PR-LWT 迁移 | ✅ `models/lifting_dwt.py`，验证全通过 |
| 去噪网络 | ✅ `models/denoiser.py`，**2,159 参数**，闭环 rel_L2 = 1.05e-07 |
| 训练脚本 | ✅ `experiments/train.py`（含 identity 平凡基线） |
| 指标与 dival 对齐 | ⏳ 待做（需独立环境） |
| 基线复现（RED-CNN 等） | ⏳ |
| 训练与评测 | ⏳ 等数据 |

### 已实测确认的数据格式

```
HDF5，数据集名 /data
shape = (N, 362, 362)   dtype = float32   值域 [0, 1]
ground_truth_test.zip 含 28 个分片 ground_truth_test_{000..027}.hdf5
```

### 实现中修掉的一个坑

**362 是偶数，但 362/2 = 181 是奇数**，两级小波分解会在第二级因
偶奇切分长度不等而报错。必须按 `2^levels`（而非 2）对齐边长。
`LoDoPaB` 的单片尺寸恰好踩中这一点，换成 256 或 512 就不会暴露。

### 下载瓶颈（重要）

实测经代理访问 Zenodo 的吞吐：

| 并发数 | 吞吐 |
|---|---|
| 1 | 1.0 MB/s |
| 4 | 0.58–3.1 MB/s（不稳定） |
| **16** | **2.09 MB/s** |

16 路未显著优于 4 路，**说明瓶颈在代理而非段数**。全量 55 GB 需 10+ 小时。
下载器 `data/download.sh` 支持并行分段 + 断点续传，按 test → val → train 顺序执行。

---

## 八、前身项目的可复用资产

| 资产 | 位置 | 复用方式 |
|---|---|---|
| PR-LWT 模块 + 验证测试 | `../MGF-Net/models/lifting_dwt.py`、`tests/test_lifting_pr.py` | 复制过来直接用 |
| 实验运行器（固定种子 / 验证集选模 / 完整配置记录） | `../MGF-Net/experiments/run_experiment.py` | 按需改造 |
| 病例级 K 折 | `../MGF-Net/experiments/run_cv.py` | LDCT 已有官方划分，可能不需要 |
| 受控对照 + 平凡基线的方法论 | 见 `../MGF-Net_项目状态_2026-09-13.md` §四 | **方法论照搬** |
