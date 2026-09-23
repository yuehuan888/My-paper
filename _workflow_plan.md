核查完毕。所有结论我都落到盘上产物上验证过，包括几处候选内部互相矛盾的关键数字。以下是重构方案。

---

# PR-LWT 论文重构方案

## 0. 先说三个我独立核实、且会改变计划的事实

**(a) §5.3 的 S1 闭环数字确实查无出处，且已提交的图与表当场打架。**
`roundtrip_checkpoints_w3.json` = `unconstrained: 0.1491628, n=1`（S1 唯一产物），论文 §5.3 写 `1.25e-01`（main.tex:531、md:387、zh:252）。我直接读了已渲染的 `paper/figures/fig2_mech.png`：面板 (a) 上印的是 **1.49e-01**。`calibrate_mismatch.py:46` 把 `TARGET=1.253e-01` 硬编码并注释"论文 §5.3"——循环引用。e9d7113 的 commit message 自述修掉了"99% 查无出处"，却用另一个查无出处的数替换。**图是对的，表是错的，改表即可自洽。**

**(b) `fig_mech2()` 已经存在且是默认产物（`want()` 在 `--only=None` 时同时跑 mech 和 mech2，后者覆盖前者）。** 双面板机制图已提交、已渲染、已在正文 §5.3:380 被引用。任何"新建机制图"的提案都是白做。

**(c) 三个关键脚本都已写好但从未执行**（`run_ablation.ps1` / `run_seeds10.ps1` / `run_redcnn_seeds.ps1`，全 untracked，`runs/` 下无 `bnd*` 目录）。且 `.pth` 全仓为 0，原始训练机路径 `D:\图像相关论文\LDCT` 本机不存在。

另外我复算了几个会写进论文的数：per-patient 配对效应 **L506 +0.2441/+0.2148、L067 +0.3475/+0.3265**（两患者同号，难患者效应更大）；SSIM 三臂 **pr 0.8762 > fixed 0.8729 > unconstrained 0.8725**，dz = +3.40 / +4.07；`history.json` 的 `val_psnr` 只有 **6 个点**（val_interval=5），证实 fig6 横轴 bug。

---

## 1. 最终应该主张的四个贡献（按"现在能否立刻支撑"排序）

| # | 贡献 | 现状 | 立刻可写？ |
|---|---|---|---|
| **C1** | **缺失的对照 + 以 bound 表述的负结果**：无约束可学习小波与固定 Haar 不可区分（−0.025 dB，CI 含零，5.6× 低于 n=5 检出线 0.136 dB），故报"任何真实差异 < 0.14 dB" | 已完整，MDES/功效/森林图/功效曲线全在 | ✅ 是（只需把 S1 数字改对） |
| **C2** | **因果机制**：注入受控闭环误差 → PR 臂单调退化至固定 Haar 水平（8.7% → 30.5500 vs 30.5591，各档 p ≤ 0.001） | 已完整，图已存在。建议把横轴由 ε 换成**实测 relL2**（4 点已在盘上），使图 (a)(b) 同轴 | ✅ 是 |
| **C3** | **PR 的失效条件是数值条件数，不是可逆性本身**：结构 PR 在 float32 下不自证；taps 有界化是**条件数要求**（实测 κ 跨 5 个数量级 vs 闭环误差单调）；Haar 初始化是该 3-tap 族唯一的 κ=1 点 | 目前只有"一个观测点"（3.6e-07→5.2e+00）。**这正是最该升级的一条** | ⚠️ 零算力可升级一半；剂量-响应需 4 次训练 |
| **C4** | **2,159 参数网络 + 双指标效率**：距同协议 RED-CNN 1.0 dB、857× 参数；补 SSIM 后捕获率 85–87%（PSNR 口径仅 74%） | 表里 PR-LWT 的 SSIM 是"—"，数字在盘上 | ✅ 是（零算力补表） |

**关键判断：C2 是最强的，C1 是最容易被误读成"跟 WINNet 撞车"的，C3 是最弱但最独特的。所以摘要/引言的顺序必须把 C2、C1 提到 WINNet 重叠部分之前——这是零算力、也是收益最大的单步改动。**

排序理由：C1/C2 的全部数字已在盘上且经我复算，唯一障碍是 S1 那一个格子的勘误；C3 的"闭式解"作为定理站不住（教科书内容、且低估了论文自己可行箱的最坏情形 11 倍、κ₁^(2L) 界偏离实测 12 个数量级），但**降级后的量测版**——"闭环误差随真实算子条件数单调、有界化是条件数要求"——是零算力、诚实、可复核的增量。**不要把闭式解立成贡献点。**

---

## 2. 每个贡献：现在写 vs 补什么

**C1 — 现在写。** 唯一依赖：§5.3 勘误（见 §5 Tier 0）。

**C2 — 现在写。** 把 `intervention_results.json` 的 4 个 `roundtrip` 值（1.6e-07 / 0.0553 / 0.0871 / 0.1178）作为横轴重画 fig2(b) 横轴标签（`make_figures.py` 已用 `cells[k]["roundtrip"]` 作横轴，只需把图注与正文表述同步）。已存在的 "Honest boundary" 段必须保留并**前移到图注**——面板 (b) 把无约束臂画成同坐标系的点，读者一眼会看出它比自身误差处的曲线预测高约 0.25–0.46 dB；主动说明"双端失配 vs 单端注入"比被审稿人发现好。

**C3 — 分两步：**
- **现在（零算力）**：写 `experiments/cond_sweep.py`，对 tap 幅度 scale ∈ {1.0…5.0} 构造 PR-LWT bank，用 `cond_analysis.analysis_matrix` + `np.linalg.svd` 求**真实**多层二维算子条件数，同跑 roundtrip。产出一张 log-log κ-vs-relL2 图 + 一段文字（≤0.3 页、不新增小节、不新增定理环境）。保留一句"κ₁=1 仅在该族 Haar 初始化处取得"。**删除**：一般 3-tap 的 max_ω 闭式、"最坏情形在 DC"、"随机 tap 比值 1.00000000"（有限算子是 O(1/n) 逼近，非恒等）。
- **重建后（4 次训练）**：`powershell -File experiments/run_ablation.ps1 -Only bound`（bound ∈ {1.0,2.0,4.0,8.0}，seed 0）。这检验**PSNR 是否跟随闭环误差**——若跟随，C3 就获得与 C2 同级的因果地位。这是性价比最高的实验。

**C4 — 现在写。** 用 `results.json` 的 `test_mean.SSIM` 填表（S1 pr 0.8762±0.00112 / S2 0.90406±0.00085；RED-CNN 0.8824 / 0.9090），加一句"以 identity→RED-CNN 增益为分母（本文自设口径）"。**前置**：先核实 RED-CNN 的 SSIM 实现口径（`gaussian_weights/sigma/win_size`），否则等于把未验证指标升进主表。

---

## 3. 摘要与 Introduction 改写草稿（英文，可直接用）

### Abstract

> Making the wavelet transform learnable is a natural extension of wavelet-domain image denoising, and recent invertible-wavelet denoisers build perfect reconstruction (PR) into the architecture. Whether that structure is *what makes learning pay* has never been tested: prior work assumes invertibility, so it cannot observe its absence. We make invertibility **optional** in a controlled three-arm study — an unconstrained learnable analysis–synthesis pair, a structurally perfect-reconstructing lifting wavelet (PR-LWT), and fixed Haar — holding capacity, split, schedule and evaluation fixed. Imposing PR makes learning pay: **+0.30 dB over unconstrained learning (p = 0.0005, $d_z$ = 4.67)** and **+0.27 dB over fixed Haar (p = 0.0004, $d_z$ = 4.78)**, replicated on a second data split. Without it, learning buys nothing: the unconstrained arm is indistinguishable from fixed Haar (−0.025 dB, 95% CI [−0.060, +0.009]), **5.6× below the $n=5$ detection floor of 0.136 dB** — so we state the result as a **bound**, not an absence: any genuine difference is smaller than 0.14 dB, roughly half of what PR delivers. The mechanism is the transform's own reconstruction error: after training the unconstrained pair loses **12–16%** of the signal in a forward–backward pass, against **1.6e-07** — the float32 limit — for both invertible arms, a gap of six orders of magnitude. That magnitude is not merely correlated with the failure but **sufficient** to cause it: injecting a controlled round-trip error into the PR arm degrades it monotonically to the fixed-Haar level (8.7% error → 30.5500 dB vs. 30.5591 dB; p ≤ 0.001 at every dose). Structural PR is only as usable as its numerical conditioning, so we bound the lifting taps; without the bound the round-trip error degrades from 3.6e-07 to ~5. The resulting network has **2,159 parameters**, reaches **31.97 dB** on AAPM-Mayo 2016 — within 1.0 dB of a RED-CNN baseline trained under the identical protocol at **857× fewer parameters** — and captures **85–87%** of the identity-to-RED-CNN SSIM gain.

（约 250 词。若需压到 200 词，删最后一句的前半，保留 SSIM 捕获率。）

### Introduction（替换 §1 的 34–77 行）

> Low-dose CT (LDCT) reduces radiation exposure at the cost of increased noise. Deep denoisers such as RED-CNN [1] are effective but parameter-heavy (~10⁶ parameters), so compact alternatives are attractive. A classical route is to denoise in a wavelet domain; a recent one is to make the wavelet itself **learnable**, so the transform adapts to modality and dose [17]–[22]. This paper asks not *how small can a denoiser be* but a prior question:
>
> > **Does making the wavelet learnable actually help?**
>
> Our answer, from a controlled three-arm study, is: **only if the transform is kept invertible by construction.** Parameterizing the wavelet as a **lifting scheme** [23], [24] — the inverse *reuses* the forward prediction/update filters rather than being a second set of parameters — makes learning pay off: **+0.30 dB over unconstrained learning and +0.27 dB over fixed Haar**, at large effect size ($d_z \approx 4.7$) and consistent across two data splits.
>
> The inverted case is equally informative. An unconstrained learnable analysis–synthesis pair performs **statistically indistinguishably from a fixed Haar basis**. We are careful how we state this: the design resolves a 0.136 dB effect at $n=5$ seeds, whereas the unconstrained–fixed gap is 0.025 dB — **5.6× below that floor**. So we report a **bound**, not an absence: any genuine difference is smaller than **0.14 dB**, roughly half of what PR delivers. A weak experiment could not have resolved the effect that PR produces.
>
> **Why** it fails is the substance of the paper. After training, the unconstrained pair loses **12–16%** of the signal in a forward–backward pass, against **1.6e-07** for both invertible arms. We show this is **causal, not merely correlated**: holding architecture, capacity and protocol fixed, we inject a controlled round-trip error into the PR arm, and it degrades **monotonically** to the fixed-Haar level (8.7% error → 30.5500 dB, against 30.5591 dB for an arm that never learned anything; p ≤ 0.001 at every dose). Prior invertible-wavelet denoisers cannot observe this state, because their designs make it unreachable.
>
> **Contributions.**
> 1. **A missing control**, and the negative result it exposes: with capacity and protocol held fixed, an unconstrained learnable wavelet gives no benefit over fixed Haar — reported as a bound of 0.14 dB, not as an absence.
> 2. **A causal mechanism**, established by intervention, not correlation: a round-trip error of the same magnitude is *sufficient* to destroy the benefit, monotonically and significantly. Prior invertible-wavelet denoisers cannot observe this state.
> 3. **A conditioning requirement, not merely a reversibility one**: structural PR does not enforce itself in float32. Bounding the lifting taps is what makes it usable — without the bound the round-trip error degrades from 3.6e-07 to ~5 — and the error tracks the condition number of the multi-level analysis operator, not the PR property as such. The invertible-network literature does not report this.
> 4. **A 2,159-parameter network** that comes within 1.0 dB of a same-protocol RED-CNN baseline at 857× fewer parameters, and captures 85–87% of its SSIM gain.

**改动要点**：(i) 把 C2/C1 前置，WINNet 相关的"可学习小波"降为背景；(ii) 摘要去掉 `12–15%` 改为 `12–16%`；(iii) 贡献 3 从"实践警告"改写为"条件数要求"；(iv) 贡献 1 用 bound 表述（比 "negative result" 更强，不退回弱标签）。

---

## 4. 应明确放弃的候选（及理由）

**放弃 A 类：数学/理论包装（会反噬论文）**
- **单级提升格式条件数闭式解** 作为定理/贡献点。教科书内容（Bölcskei 1998 已建立 frame bound = polyphase 特征值）；它低估论文自己可行箱的最坏情形 **11 倍**（箱顶点真实 κ₁=43.6，实测 relL2 7.1e-06，比论文报的 3.6e-07 差 20 倍）；κ₁^(2L) 界在 L=2 就给 9.78e-02、已超有害阈值，与论文"L=2 安全"的前提冲突。**只保留降级版一句 + 量测曲线。**
- **"Haar 初始化是条件数全局最优 / 学习必然让条件数变差"**。当且仅当为假（(−1,−½) 是第二个极小点）；且 GD 不优化 κ₁，实测训练后闭环 1.64e-07→1.55e-07 未退化。
- **"bound 不随深度迁移"设计律**。声称的六个数字全仓零命中；其定理与自身实测矛盾；cond_analysis.py 无 `__main__`，docstring 声称的自检不存在。
- **"控制量应是 taps 之和而非 max|tap|"**。处方危险：Σtap=0 时允许 κ₁=3.4e5、relL2=1.78e10。max|tap| 恰恰是对的量。
- **√2 是罪魁**：作为实验性反驳被驳回（`κ(DA) ≤ 2κ(A)` 是一行定理），但**保留一句脚注**排除它，不引数字。
- **borrowed "representation error" 命名**：误用 Asim 的定义（其定义下两臂都 = 0）。正确术语是 **perfect-reconstruction error**，正确引用是 **Behrmann et al., AISTATS 2021（爆炸逆）**。

**放弃 B 类：数据考古的再分析（伪重复/同源/已被论文覆盖）**
- 逐切片 2175/2175 全同号、逐切片 SSIM 胜率、干预逐切片剂量响应 → 与论文 §6 明文声明的"以种子为分析单元"冲突，是审稿人一句 pseudoreplication 就打回的自相矛盾。
- 难度五分位分层、"PR 增益随难度单调" → 合并分层被患者构成主导（Q5 87 张里 85 张来自 L506），患者内非单调且低于检出线。**只保留 per-patient 配对一致性**（L506/L067 同号、难患者更大）。
- "每单位误差杀伤力 1.68×" → 循环恒等式；支点 0.149 与论文表 1.25e-01 冲突；+0.52 dB 是越界 27% 外推。
- 30 轮未收敛伪影 / 方差反转 / 收敛性下界 → 已是 limitation 3/6；"方差反转"在 S1 由单个种子驱动、在 S2 反向。
- pilot 跨代码版本复现 → `git log 7c282fc..9b0faad -- lifting_dwt.py` **为空**，机制代码逐字节未变，检验的是评测管线不是机制。
- S2 是 S1 子集 → 已在 §4.3/§4.4/§6.2 披露；且难患者 L067 效应更大，直接反驳"效应来自易患者"。
- RED-CNN 数据饥饿线索 → 正确口径下方向反转（PR +0.084 vs RED-CNN +0.068）；低于论文自报 MDES 近 2 倍。
- 非参 p 值下界（0.0625）→ 会把摘要的 p<0.001 改成"不显著"，是净负收益。**只加一条脚注**说明以效应量与 MDES 立论。
- 换 HU 口径重评 / 第二数据集 / 下游任务 / 患者级混合效应 → 前者需重训全部；后两者是已知 limitation，且 2 患者无法拟合混合效应。

**放弃 C 类：写作模板与已存在的产物**
- ICML/AAAI 摘要与贡献列表模板照搬 → 第 5/6 步在本文无诚实对应物（无数学分析、不刷 SOTA）；"first" 类措辞会被 DeSPAWN 的 CQF 消融击穿。
- 双面板机制图、"逐切片散点 + 训练曲线"新图 → **全部已存在**（`fig_mech2()`、`fig5_slice_dist.png`、`fig6_convergence.png`）。
- Reverse-Convolution / PR-松弛文献作为新颖性威胁 → 现象不同、方向被读反。
- 无训练基线（BM3D/软阈值）/ 正交臂 / 实跑 LINN-WINNet → 加基线按定义不是贡献；正交臂是 WINNet 已做过的实验；后者是重工程。**前者的正确用法是补一行 Related Work 论证"无约束臂就是领域默认"（引 LWFSN, TMI 2022）。**

---

## 5. 按性价比排序的行动清单

### Tier 0 — 零算力，现在做（论文正确性，阻塞投稿）
1. **§5.3 S1 勘误**：`1.25e-01` → `1.49e-01` 并标注 n（S1 n=1、S2 n=3）；删 `on the final checkpoint of every seed`（md:363-364 与 **main.tex 表注:522** 都写了）；删 `S1 per-seed range 0.11–0.15`。同步改 `calibrate_mismatch.py:46-48`（去掉"论文 §5.3"循环出处）、`README.md:111-112`、`conference_paper_zh.md:252`。
2. **口径统一**：全文 `12–15%` → `12–16%`；补 `S1 14.9% / S2 均值 14.5% / 逐种子 11.9–16.4%`。
3. **修 fig6 横轴 bug**：`make_figures_ext.py:187` 的 `ep = np.arange(1, len(m)+1)` → 验证轮次 `[5,10,15,20,25,30]`。当前把 6 个验证点画在 x=1..6，时间轴压缩 5 倍。
4. **补 SSIM**：Table 5.1 两格 + §5.2 / 干预表加 SSIM 列（数字见 §2 C4）。这是我复算后确认的**加强**项，不是削弱项（dz +3.40/+4.07，S2 复现）。
5. **摘要 + 引言重排**：按 §3 草稿。这是零算力收益最大的单步。
6. **效率主张限定为容量**：§1/§5.1 加一句 "our efficiency claim is a capacity claim (857× fewer parameters), not a speed claim; latency, FLOPs and peak memory were not measured."
7. **per-patient 一致性句**（§5.2）："both test patients show the same direction and the harder one more strongly (PR−fixed: L506 +0.215, L067 +0.327 dB)"。
8. **参考文献修补**：[14] FMDNet 在英文 §2 被删（补回 3 个词）；[29] 散射网络全文未引（删或引）；英文 §6 补 LoDoPaP 说明（中文版 line 279 有现成句子）；**新增 Behrmann et al., AISTATS 2021**（贡献 3 的对口文献）与 **Zavala-Mondragon et al., TMI 2022 / LWFSN**（证明无约束臂是领域默认而非自造稻草人）。
9. **代码注释勘误**：`lifting_dwt.py:97`（`bound=0.5 → P∈[−0.5,1.5], U∈[−0.5,1.0]`，"|taps|≤1.5" 只对 P 成立）；`lifting_dwt.py:86-88` 中间行 `3.5 → 1.7e-02`（实测，非 1e-04）；`cond_analysis.py` docstring 的"见 __main__ 的自检"（不存在）删除或补上 `__main__`；`verify_roundtrip.json` 残留的 `paper_relL2: 0.99`。
10. **防呆 + 可复算（3 行）**：`verify_roundtrip.py` 的 `from_checkpoints` 在写出空 dict 时**断言失败**（现在静默 `continue`，会覆盖唯一幸存记录）；`train.py` 每轮把 `transform_roundtrip` 写进 `history.json`（这样 checkpoint 丢了也能重建闭环数）。
11. **C3 加固段**（≤0.3 页）：写 `experiments/cond_sweep.py`（numpy-only），出 log-log κ-vs-relL2 图 + 一句"κ₁=1 仅在该族 Haar 初始化处取得"。**不新增小节、不给闭式。**

### Tier 1 — 重建后廉价、高价值
12. **S1 闭环 n=1 → n=5**（修 CONFIRMED 勘误的根治）：**先确认原始训练机是否还留有 `w3_unconstrained_s*/best.pth`**——若有，几乎免费：
    ```
    python experiments/verify_roundtrip.py --checkpoints --prefix w3_ --n-seeds 5
    ```
    若无，重训 `w3_unconstrained_s0..s4`（5 × 30 epoch）后运行上句。附带把 pr/fixed 的 S1 也补到 n=5（同一命令一次跑完）。
13. **bound 剂量-响应**（把 C3 升到因果同级）：
    ```
    powershell -File experiments/run_ablation.ps1 -Only bound   # 4 runs, seed 0
    ```
    若曲线干净（PSNR 跟随闭环误差），把最有信息的两档扩到 n=5。

### Tier 2 — 重建后中等成本
14. `-Only size`（4 runs）：回应"换变换尺寸结论是否成立"。
15. `run_seeds10.ps1`（15 runs）：把界从 0.14 收到 0.09 dB——这是让"任何真实差异 < X"这一可引用否定命题变强的唯一途径。
16. `run_redcnn_seeds.ps1`（8 runs × 6× 成本）：消除 limitation 4 的 n=1 不对称。
17. （可选）`bounded_unc` 臂（5 runs）：64 参数 + 有界 + 无 PR，闭合"是 PR 还是 tanh 盒子"的最后一个口子。优先级低于 13，因为 ε 干预已在 PR 臂内部回答了它。

### 版面
现稿正文 ≈3.3 页 + 6 表 4 图 ≈ 4.5–5 页，EI 上限 5 页。C3 加固段 + SSIM 列 + 引言重排约 +0.3 页。**补偿方案：合并 `fig1_arms` 与 `fig2(a)`**（都是三臂对比，只差纵轴），可省约 0.3 页。切勿新增第 10 张图。

---

**一句话总结给决策用**：这篇论文最终应主张的是"**缺失的对照 + 因果干预 + 条件数要求**"三件事，而不是"我们提出了一个可学习小波"。所有需要新算力的项都排在零算力勘误之后；最该先做的两件事，一是改对 §5.3 那一个格子和重排摘要/引言的顺序（零算力、今天就完），二是重建后先跑 `verify_roundtrip.py --checkpoints --prefix w3_ --n-seeds 5` 和 `run_ablation.ps1 -Only bound`（合计 4–9 次训练），其余一律不做。