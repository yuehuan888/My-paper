# 数据集说明

本仓库**不包含**训练/测试图像。原因：

1. 医学影像数据集体积较大，不适合放在 Git 仓库中；
2. 数据集由第三方发布，其许可条款与本仓库代码许可相互独立。

## 获取方式

本项目使用 **Harvard Medical School / AANLIB** 公开医学影像数据集（经配准的 CT-MRI、PET-MRI、SPECT-MRI 图像对）。

推荐下载源：

- `https://github.com/hanna-xu/Harvard-Medical-Image-Dataset` —— U2Fusion / EMFusion 作者维护的配准版本，也是多数医学融合论文使用的版本
- 亦可从 U2Fusion、SeAFusion、CDDFuse 等官方仓库的 README 中获取其使用的数据划分

## 目录结构

下载后按如下结构放置：

```
data/
├── train/
│   ├── ct/      # CT 图像
│   └── mri/     # MRI 图像（文件名需与 ct/ 中的一一对应）
└── test/
    ├── ct/
    └── mri/
```

对应关系由**文件名**建立：`data/train/ct/1.png` 与 `data/train/mri/1.png` 构成一对。

## 数据划分

> ⚠️ 论文中报告的所有结果必须基于**固定的、公开可查的数据划分**。

划分好的文件 ID 列表存放于 `splits/` 目录，随代码一同提交。复现实验时请直接使用该划分，不要重新随机划分，否则结果无法与论文对齐。

（当前项目处于 Phase A 阶段，划分文件尚未建立。详见仓库根目录的 `发表路线与实验计划_2026-09-12.md`）

## 引用

使用该数据集时请引用其原始来源：

```bibtex
@article{xu2020u2fusion,
  title   = {U2Fusion: A Unified Unsupervised Image Fusion Network},
  author  = {Xu, Han and Ma, Jiayi and Jiang, Junjun and Guo, Xiaojie and Ling, Haibin},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year    = {2022}
}
```
