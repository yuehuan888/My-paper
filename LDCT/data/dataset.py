"""
LoDoPaB-CT 数据集加载。

数据格式（**实测确认**，非推测）
------------------------------------------------------------------
下载的 zip 解压后为若干 HDF5 分片：

    ground_truth_test_000.hdf5 ... _027.hdf5     28 个分片
    observation_test_000.hdf5  ...                （低剂量，数量应对应）

每个 hdf5：

    /data   shape=(N, 362, 362)   dtype=float32   值域 [0, 1]

注：分片数、每片切片数、dtype、值域均为实测结果
（见 `LDCT/data/raw/ground_truth_test_000.hdf5` 的检查输出）。
不同 split 的分片数与每片切片数可能不同，故本模块**不硬编码**，
一律在运行时扫描并校验。

设计原则（沿用上一个项目的教训）
------------------------------------------------------------------
- **显式校验，不静默截断**：observation 与 ground truth 的分片数、每片
  切片数必须一致，否则报错。上一个项目就吃过"两侧 zip 后静默配对"的亏。
- **不做隐式值域变换**：数据本就是 [0,1] float32，原样读出。
- **按需打开文件**：分片文件较大，不全部常驻。
"""

from __future__ import annotations

import json
import os
import re
from typing import Callable, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

SHARD_RE = re.compile(r"^(?P<prefix>.+)_(?P<idx>\d+)\.hdf5$")


def _scan(prefix_with_split: str, root: str) -> list[str]:
    """返回按分片序号排序的文件路径列表。"""
    if not os.path.isdir(root):
        raise FileNotFoundError(f"目录不存在: {root}")
    hits = []
    for fn in os.listdir(root):
        if not fn.endswith(".hdf5"):
            continue
        m = SHARD_RE.match(fn)
        if m and m.group("prefix") == prefix_with_split:
            hits.append((int(m.group("idx")), os.path.join(root, fn)))
    if not hits:
        raise FileNotFoundError(
            f"在 {root} 中未找到 {prefix_with_split}_<idx>.hdf5 形式的文件"
        )
    hits.sort()
    idxs = [i for i, _ in hits]
    if idxs != list(range(len(idxs))):
        raise ValueError(f"{prefix_with_split} 的分片序号不连续: {idxs}")
    return [p for _, p in hits]


class LoDoPaBDataset(Dataset):
    """低剂量 CT 去噪数据集：返回 (observation, ground_truth) 对。

    Args:
        root:       含 .hdf5 分片的目录
        split:      'train' | 'validation' | 'test'
        patch_size: 训练时随机裁块的大小；None 表示返回全图 362×362
        is_training: 是否做随机裁块与增强
        transform:  可选，作用于 (obs, gt) 的函数
    """

    def __init__(self, root: str, split: str = "train",
                 patch_size: int | None = None, is_training: bool = False,
                 transform: Callable | None = None,
                 cache: bool = False):
        # ⚠️ 2026-09-16：补 `cache` 形参。`train.py` 的 make_dataset 恒定传
        #    `cache=True`（`train.py:57-61` 与 :238-243），而本类原先没有这个形参
        #    -> `--dataset lodopab` 一跑就 TypeError。这是跨数据集实验的前置。
        #    LoDoPaB 的 hdf5 未压缩，逐样本开文件不如 AAPM 的 gzip 那样致命，
        #    但仍按同一语义支持缓存：首次读取时整片载入内存。
        self.root = root
        self.split = split
        self.patch_size = patch_size
        self.is_training = is_training
        self.cache = cache
        self._cache: dict = {}
        self.transform = transform

        self.gt_files = _scan(f"ground_truth_{split}", root)
        self.obs_files = _scan(f"observation_{split}", root)
        if len(self.gt_files) != len(self.obs_files):
            raise ValueError(
                f"ground_truth 与 observation 的分片数不一致："
                f"{len(self.gt_files)} vs {len(self.obs_files)}"
            )

        # 扫描每片的切片数并校验两侧一一对应
        self._shapes = []
        for g, o in zip(self.gt_files, self.obs_files):
            with h5py.File(g, "r") as f:
                ng = f["data"].shape
            with h5py.File(o, "r") as f:
                no = f["data"].shape
            if ng[1:] != no[1:]:
                raise ValueError(
                    f"切片尺寸不一致：\n  {g}: {ng}\n  {o}: {no}"
                )
            if ng[0] != no[0]:
                raise ValueError(
                    f"切片数不一致：\n  {g}: {ng[0]}\n  {o}: {no[0]}"
                )
            self._shapes.append((ng[0], ng[1], ng[2]))

        # 全局索引：第 i 个样本落在哪个分片的第几片
        self._index = []
        for fi, (n, _, _) in enumerate(self._shapes):
            self._index.extend((fi, si) for si in range(n))

    # ------------------------------------------------------------ 元信息
    @property
    def n_slices(self) -> int:
        return len(self._index)

    def slice_shape(self) -> tuple[int, int]:
        _, h, w = self._shapes[0]
        return h, w

    def summary(self) -> str:
        return (f"LoDoPaB[{self.split}]  分片={len(self.gt_files)}  "
                f"每片切片数={[n for n, _, _ in self._shapes]}  "
                f"合计={self.n_slices} 张  单片尺寸={self.slice_shape()}")

    # ------------------------------------------------------------ 取数
    def __len__(self) -> int:
        return len(self._index)

    def _read(self, fi: int, si: int):
        if self.cache:
            # 缓存**整个分片**而不是单片：单片的缓存命中率在乱序 shuffle 下接近 0，
            # 而整片载入才能消掉反复开 hdf5 的开销（这才是 cache 存在的理由）。
            if fi not in self._cache:
                with h5py.File(self.obs_files[fi], "r") as f:
                    o = np.asarray(f["data"], dtype=np.float32)
                with h5py.File(self.gt_files[fi], "r") as f:
                    g = np.asarray(f["data"], dtype=np.float32)
                self._cache[fi] = (o, g)
            o, g = self._cache[fi]
            return o[si], g[si]

        with h5py.File(self.obs_files[fi], "r") as f:
            obs = f["data"][si]
        with h5py.File(self.gt_files[fi], "r") as f:
            gt = f["data"][si]
        return (np.asarray(obs, dtype=np.float32),
                np.asarray(gt, dtype=np.float32))

    def __getitem__(self, i: int):
        fi, si = self._index[i]
        obs, gt = self._read(fi, si)

        if self.transform is not None:
            obs, gt = self.transform(obs, gt)

        if self.patch_size is not None:
            h, w = obs.shape
            p = self.patch_size
            if h < p or w < p:
                raise ValueError(f"patch_size={p} 大于图像尺寸 {(h, w)}")
            if self.is_training:
                # 同一位置裁两图，保证空间对应
                rng = np.random.default_rng()
                y = int(rng.integers(0, h - p + 1))
                x = int(rng.integers(0, w - p + 1))
            else:
                y, x = (h - p) // 2, (w - p) // 2
            obs = obs[y:y + p, x:x + p]
            gt = gt[y:y + p, x:x + p]

        # 加通道维 -> [1, H, W]
        return (torch.from_numpy(obs).unsqueeze(0),
                torch.from_numpy(gt).unsqueeze(0))


def load_split(root: str, split: str, patch_size: int | None = None,
               is_training: bool = False) -> LoDoPaBDataset:
    return LoDoPaBDataset(root, split, patch_size=patch_size,
                          is_training=is_training)


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "raw", "extracted")
    for sp in ["validation", "test"]:
        try:
            ds = LoDoPaBDataset(root, sp)
            print(ds.summary())
            o, g = ds[0]
            print(f"    样本 0: obs {tuple(o.shape)} [{o.min():.4f},{o.max():.4f}] "
                  f"gt {tuple(g.shape)} [{g.min():.4f},{g.max():.4f}]")
        except FileNotFoundError as e:
            print(f"  [{sp}] 数据尚未就绪: {e}")


# ======================================================================
# AAPM-Mayo 2016 支持
# ======================================================================

class AapmDataset(Dataset):
    """AAPM-Mayo 2016 LDCT 去噪数据集（每患者一个 h5）。

    h5 结构（由 `prepare_aapm.py` 生成）：
        /observation    (N, 512, 512) float32  [0,1]
        /ground_truth   (N, 512, 512) float32  [0,1]
        attrs: patient_id, n_slices, z_positions

    划分按**患者**，避免同一患者的相邻切片跨集合（读 splits/*.json）。
    """

    def __init__(self, root: str, split_file: str, split: str = "train",
                 patch_size: int | None = None, is_training: bool = False,
                 cache: bool = False):
        with open(os.path.join(root, "index.json"), encoding="utf-8") as f:
            self.index = json.load(f)
        with open(split_file, encoding="utf-8") as f:
            sp = json.load(f)

        key = {"train": "train_patients", "val": "val_patients",
               "test": "test_patients"}[split]
        self.split = split
        self.patients = list(sp[key])
        if not self.patients:
            raise ValueError(f"划分 {split_file} 的 {key} 为空")

        missing = [p for p in self.patients if p not in self.index["patients"]]
        if missing:
            raise ValueError(f"划分中的患者不在数据集中: {missing}")

        self.root = root
        self.patch_size = patch_size
        self.is_training = is_training
        self.hu_window = tuple(self.index.get("hu_window", [0, 1]))

        # 扁平索引：(患者, 切片号)
        self._index = []
        for p in self.patients:
            n = self.index["patients"][p]["n_slices"]
            self._index.extend((p, i) for i in range(n))

        # 可选内存缓存。
        # 动机：单线程逐样本打开 gzip 压缩的 HDF5 极慢——
        # 上一轮训练在第一轮就卡了 30 分钟（1600 片 × 每片开一次文件）。
        # 训练集 1600×512×512×4B×2 ≈ 3.4 GB，本机内存足够。
        self.cache = cache
        self._cache = {}
        if cache:
            total = 0
            for p in self.patients:
                h5p = os.path.join(self.root, f"{p}.h5")
                with h5py.File(h5p, "r") as f:
                    self._cache[p] = (f["observation"][:], f["ground_truth"][:])
                total += self._cache[p][0].nbytes * 2
            print(f"    [cache] {split}: 已载入内存 {total/1073741824:.2f} GB")

    @property
    def n_slices(self) -> int:
        return len(self._index)

    def slice_shape(self):
        p0 = self.patients[0]
        sh = self.index["patients"][p0]["shape"]
        return sh[0], sh[1]

    def summary(self) -> str:
        return (f"AAPM[{self.split}]  患者={self.patients}  "
                f"切片={self.n_slices}  尺寸={self.slice_shape()}  "
                f"HU窗={self.hu_window}")

    def __len__(self):
        return len(self._index)

    def __getitem__(self, i):
        pid, si = self._index[i]
        if self.cache:
            o, g = self._cache[pid]
            obs = np.asarray(o[si], dtype=np.float32)
            gt = np.asarray(g[si], dtype=np.float32)
        else:
            h5p = os.path.join(self.root, f"{pid}.h5")
            with h5py.File(h5p, "r") as f:
                obs = np.asarray(f["observation"][si], dtype=np.float32)
                gt = np.asarray(f["ground_truth"][si], dtype=np.float32)

        if self.patch_size is not None:
            h, w = obs.shape
            p = self.patch_size
            if h < p or w < p:
                raise ValueError(f"patch_size={p} > 图像尺寸 {(h, w)}")
            if self.is_training:
                # ⚠️ 2026-09-16 修：此处原为 `np.random.default_rng()`。
                #    那是**全新生成器**，由 OS 熵播种，**完全绕开**
                #    train.py 里的 random.seed / np.random.seed / torch.manual_seed。
                #    后果：同一 `--seed` 跑两次，裁块位置不同 -> 结果本就不同，
                #    种子间方差被人为抬高。这直接污染了全部配对统计。
                #
                #    改用 torch 的全局 RNG：它已被 train.py 按 `--seed` 播种，
                #    且状态会随训练自然推进 —— 既**可复现**，又**保住逐轮多样性**
                #    （若改成按样本号派生固定种子，1600 个样本就只有 1600 种裁块，
                #     30 轮里反复用同一批，等于削弱增广）。
                y = int(torch.randint(0, h - p + 1, (1,)).item())
                x = int(torch.randint(0, w - p + 1, (1,)).item())
            else:
                y, x = (h - p) // 2, (w - p) // 2
            obs, gt = obs[y:y+p, x:x+p], gt[y:y+p, x:x+p]

        return torch.from_numpy(obs).unsqueeze(0), torch.from_numpy(gt).unsqueeze(0)
