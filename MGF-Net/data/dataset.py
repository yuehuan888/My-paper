"""
Data loader for medical image fusion.
Supports Harvard medical image dataset (h5 format) and directory-based loading.
For proof-of-concept with limited data, uses heavy augmentation.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from PIL import Image
import random


class MedicalFusionDataset(Dataset):
    """Dataset for medical image fusion (CT-MRI, PET-MRI, SPECT-MRI).

    For proof-of-concept training with limited data, supports:
    - Random cropping with aggressive augmentation
    - Directory-based paired image loading
    """

    def __init__(self, data_path, mode='dir', patch_size=128, is_training=True,
                 source1_name='ct', source2_name='mri', oversample=20):
        self.data_path = data_path
        self.mode = mode
        self.patch_size = patch_size
        self.is_training = is_training
        self.oversample = oversample if is_training else 1

        if mode == 'h5':
            self._load_h5(data_path)
        else:
            self._load_dir(data_path, source1_name, source2_name)

    def _load_h5(self, path):
        """Load data from H5 file."""
        with h5py.File(path, 'r') as f:
            self.data = f['data'][:]
        self.data = self.data.astype(np.float32) / 255.0
        if self.data.ndim == 4 and self.data.shape[3] > 2:
            self.data = np.transpose(self.data, (0, 3, 2, 1))
        print(f"Loaded {len(self.data)} image pairs from H5, shape: {self.data.shape}")

    def _load_dir(self, path, s1_name, s2_name):
        """Load paired images from ct/ and mri/ subdirectories."""
        s1_dir = os.path.join(path, s1_name)
        s2_dir = os.path.join(path, s2_name)

        files1 = sorted([f for f in os.listdir(s1_dir) if f.lower().endswith(('.png', '.jpg', '.tif', '.bmp'))])
        files2 = sorted([f for f in os.listdir(s2_dir) if f.lower().endswith(('.png', '.jpg', '.tif', '.bmp'))])

        self.pairs = []
        for f1, f2 in zip(files1, files2):
            self.pairs.append((
                os.path.join(s1_dir, f1),
                os.path.join(s2_dir, f2)
            ))
        print(f"Loaded {len(self.pairs)} image pairs from directory")

    def _augment(self, ct, mri):
        """Apply random augmentation to paired images."""
        # Random horizontal flip
        if random.random() > 0.5:
            ct = np.fliplr(ct)
            mri = np.fliplr(mri)

        # Random vertical flip
        if random.random() > 0.5:
            ct = np.flipud(ct)
            mri = np.flipud(mri)

        # Random rotation (90-degree multiples)
        k = random.randint(0, 3)
        if k > 0:
            ct = np.rot90(ct, k)
            mri = np.rot90(mri, k)

        # Random brightness adjustment (within 10%)
        if random.random() > 0.5:
            factor = 0.9 + random.random() * 0.2
            ct = np.clip(ct * factor, 0, 1)
            mri = np.clip(mri * factor, 0, 1)

        # Small Gaussian noise
        if random.random() > 0.5:
            noise_std = random.uniform(0, 0.01)
            ct = np.clip(ct + np.random.randn(*ct.shape) * noise_std, 0, 1)
            mri = np.clip(mri + np.random.randn(*mri.shape) * noise_std, 0, 1)

        return ct, mri

    def _extract_patch(self, ct, mri):
        """Extract random or center patch."""
        H, W = ct.shape
        if H < self.patch_size or W < self.patch_size:
            pad_h = max(0, self.patch_size - H)
            pad_w = max(0, self.patch_size - W)
            ct = np.pad(ct, ((0, pad_h), (0, pad_w)), mode='reflect')
            mri = np.pad(mri, ((0, pad_h), (0, pad_w)), mode='reflect')
            H, W = ct.shape

        if self.is_training:
            y = random.randint(0, H - self.patch_size)
            x = random.randint(0, W - self.patch_size)
        else:
            y = (H - self.patch_size) // 2
            x = (W - self.patch_size) // 2

        return ct[y:y+self.patch_size, x:x+self.patch_size], \
               mri[y:y+self.patch_size, x:x+self.patch_size]

    def __len__(self):
        n = len(self.pairs) if self.mode == 'dir' else len(self.data)
        return n * self.oversample

    def __getitem__(self, idx):
        if self.mode == 'h5':
            real_idx = idx % len(self.data)
            img_pair = self.data[real_idx]
            ct = img_pair[:, :, 0]
            mri = img_pair[:, :, 1]
        else:
            real_idx = idx % len(self.pairs)
            ct_path, mri_path = self.pairs[real_idx]
            ct = np.array(Image.open(ct_path).convert('L'), dtype=np.float32) / 255.0
            mri = np.array(Image.open(mri_path).convert('L'), dtype=np.float32) / 255.0

        if self.is_training:
            ct, mri = self._augment(ct, mri)

        if self.patch_size is not None:
            ct, mri = self._extract_patch(ct, mri)

        ct = torch.from_numpy(ct.copy()).float().unsqueeze(0)
        mri = torch.from_numpy(mri.copy()).float().unsqueeze(0)

        return ct, mri


def create_dataloader(data_path, mode='dir', batch_size=16, patch_size=128,
                      is_training=True, num_workers=0, source1_name='ct',
                      source2_name='mri', oversample=20):
    """Create data loader for medical image fusion."""
    dataset = MedicalFusionDataset(
        data_path=data_path,
        mode=mode,
        patch_size=patch_size,
        is_training=is_training,
        source1_name=source1_name,
        source2_name=source2_name,
        oversample=oversample
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_training
    )
    return dataloader
