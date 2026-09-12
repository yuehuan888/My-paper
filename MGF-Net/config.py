"""Configuration for MGF-Net training and testing."""

import os

class Config:
    # Paths
    data_path = r"D:\图像相关论文\MGF-Net\data\train"  # Directory with ct/ and mri/ subdirs
    data_mode = 'dir'  # 'dir' or 'h5'
    test_ct_dir = r"D:\图像相关论文\MGF-Net\data\test\ct"
    test_mri_dir = r"D:\图像相关论文\MGF-Net\data\test\mri"
    output_dir = r"D:\图像相关论文\MGF-Net\outputs"
    checkpoint_dir = r"D:\图像相关论文\MGF-Net\checkpoints"

    # Training (small data proof-of-concept)
    batch_size = 4
    patch_size = 128
    epochs = 100
    oversample = 25  # Augment 8 images * 25 = 200 samples per epoch
    learning_rate = 1e-3
    lr_decay = 0.5
    lr_decay_epochs = [40, 70, 90]

    # Loss weights
    alpha_ssim = 1.0    # SSIM loss weight
    beta_l1 = 10.0      # L1 loss weight
    gamma_grad = 5.0    # Gradient loss weight
    delta_freq = 0.5    # Frequency loss weight (reserved)

    # Model
    in_channels = 1
    mid_channels = 32  # v2: internal channels for FusionBlocks
    learnable_dwt = True

    # GPU
    device = 'cuda'
    num_workers = 0
    use_amp = True  # Mixed precision for 4GB VRAM

    # Logging
    log_interval = 10
    save_interval = 10

    @classmethod
    def ensure_dirs(cls):
        os.makedirs(cls.output_dir, exist_ok=True)
        os.makedirs(cls.checkpoint_dir, exist_ok=True)


config = Config()
