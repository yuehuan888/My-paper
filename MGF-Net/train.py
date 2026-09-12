"""
Training script for MGF-Net (GPU-optimized + live visualization).

Usage:
    python train.py
"""

import os, sys, time, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import config
from models.mgf_net import MGFNet, count_parameters
from losses import MGFusionLoss
from data.dataset import MedicalFusionDataset


class ProgressMeter:
    """Real-time training progress display with GPU monitoring."""
    def __init__(self, num_batches, epochs):
        self.num_batches = num_batches
        self.epochs = epochs
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.losses = AverageMeter()

    def display(self, epoch, batch, loss, loss_dict, lr, batch_time, data_time):
        gpu_info = ''
        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated() / 1024**2
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            gpu_info = f' | GPU: {mem_used:.0f}/{mem_total:.0f}MB'

        bal = loss_dict.get('bal', 0)
        bal_info = f'Bal:{bal:.4f}' if 'bal' in loss_dict else ''
        print(f'\rEpoch [{epoch+1:3d}/{self.epochs}] '
              f'Batch [{batch+1:3d}/{self.num_batches}] '
              f'Loss: {loss:.4f} '
              f'| S:{loss_dict["ssim"]:.4f} '
              f'L1:{loss_dict["l1"]:.4f} '
              f'G:{loss_dict["grad"]:.4f} '
              f'| LR:{lr:.6f} '
              f'| {bal_info} '
              f'| Time:{batch_time:.1f}s{gpu_info}', end='')


class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def plot_live(loss_history, save_path, epoch):
    """Plot and save training curves in real-time."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'MGF-Net Training Progress - Epoch {epoch}', fontsize=14, fontweight='bold')

    colors = {'total': '#1f77b4', 'ssim': '#2ca02c', 'l1': '#d62728', 'grad': '#ff7f0e'}
    titles = {'total': 'Total Loss', 'ssim': 'SSIM Loss', 'l1': 'L1 Loss', 'grad': 'Gradient Loss'}

    for ax_idx, key in enumerate(['total', 'ssim', 'l1', 'grad']):
        row, col = ax_idx // 2, ax_idx % 2
        if key in loss_history and len(loss_history[key]) > 0:
            x = np.arange(len(loss_history[key]))
            axes[row, col].plot(x, loss_history[key], color=colors[key], alpha=0.8, linewidth=0.8)
            # Smoothed trend
            if len(loss_history[key]) > 20:
                kernel = np.ones(20) / 20
                smoothed = np.convolve(loss_history[key], kernel, mode='valid')
                axes[row, col].plot(np.arange(len(smoothed)) + 19, smoothed, 'k-', linewidth=1.5, alpha=0.7)
        axes[row, col].set_title(titles[key])
        axes[row, col].set_xlabel('Step')
        axes[row, col].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def train():
    config.ensure_dirs()

    # GPU setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        print(f'[GPU] {torch.cuda.get_device_name(0)}')
        print(f'[GPU] Memory: {torch.cuda.get_device_properties(0).total_memory/1024**2:.0f} MB')
        # Mixed precision only on GPU
        use_amp = True
    else:
        device = torch.device('cpu')
        use_amp = False
        print('[WARNING] No GPU found, using CPU!')

    print(f'[Device] {device}')

    # Dataset
    dataset = MedicalFusionDataset(
        data_path=config.data_path, mode=config.data_mode,
        patch_size=config.patch_size, is_training=True,
        oversample=config.oversample
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size,
                            shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    print(f'[Data] {len(dataset)} samples, {len(dataloader)} batches/epoch')

    # Model
    model = MGFNet(in_channels=config.in_channels, mid_channels=config.mid_channels,
                   learnable_dwt=config.learnable_dwt).to(device)
    n_params = count_parameters(model)
    print(f'[Model] {n_params:,} parameters')

    # Loss & Optimizer
    criterion = MGFusionLoss(alpha=config.alpha_ssim, beta=config.beta_l1, gamma=config.gamma_grad).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.lr_decay_epochs, gamma=config.lr_decay)

    # Mixed precision
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # Tracking
    loss_history = {'total': [], 'ssim': [], 'l1': [], 'grad': []}
    epoch_losses = AverageMeter()
    best_loss = float('inf')
    start_time = time.time()
    progress = ProgressMeter(len(dataloader), config.epochs)

    print(f'\n{"="*60}')
    print(f'Training: {config.epochs} epochs | BS={config.batch_size} | Patch={config.patch_size}')
    print(f'LR={config.learning_rate} | AMP={use_amp}')
    print(f'{"="*60}\n')

    step = 0
    for epoch in range(config.epochs):
        model.train()
        epoch_losses.reset()
        batch_start = time.time()

        for batch_idx, (ct, mri) in enumerate(dataloader):
            data_time = time.time() - batch_start
            ct, mri = ct.to(device), mri.to(device)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast('cuda'):
                    fused = model(ct, mri)
                    loss, loss_dict = criterion(fused, ct, mri)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                fused = model(ct, mri)
                loss, loss_dict = criterion(fused, ct, mri)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            step += 1
            epoch_losses.update(loss.item())
            batch_time = time.time() - batch_start

            for k in loss_history:
                if k in loss_dict:
                    loss_history[k].append(loss_dict[k])

            # Live progress display
            progress.display(epoch, batch_idx, loss.item(), loss_dict,
                           optimizer.param_groups[0]['lr'], batch_time, data_time)

            batch_start = time.time()

        # End of epoch
        scheduler.step()
        print()  # newline

        avg_loss = epoch_losses.avg
        lr_now = optimizer.param_groups[0]['lr']
        elapsed = time.time() - start_time
        eta = (elapsed / (epoch + 1)) * (config.epochs - epoch - 1)

        print(f'--- Epoch {epoch+1}/{config.epochs} | '
              f'Loss: {avg_loss:.4f} | LR: {lr_now:.6f} | '
              f'Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m ---')

        # Live plot every 5 epochs
        if (epoch + 1) % 5 == 0:
            plot_live(loss_history, os.path.join(config.output_dir, 'training_progress.png'), epoch + 1)

        # Save checkpoint
        if (epoch + 1) % config.save_interval == 0:
            ckpt_path = os.path.join(config.checkpoint_dir, f'mgf_net_epoch{epoch+1}.pth')
            torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_loss, 'loss_history': loss_history}, ckpt_path)
            print(f'[Checkpoint] Saved: {ckpt_path}')

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
                torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(),
                            'loss': avg_loss}, best_path)
                print(f'[Best] New best model! Loss={best_loss:.4f}')

    # Final
    total_time = time.time() - start_time
    print(f'\n{"="*60}')
    print(f'Training Complete!')
    print(f'Total time: {total_time/60:.1f} min | Best loss: {best_loss:.4f}')
    print(f'Parameters: {n_params:,} | Device: {device}')
    print(f'{"="*60}')

    # Final plots
    plot_live(loss_history, os.path.join(config.output_dir, 'training_progress.png'), config.epochs)
    np.savez(os.path.join(config.output_dir, 'loss_history.npz'), **loss_history)
    print(f'Results saved to: {config.output_dir}')


if __name__ == '__main__':
    train()
