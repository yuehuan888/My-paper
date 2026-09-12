"""
Testing script for MGF-Net.

Evaluates trained model on test image pairs and computes quality metrics.

Usage:
    python test.py
"""

import os
import sys
import time
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from models.mgf_net import MGFNet


def compute_ssim(img1, img2, L=1.0):
    """Compute SSIM between two images."""
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    # Simple implementation for numpy arrays
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim


def compute_psnr(img1, img2, max_val=1.0):
    """Compute PSNR between two images."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))


def load_image(path):
    """Load and preprocess image."""
    img = Image.open(path).convert('L')
    img = np.array(img, dtype=np.float32) / 255.0
    return img


def test():
    config.ensure_dirs()

    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = MGFNet(in_channels=config.in_channels, mid_channels=config.mid_channels, learnable_dwt=False)
    model = model.to(device)
    model.eval()

    # Try to load best checkpoint, fall back to latest
    checkpoint_path = os.path.join(config.checkpoint_dir, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        # Find latest epoch checkpoint
        ckpts = [f for f in os.listdir(config.checkpoint_dir) if f.startswith('mgf_net_epoch')]
        if not ckpts:
            print("ERROR: No checkpoint found! Please train the model first.")
            return
        ckpts.sort(key=lambda x: int(x.split('epoch')[1].split('.pth')[0]))
        checkpoint_path = os.path.join(config.checkpoint_dir, ckpts[-1])

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']}, loss: {checkpoint['loss']:.4f}")

    # Load test images
    ct_files = sorted(os.listdir(config.test_ct_dir))
    mri_files = sorted(os.listdir(config.test_mri_dir))

    print(f"\nTesting on {len(ct_files)} image pairs...")
    print(f"{'='*60}")

    results = []
    times = []

    fig, axes = plt.subplots(len(ct_files), 4, figsize=(16, 4 * len(ct_files)))

    for i, (ct_f, mri_f) in enumerate(zip(ct_files, mri_files)):
        ct_path = os.path.join(config.test_ct_dir, ct_f)
        mri_path = os.path.join(config.test_mri_dir, mri_f)

        ct = load_image(ct_path)
        mri = load_image(mri_path)

        # Pad to multiple of 2 for DWT
        H, W = ct.shape
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2

        ct_padded = np.pad(ct, ((0, pad_h), (0, pad_w)), mode='reflect')
        mri_padded = np.pad(mri, ((0, pad_h), (0, pad_w)), mode='reflect')

        # Convert to tensor
        ct_t = torch.from_numpy(ct_padded).float().unsqueeze(0).unsqueeze(0).to(device)
        mri_t = torch.from_numpy(mri_padded).float().unsqueeze(0).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            start_t = time.time()
            fused_t = model(ct_t, mri_t)
            end_t = time.time()

        fused = fused_t.cpu().squeeze().numpy()
        # Remove padding
        fused = fused[:H, :W]
        times.append(end_t - start_t)

        # Compute metrics
        ssim_ct = compute_ssim(fused, ct)
        ssim_mri = compute_ssim(fused, mri)
        psnr_ct = compute_psnr(fused, ct)
        psnr_mri = compute_psnr(fused, mri)

        avg_ssim = (ssim_ct + ssim_mri) / 2
        avg_psnr = (psnr_ct + psnr_mri) / 2

        results.append({
            'name': ct_f,
            'ssim_ct': ssim_ct, 'ssim_mri': ssim_mri, 'ssim_avg': avg_ssim,
            'psnr_ct': psnr_ct, 'psnr_mri': psnr_mri, 'psnr_avg': avg_psnr,
            'time': end_t - start_t
        })

        print(f"  {ct_f}: SSIM={avg_ssim:.4f}, PSNR={avg_psnr:.2f}dB, Time={times[-1]:.3f}s")

        # Save fused image
        fused_path = os.path.join(config.output_dir, f"fused_{ct_f}")
        fused_uint8 = (np.clip(fused, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(fused_uint8).save(fused_path)

        # Plot
        if len(ct_files) == 1:
            axes[0].imshow(ct, cmap='gray')
            axes[0].set_title('CT')
            axes[1].imshow(mri, cmap='gray')
            axes[1].set_title('MRI')
            axes[2].imshow(fused, cmap='gray')
            axes[2].set_title('MGF-Net Fused')
            axes[3].imshow(np.abs(fused - ct), cmap='hot')
            axes[3].set_title('|Fused - CT|')
        else:
            axes[i, 0].imshow(ct, cmap='gray')
            axes[i, 0].set_title(f'CT ({ct_f})')
            axes[i, 1].imshow(mri, cmap='gray')
            axes[i, 1].set_title('MRI')
            axes[i, 2].imshow(fused, cmap='gray')
            axes[i, 2].set_title('MGF-Net')
            axes[i, 3].imshow(np.abs(fused - ct), cmap='hot')
            axes[i, 3].set_title('Residual')

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")

    avg_ssim = np.mean([r['ssim_avg'] for r in results])
    avg_psnr = np.mean([r['psnr_avg'] for r in results])
    avg_time = np.mean(times)

    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average time: {avg_time:.3f}s per image")

    # Save results
    with open(os.path.join(config.output_dir, "test_results.txt"), 'w') as f:
        f.write(f"MGF-Net Test Results\n")
        f.write(f"{'='*60}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Epoch: {checkpoint['epoch']}\n\n")
        f.write(f"{'Name':<12} {'SSIM_CT':>8} {'SSIM_MRI':>8} {'SSIM_Avg':>8} {'PSNR_CT':>8} {'PSNR_MRI':>8} {'PSNR_Avg':>8} {'Time':>8}\n")
        f.write(f"{'-'*80}\n")
        for r in results:
            f.write(f"{r['name']:<12} {r['ssim_ct']:8.4f} {r['ssim_mri']:8.4f} {r['ssim_avg']:8.4f} "
                    f"{r['psnr_ct']:8.2f} {r['psnr_mri']:8.2f} {r['psnr_avg']:8.2f} {r['time']:8.3f}\n")
        f.write(f"{'-'*80}\n")
        f.write(f"{'Average':<12} {np.mean([r['ssim_ct'] for r in results]):8.4f} "
                f"{np.mean([r['ssim_mri'] for r in results]):8.4f} {avg_ssim:8.4f} "
                f"{np.mean([r['psnr_ct'] for r in results]):8.2f} "
                f"{np.mean([r['psnr_mri'] for r in results]):8.2f} {avg_psnr:8.2f} {avg_time:8.3f}\n")

    print(f"\nResults saved to: {os.path.join(config.output_dir, 'test_results.txt')}")

    # Save comparison figure
    fig_path = os.path.join(config.output_dir, "test_comparison.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    print(f"Comparison figure saved to: {fig_path}")

    return results


if __name__ == '__main__':
    test()
