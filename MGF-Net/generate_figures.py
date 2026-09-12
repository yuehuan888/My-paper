"""
Generate comprehensive figures for the MGF-Net report.
- Run inference on all 10 CT-MRI pairs
- Create comparison figure (CT / MRI / Fused / Residual)
- Save individual fused images
"""

import os, sys, time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import config
from models.mgf_net import MGFNet


def load_image(path):
    img = Image.open(path).convert('L')
    return np.array(img, dtype=np.float32) / 255.0


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load best model
    model = MGFNet(in_channels=1, mid_channels=32, learnable_dwt=False).to(device)
    ckpt = torch.load('checkpoints/best_model.pth', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'Loaded model from epoch {ckpt["epoch"]}, loss={ckpt["loss"]:.4f}')

    # Prepare output dir
    fig_dir = 'outputs/figures'
    fused_dir = 'outputs/fused_images'
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(fused_dir, exist_ok=True)

    # Load and run all 10 images
    ct_dir = 'data/test/ct'
    mri_dir = 'data/test/mri'
    # Also use the train images for completeness
    ct_train = 'data/train/ct'
    mri_train = 'data/train/mri'

    # Collect all image pairs
    all_pairs = []

    # Test set
    test_files = sorted(os.listdir(ct_dir))
    for f in test_files:
        if f.endswith('.png'):
            all_pairs.append((os.path.join(ct_dir, f), os.path.join(mri_dir, f), 'test', f))

    # Train set (for visual comparison, model hasn't memorized them)
    train_files = sorted(os.listdir(ct_train))
    for f in train_files:
        if f.endswith('.png'):
            all_pairs.append((os.path.join(ct_train, f), os.path.join(mri_train, f), 'train', f))

    print(f'Processing {len(all_pairs)} image pairs...')

    results = []
    for ct_path, mri_path, subset, fname in all_pairs:
        ct = load_image(ct_path)
        mri = load_image(mri_path)
        H, W = ct.shape

        # Pad to multiple of 4 for 2-level DWT
        ph = (4 - H % 4) % 4
        pw = (4 - W % 4) % 4
        ct_pad = np.pad(ct, ((0, ph), (0, pw)), mode='reflect') if ph > 0 or pw > 0 else ct
        mri_pad = np.pad(mri, ((0, ph), (0, pw)), mode='reflect') if ph > 0 or pw > 0 else mri

        ct_t = torch.from_numpy(ct_pad).float().unsqueeze(0).unsqueeze(0).to(device)
        mri_t = torch.from_numpy(mri_pad).float().unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            fused_t = model(ct_t, mri_t)

        fused = fused_t.cpu().squeeze().numpy()[:H, :W]

        # Clip and save fused image
        fused_clipped = np.clip(fused, 0, 1)
        fused_uint8 = (fused_clipped * 255).astype(np.uint8)
        out_name = fname.replace('.png', f'_{subset}_fused.png')
        Image.fromarray(fused_uint8).save(os.path.join(fused_dir, out_name))

        # Compute metrics
        def ssim(a, b):
            C1, C2 = 0.0001, 0.0009
            mu_a, mu_b = np.mean(a), np.mean(b)
            va, vb = np.var(a), np.var(b)
            cov = np.mean((a - mu_a) * (b - mu_b))
            return ((2*mu_a*mu_b+C1)*(2*cov+C2)) / ((mu_a**2+mu_b**2+C1)*(va+vb+C2))

        results.append({
            'name': fname, 'subset': subset,
            'ct': ct, 'mri': mri, 'fused': fused_clipped,
            'ssim_ct': ssim(fused_clipped, ct),
            'ssim_mri': ssim(fused_clipped, mri),
            'residual': np.abs(fused_clipped - ct)
        })

    # ===== FIGURE 1: Large comparison grid (all 10 images) =====
    n = len(results)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4.5 * n))
    fig.suptitle('MGF-Net v2.1: CT-MRI Medical Image Fusion Results (10 pairs)', fontsize=16, fontweight='bold', y=0.995)

    col_titles = ['CT (Source)', 'MRI (Source)', 'MGF-Net Fused', 'Residual |Fused-CT|']
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=12, fontweight='bold')

    for i, r in enumerate(results):
        axes[i, 0].imshow(r['ct'], cmap='gray')
        axes[i, 0].set_ylabel(f'{r["name"]}\n({r["subset"]})', fontsize=10)
        axes[i, 1].imshow(r['mri'], cmap='gray')
        axes[i, 2].imshow(r['fused'], cmap='gray')
        axes[i, 2].text(0.02, 0.98, f'SSIM_CT={r["ssim_ct"]:.3f}\nSSIM_MRI={r["ssim_mri"]:.3f}',
                       transform=axes[i, 2].transAxes, fontsize=7, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        im = axes[i, 3].imshow(r['residual'], cmap='hot')
        plt.colorbar(im, ax=axes[i, 3], fraction=0.046)

        for j in range(4):
            axes[i, j].axis('off')

    plt.tight_layout()
    fig_path1 = os.path.join(fig_dir, 'all_results_comparison.png')
    fig.savefig(fig_path1, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {fig_path1}')

    # ===== FIGURE 2: Best 3 results (for the report) =====
    # Pick top 3 by average SSIM
    results_sorted = sorted(results, key=lambda r: (r['ssim_ct'] + r['ssim_mri']) / 2, reverse=True)
    top3 = results_sorted[:3]

    fig2, axes2 = plt.subplots(3, 3, figsize=(12, 10))
    fig2.suptitle('MGF-Net v2.1: Top-3 Fusion Results', fontsize=14, fontweight='bold')

    col_labels = ['CT', 'MRI', 'MGF-Net Fused']
    for j, label in enumerate(col_labels):
        axes2[0, j].set_title(label, fontsize=12, fontweight='bold')

    for i, r in enumerate(top3):
        axes2[i, 0].imshow(r['ct'], cmap='gray')
        axes2[i, 1].imshow(r['mri'], cmap='gray')
        axes2[i, 2].imshow(r['fused'], cmap='gray')
        axes2[i, 2].set_xlabel(f'SSIM_CT={r["ssim_ct"]:.3f}  SSIM_MRI={r["ssim_mri"]:.3f}  Avg={((r["ssim_ct"]+r["ssim_mri"])/2):.3f}',
                              fontsize=10)
        for j in range(3):
            axes2[i, j].axis('off')
        axes2[i, 0].set_ylabel(r['name'], fontsize=10, fontweight='bold')

    plt.tight_layout()
    fig_path2 = os.path.join(fig_dir, 'top3_fusion_results.png')
    fig2.savefig(fig_path2, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {fig_path2}')

    # ===== FIGURE 3: Zoom-in detail comparison =====
    # Show the same ROI for CT, MRI, Fused to highlight detail preservation
    r = results_sorted[0]  # Best result
    ct, mri, fused = r['ct'], r['mri'], r['fused']
    H, W = ct.shape

    # Select a center ROI
    roi_size = 80
    cy, cx = H // 2, W // 2
    y1, y2 = max(0, cy - roi_size // 2), min(H, cy + roi_size // 2)
    x1, x2 = max(0, cx - roi_size // 2), min(W, cx + roi_size // 2)

    fig3, axes3 = plt.subplots(2, 4, figsize=(16, 8))
    fig3.suptitle(f'MGF-Net v2.1: Detail Comparison - {r["name"]}', fontsize=14, fontweight='bold')

    # Row 1: full images with ROI box
    for j, (img, title) in enumerate([(ct, 'CT'), (mri, 'MRI'), (fused, 'Fused')]):
        axes3[0, j].imshow(img, cmap='gray')
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', facecolor='none')
        axes3[0, j].add_patch(rect)
        axes3[0, j].set_title(title, fontsize=11)
        axes3[0, j].axis('off')

    # Color overlay of fused vs CT (where MRI info was added)
    overlay = np.zeros((H, W, 3))
    overlay[:, :, 0] = fused  # red channel = fused
    overlay[:, :, 1] = ct     # green channel = CT
    axes3[0, 3].imshow(overlay)
    axes3[0, 3].set_title('Overlay (R:Fused, G:CT)\nYellow=MRI detail added', fontsize=10)
    axes3[0, 3].axis('off')

    # Row 2: zoomed ROI
    for j, (img, title) in enumerate([(ct, 'CT (zoom)'), (mri, 'MRI (zoom)'), (fused, 'Fused (zoom)')]):
        axes3[1, j].imshow(img[y1:y2, x1:x2], cmap='gray')
        axes3[1, j].set_title(title, fontsize=11)
        axes3[1, j].axis('off')

    # Zoom overlay
    overlay_zoom = np.zeros((roi_size, roi_size, 3))
    overlay_zoom[:, :, 0] = fused[y1:y2, x1:x2]
    overlay_zoom[:, :, 1] = ct[y1:y2, x1:x2]
    axes3[1, 3].imshow(overlay_zoom)
    axes3[1, 3].set_title('Zoom Overlay', fontsize=10)
    axes3[1, 3].axis('off')

    plt.tight_layout()
    fig_path3 = os.path.join(fig_dir, 'detail_comparison.png')
    fig3.savefig(fig_path3, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {fig_path3}')

    # ===== Save metrics summary =====
    with open(os.path.join(fig_dir, 'metrics_summary.txt'), 'w') as f:
        f.write(f'MGF-Net v2.1 - Full Results ({len(results)} images)\n')
        f.write(f'Model: epoch {ckpt["epoch"]}, loss={ckpt["loss"]:.4f}\n')
        f.write(f'{"="*70}\n')
        f.write(f'{"Name":<12} {"Subset":<8} {"SSIM_CT":>8} {"SSIM_MRI":>8} {"SSIM_Avg":>8}\n')
        f.write(f'{"-"*50}\n')
        for r in results:
            avg = (r['ssim_ct'] + r['ssim_mri']) / 2
            f.write(f'{r["name"]:<12} {r["subset"]:<8} {r["ssim_ct"]:8.4f} {r["ssim_mri"]:8.4f} {avg:8.4f}\n')

        all_ct = [r['ssim_ct'] for r in results]
        all_mri = [r['ssim_mri'] for r in results]
        all_avg = [(c+m)/2 for c, m in zip(all_ct, all_mri)]
        f.write(f'{"-"*50}\n')
        f.write(f'{"Overall Avg":<20} {np.mean(all_ct):8.4f} {np.mean(all_mri):8.4f} {np.mean(all_avg):8.4f}\n')
        f.write(f'{"Overall Std":<20} {np.std(all_ct):8.4f} {np.std(all_mri):8.4f} {np.std(all_avg):8.4f}\n')

    print(f'\n=== Summary ===')
    print(f'SSIM_CT:  {np.mean(all_ct):.4f} ± {np.std(all_ct):.4f}')
    print(f'SSIM_MRI: {np.mean(all_mri):.4f} ± {np.std(all_mri):.4f}')
    print(f'SSIM_Avg: {np.mean(all_avg):.4f} ± {np.std(all_avg):.4f}')
    print(f'CT/MRI ratio: {np.mean(all_ct)/np.mean(all_mri):.2f}:1')
    print(f'\nAll outputs saved to: {fig_dir}/')
    print(f'Fused images saved to: {fused_dir}/')


if __name__ == '__main__':
    main()
