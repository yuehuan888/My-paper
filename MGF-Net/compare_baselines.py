"""
Compare MGF-Net v2.1 with EMFusion and simple baselines.

Methods:
1. Simple Average: (CT + MRI) / 2
2. Weighted Average: 0.6*CT + 0.4*MRI
3. Laplacian Pyramid Fusion (multi-scale)
4. EMFusion (IF 2021, pretrained model)
5. MGF-Net v2.1 (our method)
"""

import os, sys, time, warnings, numpy as np
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torch, torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.mgf_net import MGFNet


def load_image(path):
    return np.array(Image.open(path).convert('L'), dtype=np.float32) / 255.0


def compute_ssim(a, b):
    """SSIM between two numpy arrays."""
    C1, C2 = 0.0001, 0.0009
    mu_a, mu_b = np.mean(a), np.mean(b)
    va, vb = np.var(a), np.var(b)
    cov = np.mean((a - mu_a) * (b - mu_b))
    return ((2*mu_a*mu_b+C1)*(2*cov+C2)) / ((mu_a**2+mu_b**2+C1)*(va+vb+C2))


def compute_psnr(a, b, max_val=1.0):
    mse = np.mean((a - b)**2)
    return 20*np.log10(max_val/np.sqrt(mse)) if mse > 0 else float('inf')


def laplacian_pyramid_fusion(ct, mri, levels=4):
    """Simple Laplacian pyramid fusion: average low-freq, max high-freq."""
    ct_pyr, mri_pyr = [], []
    ct_cur, mri_cur = ct.copy(), mri.copy()
    shapes = [(ct_cur.shape[0], ct_cur.shape[1])]

    for _ in range(levels):
        h, w = ct_cur.shape
        nh, nw = h//2, w//2
        ct_blur = np.array(Image.fromarray((ct_cur*255).astype(np.uint8)).resize((nw, nh), Image.BILINEAR)) / 255.0
        ct_up = np.array(Image.fromarray((ct_blur*255).astype(np.uint8)).resize((w, h), Image.BILINEAR)) / 255.0
        ct_high = ct_cur - ct_up
        mri_blur = np.array(Image.fromarray((mri_cur*255).astype(np.uint8)).resize((nw, nh), Image.BILINEAR)) / 255.0
        mri_up = np.array(Image.fromarray((mri_blur*255).astype(np.uint8)).resize((w, h), Image.BILINEAR)) / 255.0
        mri_high = mri_cur - mri_up
        ct_pyr.append(ct_high); mri_pyr.append(mri_high)
        shapes.append((nh, nw))
        ct_cur, mri_cur = ct_blur, mri_blur

    # Reconstruct: fuse base, then add back details
    fused = (ct_cur + mri_cur) / 2
    for h_ct, h_mri, (th, tw) in zip(reversed(ct_pyr), reversed(mri_pyr), reversed(shapes[:-1])):
        fused = np.array(Image.fromarray((fused*255).astype(np.uint8)).resize((tw, th), Image.BILINEAR)) / 255.0
        h_fused = np.maximum(np.abs(h_ct), np.abs(h_mri)) * np.sign(h_ct + h_mri)
        fused = fused + h_fused

    return np.clip(fused, 0, 1)


def run_emfusion(ct, mri, sess, ct_ph, mri_ph, output_tensor):
    """Run EMFusion inference using rebuilt TF1 graph."""
    fused = sess.run(output_tensor, {ct_ph: ct[None,:,:,None], mri_ph: mri[None,:,:,None]})
    return np.clip(fused[0,:,:,0], 0, 1)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths
    test_ct_dir = 'data/test/ct'
    test_mri_dir = 'data/test/mri'
    train_ct_dir = 'data/train/ct'
    train_mri_dir = 'data/train/mri'

    # Collect all image pairs
    all_pairs = []
    for d_ct, d_mri, subset in [(test_ct_dir, test_mri_dir, 'test'),
                                 (train_ct_dir, train_mri_dir, 'train')]:
        files = sorted([f for f in os.listdir(d_ct) if f.endswith('.png')])
        for f in files:
            all_pairs.append((os.path.join(d_ct, f), os.path.join(d_mri, f), subset, f))

    print(f'Total image pairs: {len(all_pairs)}')

    # ===== Load MGF-Net v2.1 =====
    mgf = MGFNet(in_channels=1, mid_channels=32, learnable_dwt=False).to(device)
    ckpt = torch.load('checkpoints/best_model.pth', map_location=device, weights_only=False)
    mgf.load_state_dict(ckpt['model_state_dict'])
    mgf.eval()
    print('MGF-Net v2.1 loaded')

    # ===== Load EMFusion =====
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()

    # Import EMFusion's FNet class (TF1 code, need compat aliases)
    emf_code_dir = r'D:/图像相关论文/EMFusion/CT-MRI_code'
    sys.path.insert(0, emf_code_dir)
    # Monkey-patch TF1 APIs that fnet.py uses
    tf.variable_scope = tf.compat.v1.variable_scope
    tf.truncated_normal = tf.compat.v1.truncated_normal
    _orig_conv2d = tf.nn.conv2d
    def _patched_conv2d(input, filter, strides, padding, **kw):
        return _orig_conv2d(input=input, filters=filter, strides=strides, padding=padding, **kw)
    tf.nn.conv2d = _patched_conv2d
    from fnet import FNet

    emf_ckpt = r'D:/图像相关论文/EMFusion/CT-MRI_code/models/3200/3200.ckpt'

    emf_graph = tf.Graph()
    emf_sess = tf.compat.v1.Session(graph=emf_graph)
    emf_ct = emf_mri = emf_output = None

    with emf_graph.as_default():
        # Rebuild graph with variable-size input (like the original test.py)
        emf_ct = tf.compat.v1.placeholder(tf.float32, shape=(1, None, None, 1), name='CT')
        emf_mri = tf.compat.v1.placeholder(tf.float32, shape=(1, None, None, 1), name='MRI')
        fnet = FNet('fnet')
        emf_output = fnet.transform(CT=emf_ct, MRI=emf_mri)

        # Restore only trainable variables (not the full metagraph)
        t_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='fnet')
        saver = tf.compat.v1.train.Saver(var_list=t_vars)
        saver.restore(emf_sess, emf_ckpt)
        print(f'EMFusion loaded with {len(t_vars)} variables')

    # ===== Run comparison =====
    methods = {}
    results = {name: [] for name in ['Average', 'WeightedAvg', 'LaplacianPyr', 'EMFusion', 'MGF-Net v2.1']}

    for ct_path, mri_path, subset, fname in all_pairs:
        ct = load_image(ct_path)
        mri = load_image(mri_path)
        H, W = ct.shape

        # --- Simple Average ---
        avg = (ct + mri) / 2
        results['Average'].append({
            'name': fname, 'subset': subset,
            'fused': avg, 'ssim_ct': compute_ssim(avg, ct), 'ssim_mri': compute_ssim(avg, mri),
            'psnr_ct': compute_psnr(avg, ct), 'psnr_mri': compute_psnr(avg, mri)
        })

        # --- Weighted Average (CT-biased) ---
        wavg = 0.6*ct + 0.4*mri
        results['WeightedAvg'].append({
            'name': fname, 'subset': subset,
            'fused': wavg, 'ssim_ct': compute_ssim(wavg, ct), 'ssim_mri': compute_ssim(wavg, mri),
            'psnr_ct': compute_psnr(wavg, ct), 'psnr_mri': compute_psnr(wavg, mri)
        })

        # --- Laplacian Pyramid ---
        lap = laplacian_pyramid_fusion(ct, mri)
        results['LaplacianPyr'].append({
            'name': fname, 'subset': subset,
            'fused': lap, 'ssim_ct': compute_ssim(lap, ct), 'ssim_mri': compute_ssim(lap, mri),
            'psnr_ct': compute_psnr(lap, ct), 'psnr_mri': compute_psnr(lap, mri)
        })

        # --- EMFusion ---
        emf = run_emfusion(ct, mri, emf_sess, emf_ct, emf_mri, emf_output)
        results['EMFusion'].append({
            'name': fname, 'subset': subset,
            'fused': emf, 'ssim_ct': compute_ssim(emf, ct), 'ssim_mri': compute_ssim(emf, mri),
            'psnr_ct': compute_psnr(emf, ct), 'psnr_mri': compute_psnr(emf, mri)
        })

        # --- MGF-Net v2.1 ---
        ph = (4 - H%4)%4; pw = (4 - W%4)%4
        ct_pad = np.pad(ct, ((0,ph),(0,pw)), mode='reflect') if ph>0 or pw>0 else ct
        mri_pad = np.pad(mri, ((0,ph),(0,pw)), mode='reflect') if ph>0 or pw>0 else mri
        ct_t = torch.from_numpy(ct_pad).float().unsqueeze(0).unsqueeze(0).to(device)
        mri_t = torch.from_numpy(mri_pad).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            mgf_out = mgf(ct_t, mri_t).cpu().squeeze().numpy()[:H,:W]
        results['MGF-Net v2.1'].append({
            'name': fname, 'subset': subset,
            'fused': mgf_out, 'ssim_ct': compute_ssim(mgf_out, ct), 'ssim_mri': compute_ssim(mgf_out, mri),
            'psnr_ct': compute_psnr(mgf_out, ct), 'psnr_mri': compute_psnr(mgf_out, mri)
        })

        print(f'  [{fname}] done')

    emf_sess.close()

    # ===== Print Results =====
    print(f'\n{"="*90}')
    print(f'METHOD COMPARISON (10 CT-MRI pairs)')
    print(f'{"="*90}')
    print(f'{"Method":<20} {"SSIM_CT":>8} {"SSIM_MRI":>8} {"SSIM_Avg":>8} {"PSNR_CT":>8} {"PSNR_MRI":>8} {"PSNR_Avg":>8} {"CT/MRI比":>8}')
    print(f'{"-"*90}')

    summary = {}
    for method_name in ['Average', 'WeightedAvg', 'LaplacianPyr', 'EMFusion', 'MGF-Net v2.1']:
        r = results[method_name]
        ssim_ct = np.mean([x['ssim_ct'] for x in r])
        ssim_mri = np.mean([x['ssim_mri'] for x in r])
        ssim_avg = (ssim_ct + ssim_mri) / 2
        psnr_ct = np.mean([x['psnr_ct'] for x in r])
        psnr_mri = np.mean([x['psnr_mri'] for x in r])
        psnr_avg = (psnr_ct + psnr_mri) / 2
        ratio = ssim_ct / ssim_mri if ssim_mri > 0 else float('inf')

        summary[method_name] = {
            'ssim_ct': ssim_ct, 'ssim_mri': ssim_mri, 'ssim_avg': ssim_avg,
            'psnr_ct': psnr_ct, 'psnr_mri': psnr_mri, 'psnr_avg': psnr_avg,
            'ratio': ratio
        }

        # Highlight our method
        suffix = '  <-- OUR METHOD' if method_name == 'MGF-Net v2.1' else ''
        print(f'{method_name:<20} {ssim_ct:8.4f} {ssim_mri:8.4f} {ssim_avg:8.4f} {psnr_ct:8.2f} {psnr_mri:8.2f} {psnr_avg:8.2f} {ratio:8.3f}{suffix}')

    print(f'{"="*90}')

    # ===== Generate Comparison Figure =====
    # Pick 3 representative test images
    fig_pairs = [p for p in all_pairs if p[3] in ['5.png', '10.png', '6.png']]
    if len(fig_pairs) < 3:
        fig_pairs = all_pairs[:3]

    n_methods = 5
    fig, axes = plt.subplots(len(fig_pairs), n_methods + 2, figsize=(18, 5*len(fig_pairs)))
    method_names = ['Average', 'WeightedAvg', 'LaplacianPyr', 'EMFusion', 'MGF-Net v2.1']

    col_labels = ['CT', 'MRI'] + method_names
    for j, label in enumerate(col_labels):
        axes[0, j].set_title(label, fontsize=11, fontweight='bold')

    for i, (ct_path, mri_path, subset, fname) in enumerate(fig_pairs):
        ct = load_image(ct_path)
        mri = load_image(mri_path)

        axes[i, 0].imshow(ct, cmap='gray')
        axes[i, 1].imshow(mri, cmap='gray')

        for j, method_name in enumerate(method_names):
            r = [x for x in results[method_name] if x['name'] == fname][0]
            axes[i, j+2].imshow(r['fused'], cmap='gray')
            s = (r['ssim_ct'] + r['ssim_mri']) / 2
            axes[i, j+2].set_xlabel(f'SSIM={s:.3f}', fontsize=8)

        for j in range(n_methods + 2):
            axes[i, j].axis('off')
        axes[i, 0].set_ylabel(fname, fontsize=10, fontweight='bold')

    plt.tight_layout()
    fig_path = 'outputs/figures/method_comparison.png'
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'\nComparison figure saved: {fig_path}')

    # ===== Bar Chart =====
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    # SSIM bar chart
    names = list(summary.keys())
    x = np.arange(len(names))
    w = 0.3
    axes2[0].bar(x - w/2, [summary[n]['ssim_ct'] for n in names], w, label='SSIM_CT', color='#4472C4')
    axes2[0].bar(x + w/2, [summary[n]['ssim_mri'] for n in names], w, label='SSIM_MRI', color='#ED7D31')
    axes2[0].set_xticks(x)
    axes2[0].set_xticklabels(names, rotation=15, ha='right', fontsize=8)
    axes2[0].set_ylabel('SSIM')
    axes2[0].set_title('SSIM: Structure Preservation')
    axes2[0].legend()
    axes2[0].grid(axis='y', alpha=0.3)

    # Balance ratio
    ratios = [summary[n]['ratio'] for n in names]
    colors = ['#2ca02c' if r < 1.5 else '#d62728' for r in ratios]
    axes2[1].bar(names, ratios, color=colors)
    axes2[1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Perfect balance')
    axes2[1].set_xticklabels(names, rotation=15, ha='right', fontsize=8)
    axes2[1].set_ylabel('CT/MRI Balance Ratio')
    axes2[1].set_title('Modality Balance (1.0 = perfect)')
    axes2[1].legend()
    axes2[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig2_path = 'outputs/figures/method_metrics_chart.png'
    plt.savefig(fig2_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Metrics chart saved: {fig2_path}')

    # Save summary to text
    with open('outputs/figures/method_comparison.txt', 'w') as f:
        f.write(f'METHOD COMPARISON (10 CT-MRI pairs)\n')
        f.write(f'{"="*90}\n')
        f.write(f'{"Method":<20} {"SSIM_CT":>8} {"SSIM_MRI":>8} {"SSIM_Avg":>8} {"PSNR_CT":>8} {"PSNR_MRI":>8} {"PSNR_Avg":>8} {"CT/MRI":>8}\n')
        f.write(f'{"-"*90}\n')
        for method_name in names:
            s = summary[method_name]
            f.write(f'{method_name:<20} {s["ssim_ct"]:8.4f} {s["ssim_mri"]:8.4f} {s["ssim_avg"]:8.4f} {s["psnr_ct"]:8.2f} {s["psnr_mri"]:8.2f} {s["psnr_avg"]:8.2f} {s["ratio"]:8.3f}\n')
        f.write(f'{"="*90}\n')
        # Best per metric
        best_ssim = max(summary, key=lambda n: summary[n]['ssim_avg'])
        best_bal = min(summary, key=lambda n: abs(summary[n]['ratio'] - 1.0))
        f.write(f'Best SSIM_Avg: {best_ssim} ({summary[best_ssim]["ssim_avg"]:.4f})\n')
        f.write(f'Best Balance: {best_bal} ({summary[best_bal]["ratio"]:.3f})\n')

    print(f'Summary saved: outputs/figures/method_comparison.txt')
    return summary


if __name__ == '__main__':
    main()
