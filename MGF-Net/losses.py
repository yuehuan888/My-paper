"""
v2 Loss functions with adaptive modality balancing.

Key change: learnable modality weights w_CT, w_MRI replace fixed loss weights.
The model learns to balance CT and MRI fidelity during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SSIMLoss(nn.Module):
    """1 - SSIM structural loss."""
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-coords**2 / (2*sigma**2)); g /= g.sum()
        w = (g[:,None] * g[None,:]).expand(1, 1, window_size, window_size).contiguous()
        self.register_buffer('window', w)

    def forward(self, img1, img2):
        img1, img2 = img1.float(), img2.float()
        _, C, _, _ = img1.shape
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)
        w = self.window[:, :C]
        mu1 = F.conv2d(img1, w, padding=5, groups=C)
        mu2 = F.conv2d(img2, w, padding=5, groups=C)
        s1 = F.conv2d(img1*img1, w, padding=5, groups=C) - mu1**2
        s2 = F.conv2d(img2*img2, w, padding=5, groups=C) - mu2**2
        s12 = F.conv2d(img1*img2, w, padding=5, groups=C) - mu1*mu2
        C1, C2 = 0.01**2, 0.03**2
        ssim = ((2*mu1*mu2+C1)*(2*s12+C2)) / ((mu1**2+mu2**2+C1)*(s1+s2+C2))
        return 1 - ssim.mean()


class GradientLoss(nn.Module):
    """Edge preservation: fused gradient should match max(source gradients)."""
    def __init__(self):
        super().__init__()
        sx = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]])/4.
        sy = torch.tensor([[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]])/4.
        self.register_buffer('sx', sx.view(1,1,3,3))
        self.register_buffer('sy', sy.view(1,1,3,3))

    def _grad(self, x):
        x = x.float()
        B, C, H, W = x.shape
        xr = x.reshape(B*C, 1, H, W)
        gx = F.conv2d(xr, self.sx, padding=1)
        gy = F.conv2d(xr, self.sy, padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-8).reshape(B, C, H, W)

    def forward(self, fused, ct, mri):
        gf = self._grad(fused)
        gm = torch.max(self._grad(ct), self._grad(mri))
        return F.l1_loss(gf, gm)


class MGFusionLoss(nn.Module):
    """v2.1: Dual-constraint fusion loss with balance regularization.

    Key fix from v2:
    - Removed learnable modality weights (they drifted to extremes)
    - Added L_balance = |L1(fused,CT) - L1(fused,MRI)| to prevent model
      from favoring one modality over the other
    - Fixed w_ct = w_mri = 0.5 for L1 loss

    L = alpha*ssim + beta*l1_balanced + gamma*grad + delta*balance
    """
    def __init__(self, alpha=1.0, beta=10.0, gamma=5.0, delta=2.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta  # balance regularization strength

        self.ssim_loss = SSIMLoss()
        self.grad_loss = GradientLoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, fused, ct, mri):
        # SSIM (averaged symmetrically)
        loss_ssim = (self.ssim_loss(fused, ct) + self.ssim_loss(fused, mri)) / 2

        # L1 with equal weights
        l1_ct = self.l1_loss(fused, ct)
        l1_mri = self.l1_loss(fused, mri)
        loss_l1 = 0.5 * (l1_ct + l1_mri)

        # Balance constraint: penalize asymmetry
        # The model shouldn't just copy one modality
        loss_balance = torch.abs(l1_ct - l1_mri)

        # Gradient loss
        loss_grad = self.grad_loss(fused, ct, mri)

        total = (self.alpha * loss_ssim +
                 self.beta * loss_l1 +
                 self.gamma * loss_grad +
                 self.delta * loss_balance)

        loss_dict = {
            'total': total.item(),
            'ssim': loss_ssim.item(),
            'l1': loss_l1.item(),
            'grad': loss_grad.item(),
            'bal': loss_balance.item(),
            'l1_ct': l1_ct.item(),
            'l1_mri': l1_mri.item(),
        }
        return total, loss_dict
