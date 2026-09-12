"""
Edge-Aware Refinement (EAR) Module

Refines the fused image by explicitly enhancing edge information.
Uses a lightweight residual block with edge-awareness from
the input modalities' gradient information.

Key innovation: Instead of just post-processing, we compute
edge maps from the source images and use them as attention
to guide the refinement process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeExtractor(nn.Module):
    """Extract edge information using Sobel-like learnable filters."""

    def __init__(self):
        super(EdgeExtractor, self).__init__()
        # Learnable edge detection kernels (initialized as Sobel approximations)
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 4.0
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]) / 4.0

        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def forward(self, x):
        """Extract gradient magnitude as edge map.

        Args:
            x: [B, C, H, W]
        Returns:
            edge_map: [B, C, H, W] gradient magnitude
        """
        B, C, H, W = x.shape
        x_flat = x.reshape(B * C, 1, H, W)

        gx = F.conv2d(x_flat, self.sobel_x, padding=1)
        gy = F.conv2d(x_flat, self.sobel_y, padding=1)

        edge_map = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        edge_map = edge_map.reshape(B, C, H, W)
        return edge_map


class EdgeRefineModule(nn.Module):
    """Edge-Aware Refinement module.

    Takes the initial fused image, computes edge maps from it and
    the source images, and refines to preserve sharp edges.
    """

    def __init__(self, in_channels=1, num_features=32):
        super(EdgeRefineModule, self).__init__()
        self.edge_extractor = EdgeExtractor()

        # Edge-guided refinement network
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, num_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
        )

        self.refine_block = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, fused, ct, mri):
        """
        Args:
            fused: [B, C, H, W] initial fused image
            ct:    [B, C, H, W] CT source
            mri:   [B, C, H, W] MRI source
        Returns:
            refined: [B, C, H, W] edge-refined fused image
        """
        # Extract edge maps
        edge_fused = self.edge_extractor(fused)
        edge_ct = self.edge_extractor(ct)
        edge_mri = self.edge_extractor(mri)

        # Concatenate edge information
        edge_input = torch.cat([edge_fused, edge_ct, edge_mri], dim=1)

        # Edge-guided refinement
        edge_feat = self.edge_conv(edge_input)
        residual = self.refine_block(edge_feat)

        refined = fused + residual
        return torch.clamp(refined, 0, 1)
