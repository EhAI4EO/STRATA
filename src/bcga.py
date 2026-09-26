"""Building-Context-Guided Attention (BCGA).

Gates the deepest encoder feature map using a prior derived from the
building footprint mask: a building-interior channel, a dilated
near-building "context" channel, and a background channel. The prior is
encoded and combined with the features themselves to produce a
multiplicative gate.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BuildingContextGuidedAttention(nn.Module):
    """Building-mask-conditioned feature gating.

    Args:
        in_channels: number of channels of the feature map to be gated.
        context_kernel: size of the square max-pooling kernel used to
            dilate the building mask into a "near-building context" band.
    """

    def __init__(self, in_channels: int = 512, context_kernel: int = 9):
        super().__init__()
        self.context_kernel = context_kernel

        self.prior_encoder = nn.Sequential(
            nn.Conv2d(3, in_channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid(),
        )

        self.feature_gate = nn.Sequential(
            nn.Conv2d(in_channels + 3, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Sigmoid(),
        )

    def make_building_priors(self, building_mask: torch.Tensor, out_size) -> torch.Tensor:
        """Build a 3-channel prior: [building, near-building context, background].

        Args:
            building_mask: ``[B, H, W]`` or ``[B, 1, H, W]`` binary mask.
            out_size: target spatial size ``(h, w)`` matching the feature map.
        """
        if building_mask.dim() == 3:
            building_mask = building_mask.unsqueeze(1)

        b = F.interpolate(building_mask.float(), size=out_size, mode="nearest")
        b = (b > 0).float()

        pad = self.context_kernel // 2
        dilated = F.max_pool2d(b, kernel_size=self.context_kernel, stride=1, padding=pad)

        context = torch.clamp(dilated - b, 0.0, 1.0)
        background = 1.0 - dilated

        return torch.cat([b, context, background], dim=1)

    def forward(self, features: torch.Tensor, building_mask: torch.Tensor) -> torch.Tensor:
        """Gate ``features`` using the building-context prior.

        Args:
            features: ``[B, C, h, w]`` feature map to be modulated.
            building_mask: ``[B, H, W]`` (or ``[B, 1, H, W]``) building footprint.

        Returns:
            The gated feature map, same shape as ``features``.
        """
        prior = self.make_building_priors(building_mask, features.shape[-2:])

        prior_gate = self.prior_encoder(prior)
        feature_gate = self.feature_gate(torch.cat([features, prior], dim=1))

        guided = features * (1.0 + 0.5 * prior_gate * feature_gate)
        return guided
