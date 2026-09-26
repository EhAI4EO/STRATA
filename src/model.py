"""STRATA model: SegFormer-B2 encoder + BCGA + lightweight multi-scale decoder.

This module reproduces the architecture defined in the original research
notebook under the class name ``SAHARASegFormer``. It is renamed here to
``STRATA`` for clarity; no tensor operation, shape, or parameter is changed.

Architecture summary
---------------------
1. A pretrained SegFormer-B2 encoder (``nvidia/segformer-b2-finetuned-ade-512-512``)
   produces four hierarchical feature maps with channel widths
   ``[64, 128, 320, 512]``.
2. Building-Context-Guided Attention (:class:`~src.bcga.BuildingContextGuidedAttention`)
   gates the deepest feature map (512 channels) using a prior derived from
   the building-footprint mask.
3. A lightweight decoder projects all four feature maps to a common width,
   fuses them, applies a small local-context refinement term, and predicts
   per-pixel class logits at the input resolution.

Output classes (harmonized STRATA scheme): 0=Background, 1=Intact,
2=Damaged, 3=Destroyed.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerModel

from .bcga import BuildingContextGuidedAttention

SEGFORMER_B2_CHECKPOINT = "nvidia/segformer-b2-finetuned-ade-512-512"
SEGFORMER_B2_HIDDEN_SIZES = (64, 128, 320, 512)


class SegFormerEncoder(nn.Module):
    """Thin wrapper around a pretrained HuggingFace SegFormer backbone.

    Returns the four hierarchical hidden states (before the SegFormer
    decode head), matching stage widths ``[64, 128, 320, 512]`` for the
    B2 variant.
    """

    def __init__(self, checkpoint: str = SEGFORMER_B2_CHECKPOINT):
        super().__init__()
        self.backbone = SegformerModel.from_pretrained(checkpoint)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.backbone(pixel_values=x, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        f1, f2, f3, f4 = hidden_states[0], hidden_states[1], hidden_states[2], hidden_states[3]
        return f1, f2, f3, f4


class LightweightDamageDecoder(nn.Module):
    """Multi-scale feature fusion decoder with a light context-refinement term.

    Projects each encoder stage to ``proj_channels`` channels, upsamples all
    stages to the resolution of the shallowest stage, concatenates, fuses
    with a small conv stack, and classifies.
    """

    def __init__(
        self,
        in_channels: Sequence[int] = SEGFORMER_B2_HIDDEN_SIZES,
        num_classes: int = 4,
        proj_channels: int = 128,
        fusion_channels: int = 256,
    ):
        super().__init__()

        self.proj1 = nn.Conv2d(in_channels[0], proj_channels, 1)
        self.proj2 = nn.Conv2d(in_channels[1], proj_channels, 1)
        self.proj3 = nn.Conv2d(in_channels[2], proj_channels, 1)
        self.proj4 = nn.Conv2d(in_channels[3], proj_channels, 1)

        self.fusion = nn.Sequential(
            nn.Conv2d(proj_channels * 4, fusion_channels, 1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_channels, proj_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(proj_channels),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Conv2d(proj_channels, num_classes, 1)

    def forward(
        self,
        f1: torch.Tensor,
        f2: torch.Tensor,
        f3: torch.Tensor,
        f4: torch.Tensor,
        out_size: Union[Tuple[int, int], torch.Size],
    ) -> torch.Tensor:
        f1 = self.proj1(f1)

        f2 = self.proj2(f2)
        f2 = F.interpolate(f2, size=f1.shape[-2:], mode="bilinear", align_corners=False)

        f3 = self.proj3(f3)
        f3 = F.interpolate(f3, size=f1.shape[-2:], mode="bilinear", align_corners=False)

        f4 = self.proj4(f4)
        f4 = F.interpolate(f4, size=f1.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([f1, f2, f3, f4], dim=1)
        x = self.fusion(x)

        # Local-context refinement: a small residual from a 3x3 average pool.
        context_feat = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        x = x + 0.3 * context_feat

        logits = self.classifier(x)
        logits = F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
        return logits


class STRATA(nn.Module):
    """Structure-guided Transfer Architecture for building-damage segmentation.

    Named ``SAHARASegFormer`` in the original research notebook.

    Args:
        num_classes: number of output classes (4: background/intact/damaged/destroyed).
        segformer_checkpoint: HuggingFace checkpoint id for the SegFormer-B2 encoder.
    """

    def __init__(self, num_classes: int = 4, segformer_checkpoint: str = SEGFORMER_B2_CHECKPOINT):
        super().__init__()

        self.encoder = SegFormerEncoder(checkpoint=segformer_checkpoint)

        self.building_attention = BuildingContextGuidedAttention(
            in_channels=SEGFORMER_B2_HIDDEN_SIZES[-1],
            context_kernel=9,
        )

        self.decoder = LightweightDamageDecoder(
            in_channels=SEGFORMER_B2_HIDDEN_SIZES,
            num_classes=num_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
        building_mask: torch.Tensor = None,
        return_features: bool = False,
    ):
        """Run a forward pass.

        Args:
            x: input image batch, ``[B, 3, H, W]``.
            building_mask: optional building-footprint mask, ``[B, H, W]``.
                When provided, BCGA gates the deepest feature map.
            return_features: if True, also return the (possibly gated)
                deepest feature map -- used by the trainer for auxiliary
                feature-based losses.

        Returns:
            ``logits`` of shape ``[B, num_classes, H, W]``, or
            ``(logits, f4)`` if ``return_features`` is True.
        """
        out_size = x.shape[-2:]

        f1, f2, f3, f4 = self.encoder(x)

        if building_mask is not None:
            f4 = self.building_attention(f4, building_mask)

        logits = self.decoder(f1, f2, f3, f4, out_size)

        if return_features:
            return logits, f4
        return logits


if __name__ == "__main__":
    # Lightweight shape smoke test (matches the check in the original notebook).
    model = STRATA(num_classes=4)
    dummy_x = torch.randn(2, 3, 256, 256)
    dummy_building = torch.randint(0, 2, (2, 256, 256)).float()
    dummy_logits = model(dummy_x, dummy_building)
    print(dummy_logits.shape)  # expected: torch.Size([2, 4, 256, 256])
