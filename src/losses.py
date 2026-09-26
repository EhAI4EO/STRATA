"""Loss functions and building-mask-derived weighting/masking utilities.

Only the loss components actually exercised by the final training loop
(:func:`~src.trainer.train_one_experiment`) are included here:

    supervised_rare_damage_loss =
        1.0 * weighted_masked_cross_entropy_loss
      + 1.2 * FocalTverskyLoss (2x boosted for classes {damaged, destroyed})
      + 0.4 * boundary_loss

applied independently to the source batch (``context_weight=0.25``) and the
target batch (``context_weight=0.5``), and combined in the trainer as
``loss_t + lambda_s(epoch) * loss_s``.

Class weights passed to the cross-entropy terms are uniform
(``[1, 1, 1, 1]``) in the reproduced pipeline -- see the audit notes in the
README. Rebalancing instead happens via the weighted samplers in
:mod:`src.sampling`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building-mask-derived masks
# ---------------------------------------------------------------------------

def make_building_context_mask(building_mask: torch.Tensor, kernel_size: int = 9) -> torch.Tensor:
    """Boolean mask of building pixels dilated by ``kernel_size``."""
    if building_mask.dim() == 3:
        building_mask = building_mask.unsqueeze(1)

    context = F.max_pool2d(
        building_mask.float(), kernel_size=kernel_size, stride=1, padding=kernel_size // 2
    )
    return context.squeeze(1) > 0


def make_train_loss_mask(building_mask: torch.Tensor, kernel_size: int = 9) -> torch.Tensor:
    """Training mask: building pixels plus their near-building context."""
    return make_building_context_mask(building_mask, kernel_size=kernel_size)


def make_train_weight_mask(
    building_mask: torch.Tensor, kernel_size: int = 9, context_weight: float = 0.25
) -> torch.Tensor:
    """Soft per-pixel weight: 1.0 inside buildings, ``context_weight`` in the
    dilated near-building ring, 0.0 elsewhere."""
    if building_mask.dim() == 3:
        building_mask = building_mask.unsqueeze(1)

    b = (building_mask.float() > 0).float()
    dilated = F.max_pool2d(b, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    context = torch.clamp(dilated - b, 0.0, 1.0)

    weight = b + context_weight * context
    return weight.squeeze(1)


# ---------------------------------------------------------------------------
# Cross-entropy variants
# ---------------------------------------------------------------------------

def masked_cross_entropy_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    class_weights: torch.Tensor = None,
) -> torch.Tensor:
    """Cross-entropy averaged only over ``valid_mask`` pixels (hard 0/1 mask).

    Used for validation-loss reporting.
    """
    per_pixel = F.cross_entropy(
        logits, target, weight=class_weights, reduction="none", ignore_index=-1
    )
    valid_mask = valid_mask.float()
    return (per_pixel * valid_mask).sum() / (valid_mask.sum() + 1e-6)


def weighted_masked_cross_entropy_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight_mask: torch.Tensor,
    class_weights: torch.Tensor = None,
) -> torch.Tensor:
    """Cross-entropy averaged over a soft per-pixel ``weight_mask``.

    Used as the CE term inside :func:`supervised_rare_damage_loss`, with
    ``weight_mask`` produced by :func:`make_train_weight_mask`.
    """
    per_pixel = F.cross_entropy(
        logits, target, weight=class_weights, reduction="none", ignore_index=-1
    )
    weight_mask = weight_mask.float()
    return (per_pixel * weight_mask).sum() / (weight_mask.sum() + 1e-6)


# ---------------------------------------------------------------------------
# Boundary-aware loss
# ---------------------------------------------------------------------------

def boundary_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    class_weights: torch.Tensor = None,
) -> torch.Tensor:
    """Cross-entropy with a 2x weight on pixels adjacent to a label boundary.

    Args:
        logits: ``[B, C, H, W]``.
        target: ``[B, H, W]`` integer labels.
        mask: hard 0/1 valid-pixel mask, ``[B, H, W]``.
        class_weights: optional per-class weight tensor, ``[C]``.
    """
    boundary = torch.zeros_like(target).float()
    boundary[:, :-1, :] += (target[:, :-1, :] != target[:, 1:, :]).float()
    boundary[:, :, :-1] += (target[:, :, :-1] != target[:, :, 1:]).float()
    boundary = (boundary > 0).float()

    weight_map = torch.ones_like(target).float() + boundary  # 1 -> normal, 2 -> boundary

    per_pixel = F.cross_entropy(
        logits, target, weight=class_weights, reduction="none", ignore_index=-1
    )

    loss = (per_pixel * weight_map * mask).sum() / ((weight_map * mask).sum() + 1e-6)
    return loss


# ---------------------------------------------------------------------------
# Focal-Tversky loss
# ---------------------------------------------------------------------------

class FocalTverskyLoss(nn.Module):
    """Focal-Tversky loss over foreground classes {1, 2, 3}.

    The per-class loss for the rare classes (2=damaged, 3=destroyed) is
    doubled relative to class 1 (intact).

    Args:
        alpha: weight on false positives.
        beta: weight on false negatives.
        gamma: focal exponent applied to ``(1 - Tversky index)``.
        smooth: numerical stabilizer.
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 1.5, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)

        total_loss = 0.0
        for c in range(1, num_classes):
            p = probs[:, c]
            g = (target == c).float()

            tp = (p * g).sum()
            fp = (p * (1 - g)).sum()
            fn = ((1 - p) * g).sum()

            tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
            loss_c = (1 - tversky) ** self.gamma

            if c in (2, 3):
                loss_c = 2.0 * loss_c

            total_loss += loss_c

        return total_loss / (num_classes - 1)


# ---------------------------------------------------------------------------
# Combined supervised loss
# ---------------------------------------------------------------------------

def supervised_rare_damage_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    building_mask: torch.Tensor,
    class_weights: torch.Tensor,
    focal_tversky_loss: FocalTverskyLoss,
    context_weight: float = 0.5,
):
    """The active STRATA supervised loss.

    ``loss = 1.0 * CE + 1.2 * FocalTversky + 0.4 * boundary``, where CE and
    boundary are computed with building-mask-derived soft/hard masks.

    Args:
        logits: ``[B, C, H, W]`` model predictions.
        target: ``[B, H, W]`` damage labels.
        building_mask: ``[B, H, W]`` building footprint.
        class_weights: ``[C]`` weight tensor (uniform in the reproduced pipeline).
        focal_tversky_loss: a shared :class:`FocalTverskyLoss` instance.
        context_weight: near-building context weight (0.25 for source
            batches, 0.5 for target batches in the trainer).

    Returns:
        Tuple of ``(total_loss, loss_ce, loss_tversky, loss_boundary, train_mask)``.
    """
    train_weight = make_train_weight_mask(building_mask, kernel_size=9, context_weight=context_weight)
    train_mask = make_train_loss_mask(building_mask, kernel_size=9)

    loss_ce = weighted_masked_cross_entropy_loss(logits, target, train_weight, class_weights)
    loss_tv = focal_tversky_loss(logits, target)
    loss_bd = boundary_loss(logits, target, train_mask, class_weights)

    loss = 1.0 * loss_ce + 1.2 * loss_tv + 0.4 * loss_bd

    return loss, loss_ce, loss_tv, loss_bd, train_mask
