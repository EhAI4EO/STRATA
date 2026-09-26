"""Evaluation metrics.

Background pixels are always excluded before computing accuracy,
precision/recall/F1, Cohen's kappa, and IoU. This unifies two
near-duplicate implementations found in the original notebook
(``compute_metrics_from_cm`` used during training/validation, and a
building-independent copy used for the independent-event test) into a
single canonical function; both computed identical formulas, so this is a
pure de-duplication with no change in reported numbers.

Reported metrics, matching the manuscript:
    accuracy, precision (macro), recall (macro), F1 (macro), Cohen's kappa,
    per-class IoU for {Intact, Damaged, Destroyed}, 3-class mIoU, and the
    rare 2-class mIoU (mean of the Damaged and Destroyed IoUs).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

CLASS_NAMES = ("intact", "damage", "destroyed")  # foreground classes, in label order 1,2,3


def update_confusion_matrix(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    num_classes: int = 4,
) -> torch.Tensor:
    """Accumulate a ``[num_classes, num_classes]`` confusion matrix for one batch.

    Args:
        pred: predicted class indices, any shape.
        target: ground-truth class indices, same shape as ``pred``.
        valid_mask: pixels to include (e.g. inside the building footprint).
        num_classes: total number of classes including background.

    Returns:
        A ``[num_classes, num_classes]`` float tensor (rows = ground truth,
        columns = prediction) for this batch only -- the caller accumulates
        it across batches.
    """
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    valid_mask = valid_mask.reshape(-1).bool()

    valid = valid_mask & (target >= 0) & (target < num_classes) & (pred >= 0) & (pred < num_classes)
    pred = pred[valid].long()
    target = target[valid].long()

    if target.numel() == 0:
        return torch.zeros(num_classes, num_classes, dtype=torch.float32)

    cm = torch.bincount(
        num_classes * target + pred, minlength=num_classes ** 2
    ).reshape(num_classes, num_classes).float()

    return cm


def compute_metrics_from_cm(
    cm: torch.Tensor,
    num_classes: int = 4,
    drop_background: bool = True,
) -> Dict[str, object]:
    """Compute the canonical STRATA metric set from an accumulated confusion matrix.

    Args:
        cm: ``[num_classes, num_classes]`` accumulated confusion matrix
            (rows = ground truth, columns = prediction).
        num_classes: total number of classes including background.
        drop_background: if True (default, matches the reported results),
            class 0 (background) is excluded before computing every metric
            below, including accuracy and kappa.

    Returns:
        A dict with keys: ``accuracy``, ``precision_macro``, ``recall_macro``,
        ``f1_macro``, ``kappa``, ``miou_3class``, ``miou_rare_2class``, plus
        per-class ``iou_<name>``, ``precision_<name>``, ``recall_<name>``,
        ``f1_<name>`` for each of ``("intact", "damage", "destroyed")``.
    """
    cm = cm.float()
    cm_eval = cm[1:, 1:] if drop_background else cm

    tp = cm_eval.diag()
    fp = cm_eval.sum(0) - tp
    fn = cm_eval.sum(1) - tp

    precision_pc = tp / (tp + fp + 1e-6)
    recall_pc = tp / (tp + fn + 1e-6)
    f1_pc = 2 * precision_pc * recall_pc / (precision_pc + recall_pc + 1e-6)
    iou_pc = tp / (tp + fp + fn + 1e-6)

    total = cm_eval.sum()
    accuracy = tp.sum() / (total + 1e-6)

    pe = (cm_eval.sum(0) * cm_eval.sum(1)).sum() / (total ** 2 + 1e-6)
    kappa = (accuracy - pe) / (1 - pe + 1e-6)

    out: Dict[str, object] = {
        "accuracy": accuracy.item(),
        "precision_macro": precision_pc.mean().item(),
        "recall_macro": recall_pc.mean().item(),
        "f1_macro": f1_pc.mean().item(),
        "kappa": kappa.item(),
        "miou_3class": iou_pc.mean().item(),
        "miou_rare_2class": ((iou_pc[1] + iou_pc[2]) / 2.0).item(),
        "iou_per_class": iou_pc.detach().cpu().numpy(),
    }

    for i, name in enumerate(CLASS_NAMES):
        out[f"iou_{name}"] = iou_pc[i].item()
        out[f"precision_{name}"] = precision_pc[i].item()
        out[f"recall_{name}"] = recall_pc[i].item()
        out[f"f1_{name}"] = f1_pc[i].item()

    return out


def building_majority_map(label_map: np.ndarray, building_mask: np.ndarray, num_classes: int = 4) -> np.ndarray:
    """Collapse a per-pixel label map to one majority class per connected building.

    Used for the "building-wise" evaluation mode: each 8-connected building
    object is assigned the majority foreground class among its pixels
    (ties broken arbitrarily by ``np.bincount().argmax()``, matching the
    original notebook).

    Args:
        label_map: ``[H, W]`` per-pixel class indices (prediction or ground truth).
        building_mask: ``[H, W]`` binary building footprint.
        num_classes: total number of classes including background.

    Returns:
        ``[H, W]`` array with one class value per connected building object.
    """
    from scipy import ndimage

    building_bin = building_mask > 0
    labeled, n = ndimage.label(building_bin)

    out = np.zeros_like(label_map, dtype=np.uint8)
    for i in range(1, n + 1):
        region = labeled == i
        vals = label_map[region]
        vals = vals[vals > 0]
        cls = np.bincount(vals, minlength=num_classes).argmax() if len(vals) else 0
        out[region] = cls

    return out
