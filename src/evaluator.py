"""Post-training evaluation utilities.

Two evaluation modes are reproduced from the original notebook:

1. :func:`evaluate_by_event` -- breaks the *validation-event* metrics
   (same target domain used during training/checkpoint selection) down by
   individual event, in addition to a global row. Mirrors the notebook's
   event-based validation Excel export.

2. :func:`evaluate_independent_event` -- quantitative pixel-wise **and**
   building-wise (majority-vote per connected building) metrics on a
   held-out, independent disaster event never seen during training
   (Hurricane Dorian in the original notebook). Both a global row and a
   per-patch breakdown are exported, matching the notebook's Excel sheets.

Both modes use the canonical metric set in :mod:`src.metrics` and always
drop background before computing accuracy/precision/recall/F1/kappa/IoU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import building_majority_map, compute_metrics_from_cm, update_confusion_matrix

PathLike = Union[str, Path]

CLASS_NAMES = ("intact", "damage", "destroyed")


def _clean_event_name(event_item) -> str:
    """Normalize an ``event_name`` batch element (str/bytes/tuple) into a plain string."""
    if isinstance(event_item, (list, tuple)):
        event_item = event_item[0]
    if isinstance(event_item, bytes):
        event_item = event_item.decode("utf-8")
    event_item = str(event_item)
    return Path(event_item).name.split("_")[0]


def evaluate_by_event(
    model: torch.nn.Module,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    num_classes: int,
    device: torch.device,
    out_excel_path: PathLike,
    best_epoch: int = -1,
) -> pd.DataFrame:
    """Evaluate ``model`` on ``val_loader``, broken down by event, and export to Excel.

    A ``GLOBAL_ALL_EVENTS`` row (the same as the trainer's final-validation
    export) is appended for cross-checking.
    """
    from .losses import masked_cross_entropy_loss

    model.eval()

    event_cm: Dict[str, torch.Tensor] = {}
    global_cm = torch.zeros(num_classes, num_classes, device=device)
    global_loss_sum, global_loss_steps = 0.0, 0
    event_pixel_count: Dict[str, int] = {}

    with torch.no_grad():
        for weak_img, _strong_img, damage, building, event_name in tqdm(val_loader, desc="Event-based validation"):
            img = weak_img.to(device, non_blocking=True)
            damage = damage.to(device, non_blocking=True)
            building = building.to(device, non_blocking=True)

            logits = model(img, building_mask=building)
            preds = torch.argmax(logits, dim=1)
            valid_mask = building > 0

            loss_val = masked_cross_entropy_loss(logits, damage, valid_mask, class_weights)
            global_loss_sum += loss_val.item()
            global_loss_steps += 1

            for b in range(img.shape[0]):
                ev = _clean_event_name(event_name[b])
                if ev not in event_cm:
                    event_cm[ev] = torch.zeros(num_classes, num_classes, device=device)
                    event_pixel_count[ev] = 0

                cm_b = update_confusion_matrix(
                    preds[b], damage[b], valid_mask[b], num_classes
                ).to(device)
                event_cm[ev] += cm_b
                global_cm += cm_b
                event_pixel_count[ev] += int(valid_mask[b].sum().item())

    rows = []
    for ev in sorted(event_cm.keys()):
        metrics = compute_metrics_from_cm(event_cm[ev], num_classes=num_classes, drop_background=True)
        rows.append(_metrics_to_row({"event": ev, "num_valid_pixels": event_pixel_count[ev]}, metrics, best_epoch))

    global_metrics = compute_metrics_from_cm(global_cm, num_classes=num_classes, drop_background=True)
    rows.append(
        _metrics_to_row(
            {"event": "GLOBAL_ALL_EVENTS", "num_valid_pixels": int(sum(event_pixel_count.values()))},
            global_metrics,
            best_epoch,
        )
    )

    df = pd.DataFrame(rows)
    Path(out_excel_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_excel_path, index=False)
    print(f"Saved event-based validation Excel: {out_excel_path}")
    return df


def _metrics_to_row(prefix: dict, metrics: dict, best_epoch: int) -> dict:
    row = dict(prefix)
    row["best_epoch"] = best_epoch
    row["accuracy"] = metrics["accuracy"]
    row["precision_macro"] = metrics["precision_macro"]
    row["recall_macro"] = metrics["recall_macro"]
    row["f1_macro"] = metrics["f1_macro"]
    row["kappa"] = metrics["kappa"]
    row["miou_3class"] = metrics["miou_3class"]
    row["miou_rare_2class"] = metrics["miou_rare_2class"]
    for name in CLASS_NAMES:
        row[f"iou_{name}"] = metrics[f"iou_{name}"]
        row[f"precision_{name}"] = metrics[f"precision_{name}"]
        row[f"recall_{name}"] = metrics[f"recall_{name}"]
        row[f"f1_{name}"] = metrics[f"f1_{name}"]
    return row


def evaluate_independent_event(
    model: torch.nn.Module,
    test_loader: DataLoader,
    num_classes: int,
    device: torch.device,
    out_excel_path: PathLike,
    event_name: str = "INDEPENDENT_EVENT",
) -> Dict[str, pd.DataFrame]:
    """Evaluate ``model`` on an independent-event test set, both pixel-wise
    and building-wise (majority vote per connected building object).

    ``test_loader`` must yield ``(image, damage_mask, building_mask, event_name, file_path)``
    per sample, e.g. from a dataset built directly over the independent
    event's NPZ directory (see ``scripts/prepare_independent_event.py`` and
    ``scripts/evaluate_independent_event.py``).

    Exports one Excel workbook with sheets: ``global_pixel_metrics``,
    ``global_building_metrics``, ``patch_pixel_metrics``,
    ``patch_building_metrics``, ``cm_pixel``, ``cm_building``.
    """
    model.eval()

    cm_pixel_global = torch.zeros(num_classes, num_classes)
    cm_building_global = torch.zeros(num_classes, num_classes)
    patch_pixel_rows, patch_building_rows = [], []

    with torch.no_grad():
        for imgs, damages, buildings, events, paths in tqdm(test_loader, desc=f"Evaluating {event_name}"):
            imgs = imgs.to(device, non_blocking=True)
            buildings_gpu = buildings.to(device, non_blocking=True)

            logits = model(imgs, building_mask=buildings_gpu)
            preds = torch.argmax(logits, dim=1).cpu()
            preds = preds * (buildings > 0).long()

            for i in range(preds.shape[0]):
                pred_i = preds[i].numpy().astype(np.uint8)
                gt_i = damages[i].numpy().astype(np.uint8)
                bld_i = buildings[i].numpy().astype(np.uint8)
                valid_mask = torch.from_numpy((bld_i > 0).astype(np.uint8))

                fname = Path(paths[i]).name

                # --- pixel-wise ---
                cm_pixel = update_confusion_matrix(
                    torch.from_numpy(pred_i), torch.from_numpy(gt_i), valid_mask, num_classes
                )
                cm_pixel_global += cm_pixel
                metrics_pixel = compute_metrics_from_cm(cm_pixel, num_classes=num_classes, drop_background=True)
                patch_pixel_rows.append(
                    _metrics_to_row(
                        {"file": fname, "event": events[i], "mode": "pixel_wise",
                         "valid_building_pixels": int(valid_mask.sum().item())},
                        metrics_pixel,
                        best_epoch=-1,
                    )
                )

                # --- building-wise (majority vote per connected building) ---
                gt_majority = building_majority_map(gt_i, bld_i, num_classes)
                pred_majority = building_majority_map(pred_i, bld_i, num_classes)
                cm_building = update_confusion_matrix(
                    torch.from_numpy(pred_majority), torch.from_numpy(gt_majority), valid_mask, num_classes
                )
                cm_building_global += cm_building
                metrics_building = compute_metrics_from_cm(cm_building, num_classes=num_classes, drop_background=True)
                patch_building_rows.append(
                    _metrics_to_row(
                        {"file": fname, "event": events[i], "mode": "building_wise_majority",
                         "valid_building_pixels": int(valid_mask.sum().item())},
                        metrics_building,
                        best_epoch=-1,
                    )
                )

    global_pixel_metrics = compute_metrics_from_cm(cm_pixel_global, num_classes=num_classes, drop_background=True)
    global_building_metrics = compute_metrics_from_cm(cm_building_global, num_classes=num_classes, drop_background=True)

    global_pixel_df = pd.DataFrame([_metrics_to_row({"event": event_name, "mode": "pixel_wise_global"}, global_pixel_metrics, -1)])
    global_building_df = pd.DataFrame(
        [_metrics_to_row({"event": event_name, "mode": "building_wise_majority_global"}, global_building_metrics, -1)]
    )
    patch_pixel_df = pd.DataFrame(patch_pixel_rows)
    patch_building_df = pd.DataFrame(patch_building_rows)

    class_names_with_bg = ("background",) + CLASS_NAMES
    cm_pixel_df = pd.DataFrame(
        cm_pixel_global.numpy(),
        index=[f"true_{c}" for c in class_names_with_bg],
        columns=[f"pred_{c}" for c in class_names_with_bg],
    )
    cm_building_df = pd.DataFrame(
        cm_building_global.numpy(),
        index=[f"true_{c}" for c in class_names_with_bg],
        columns=[f"pred_{c}" for c in class_names_with_bg],
    )

    out_excel_path = Path(out_excel_path)
    out_excel_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_excel_path, engine="openpyxl") as writer:
        global_pixel_df.to_excel(writer, sheet_name="global_pixel_metrics", index=False)
        global_building_df.to_excel(writer, sheet_name="global_building_metrics", index=False)
        patch_pixel_df.to_excel(writer, sheet_name="patch_pixel_metrics", index=False)
        patch_building_df.to_excel(writer, sheet_name="patch_building_metrics", index=False)
        cm_pixel_df.to_excel(writer, sheet_name="cm_pixel")
        cm_building_df.to_excel(writer, sheet_name="cm_building")

    print(f"Saved independent-event Excel: {out_excel_path}")

    return {
        "global_pixel": global_pixel_df,
        "global_building": global_building_df,
        "patch_pixel": patch_pixel_df,
        "patch_building": patch_building_df,
    }
