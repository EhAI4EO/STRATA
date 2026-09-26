"""The active STRATA training loop.

Reproduces ``train_and_eval`` from the original notebook. One branch of
the original loop -- reached when ``epoch < WARM_UP_EPOCHS`` -- called an
undefined ``model.forward_source`` method and an undefined
``gated_damage_alignment`` function, and referenced variables (``img_s``,
``img_t``, ``loss_sep``) that were never assigned in scope. All reported
runs set ``WARM_UP_EPOCHS = 0``, so that branch never executed; it has
been removed here rather than reconstructed, per the audit notes in the
README ("do not invent missing implementation").

Training objective actually executed, per batch:
    logits_s, feat_s = model(strong_img_s, building_mask=bld_s, return_features=True)
    logits_t, feat_t = model(strong_img_t, building_mask=bld_t, return_features=True)
    loss_t = supervised_rare_damage_loss(logits_t, dmg_t, bld_t, ..., context_weight=0.5)
    loss_s = supervised_rare_damage_loss(logits_s, dmg_s, bld_s, ..., context_weight=0.25)
    loss   = 1.0 * loss_t + lambda_s(epoch) * loss_s

The cross-event rare-class prototype-transfer term present in the
original notebook (``prototype_memory`` / ``event_selective_rare_proto_loss``)
is hardcoded to zero there (annotated ``"A2: w/o Prototype Memory"``) and
is therefore omitted from this reproduction; see the README audit notes.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from .losses import FocalTverskyLoss, masked_cross_entropy_loss, supervised_rare_damage_loss
from .metrics import compute_metrics_from_cm, update_confusion_matrix
from .model import STRATA
from .sampling import build_dataloaders
from .utils import get_device, save_checkpoint, set_seed

PathLike = Union[str, Path]

CLASS_NAMES = ("intact", "damage", "destroyed")


@dataclass
class TrainConfig:
    """All hyperparameters exposed by the original notebook's config cell.

    Values are the defaults used for the ``n=100`` label-budget run
    reported in the notebook; override via YAML (see ``configs/``).
    """

    num_classes: int = 4
    epochs: int = 100
    eval_every: int = 3
    train_batch_size: int = 128
    target_labels_per_class: int = 100
    warm_up_epochs: int = 0  # kept for parity; must stay 0 (see module docstring)
    num_workers: int = 12
    early_stop_patience: int = 5
    learning_rate: float = 1e-4
    accum_steps: int = 2
    seed: int = 42
    gpu: str = "0"

    source_root: str = ""
    target_root: str = ""
    out_dir: str = ""
    splits_dir: str = "splits"
    target_events: list = field(
        default_factory=lambda: [
            "EARTHQUAKE-TURKEY",
            "TEXAS-TORNADOES",
            "HURRICANE-DELTA",
            "HURRICANE-IDA",
        ]
    )

    @property
    def event_split_path(self) -> Path:
        return Path(self.splits_dir) / "ebd_optical_event_split_target.npz"

    @property
    def fixed_split_path(self) -> Path:
        return Path(self.splits_dir) / "ebd_fixed_train_val_split.npz"

    @property
    def experiment_split_path(self) -> Path:
        return Path(self.splits_dir) / f"ebd_labeled_unlabeled_{self.target_labels_per_class}.npz"


def get_source_transfer_weight(epoch: int) -> float:
    """Epoch-dependent weight on the (low-label) source-domain loss term."""
    if epoch < 8:
        return 0.0
    if epoch < 20:
        return 0.05
    return 0.10


def train_one_experiment(config: TrainConfig) -> Optional[Dict[str, Any]]:
    """Run one full STRATA training + final-validation experiment.

    Args:
        config: a populated :class:`TrainConfig`.

    Returns:
        The metrics dict for the best checkpoint (by rare 2-class mIoU), or
        ``None`` if no checkpoint was ever saved.
    """
    set_seed(config.seed)
    device = get_device(config.gpu)

    out_dir = Path(config.out_dir)
    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    Path(config.splits_dir).mkdir(parents=True, exist_ok=True)

    from .sampling import create_event_based_split

    if not config.event_split_path.exists():
        create_event_based_split(
            root_dir=config.source_root,
            save_path=config.event_split_path,
            target_events=set(config.target_events),
        )

    source_loader, target_labeled_loader, _, target_val_loader = build_dataloaders(
        source_root=config.source_root,
        target_root=config.target_root,
        batch_size=config.train_batch_size,
        event_split_path=config.event_split_path,
        fixed_split_path=config.fixed_split_path,
        experiment_split_path=config.experiment_split_path,
        target_labels_per_class=config.target_labels_per_class,
        num_workers=config.num_workers,
    )

    model = STRATA(num_classes=config.num_classes).to(device)

    class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32, device=device)
    print("Class weights (uniform in the reproduced pipeline):", class_weights.cpu().numpy())

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-6
    )
    scaler = GradScaler(device=str(device))
    focal_tversky_loss = FocalTverskyLoss()

    best_rare_miou = -1.0
    best_metrics: Optional[Dict[str, Any]] = None
    best_epoch = -1
    early_stop_counter = 0

    train_loss_hist, val_loss_hist, miou_hist, val_metrics_hist = [], [], [], []

    target_labeled_iter = iter(target_labeled_loader)

    for epoch in range(config.epochs):
        model.train()
        total_loss_epoch = 0.0
        n_steps = 0
        train_correct = 0
        train_total = 0

        pbar = tqdm(source_loader, desc=f"Epoch {epoch + 1}/{config.epochs}", leave=True)
        optimizer.zero_grad(set_to_none=True)

        for weak_img_s, strong_img_s, dmg_s, bld_s, _event_s in pbar:
            try:
                _weak_img_t, strong_img_t, dmg_t, bld_t, _event_t = next(target_labeled_iter)
            except StopIteration:
                target_labeled_iter = iter(target_labeled_loader)
                _weak_img_t, strong_img_t, dmg_t, bld_t, _event_t = next(target_labeled_iter)

            strong_img_s = strong_img_s.to(device, non_blocking=True)
            dmg_s = dmg_s.to(device, non_blocking=True)
            bld_s = bld_s.to(device, non_blocking=True)

            strong_img_t = strong_img_t.to(device, non_blocking=True)
            dmg_t = dmg_t.to(device, non_blocking=True)
            bld_t = bld_t.to(device, non_blocking=True)

            with autocast(device_type=device.type):
                logits_s, feat_s = model(strong_img_s, building_mask=bld_s, return_features=True)
                logits_t, feat_t = model(strong_img_t, building_mask=bld_t, return_features=True)

                loss_t, _ce_t, _tv_t, _bd_t, train_mask_t = supervised_rare_damage_loss(
                    logits_t, dmg_t, bld_t, class_weights, focal_tversky_loss, context_weight=0.5
                )
                loss_s, _ce_s, _tv_s, _bd_s, train_mask_s = supervised_rare_damage_loss(
                    logits_s, dmg_s, bld_s, class_weights, focal_tversky_loss, context_weight=0.25
                )

                lambda_s = get_source_transfer_weight(epoch)
                loss = 1.0 * loss_t + lambda_s * loss_s

                if not torch.isfinite(loss):
                    print("Warning: loss is NaN/Inf. Skipping batch.")
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    continue

            with torch.no_grad():
                pred_s = torch.argmax(logits_s, dim=1)
                pred_t = torch.argmax(logits_t, dim=1)
                train_correct += ((pred_s == dmg_s) & train_mask_s).sum().item()
                train_correct += ((pred_t == dmg_t) & train_mask_t).sum().item()
                train_total += train_mask_s.sum().item()
                train_total += train_mask_t.sum().item()

            loss = loss / config.accum_steps
            scaler.scale(loss).backward()

            if (n_steps + 1) % config.accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss_epoch += loss.item() * config.accum_steps
            n_steps += 1

            pbar.set_postfix(
                {
                    "loss": f"{loss.item() * config.accum_steps:.3f}",
                    "t": f"{loss_t.item():.3f}",
                    "s": f"{loss_s.item():.3f}",
                    "ls": f"{lambda_s:.3f}",
                    "acc": f"{train_correct / max(train_total, 1):.3f}",
                }
            )

            del logits_s, logits_t, feat_s, feat_t, loss, loss_t, loss_s
            if n_steps % 200 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        train_loss = total_loss_epoch / max(n_steps, 1)
        train_acc = train_correct / max(train_total, 1)
        train_loss_hist.append(train_loss)
        print(f"\nEpoch {epoch + 1}/{config.epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

        if (n_steps % config.accum_steps) != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        is_last_epoch = (epoch + 1) == config.epochs
        if (epoch + 1) % config.eval_every == 0 or is_last_epoch:
            val_loss, metrics = _run_validation(model, target_val_loader, class_weights, config.num_classes, device)

            val_loss_hist.append(val_loss)
            miou_hist.append(metrics["miou_3class"])
            val_metrics_hist.append(metrics)

            rare_miou = metrics["miou_rare_2class"]
            print(
                f"\nValidation | Val Loss: {val_loss:.4f} | Accuracy: {metrics['accuracy']:.4f} "
                f"| Kappa: {metrics['kappa']:.4f}"
            )
            for name in CLASS_NAMES:
                print(f"{name} IoU: {metrics[f'iou_{name}']:.4f}")
            print(f"mIoU-3class: {metrics['miou_3class']:.4f} | mIoU-rare-2class: {rare_miou:.4f}")

            scheduler.step()

            if rare_miou > best_rare_miou:
                best_rare_miou = rare_miou
                best_metrics = metrics
                best_epoch = epoch + 1
                early_stop_counter = 0

                save_checkpoint(
                    model_dir / "best_model.pth",
                    epoch=epoch,
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    metrics=metrics,
                    miou_3class=metrics["miou_3class"],
                    miou_rare_2class=rare_miou,
                )
                print("Saved best model (by rare 2-class mIoU).")
            else:
                early_stop_counter += 1

            if early_stop_counter >= config.early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _export_final_validation_excel(
        model=model,
        model_dir=model_dir,
        out_dir=out_dir,
        target_val_loader=target_val_loader,
        class_weights=class_weights,
        num_classes=config.num_classes,
        device=device,
        best_epoch=best_epoch,
    )

    save_checkpoint(
        out_dir / "history_full.pth",
        train_loss=train_loss_hist,
        val_loss=val_loss_hist,
        miou=miou_hist,
        val_metrics=val_metrics_hist,
        best_rare_miou=best_rare_miou,
        best_epoch=best_epoch,
    )

    return best_metrics


def _run_validation(model, loader, class_weights, num_classes, device):
    """Run one validation pass; returns ``(val_loss, metrics_dict)``."""
    model.eval()
    val_loss_total, val_steps = 0.0, 0
    cm_total = torch.zeros(num_classes, num_classes)

    with torch.no_grad():
        for weak_img, _strong_img, damage, building, _event_name in tqdm(loader, desc="Validation", leave=False):
            img = weak_img.to(device, non_blocking=True)
            damage = damage.to(device, non_blocking=True)
            building = building.to(device, non_blocking=True)

            logits = model(img, building_mask=building)
            pred = torch.argmax(logits, dim=1)
            valid_mask = building > 0

            loss_val = masked_cross_entropy_loss(logits, damage, valid_mask, class_weights)
            val_loss_total += loss_val.item()
            val_steps += 1

            cm_total += update_confusion_matrix(pred, damage, valid_mask, num_classes).cpu()

    val_loss = val_loss_total / max(val_steps, 1)
    metrics = compute_metrics_from_cm(cm_total, num_classes=num_classes, drop_background=True)
    model.train()
    return val_loss, metrics


def _export_final_validation_excel(
    model, model_dir, out_dir, target_val_loader, class_weights, num_classes, device, best_epoch
):
    """Load the best checkpoint and export the global validation-set metrics to Excel."""
    best_model_path = Path(model_dir) / "best_model.pth"
    if best_model_path.exists():
        from .utils import load_checkpoint, load_model_weights

        checkpoint = load_checkpoint(best_model_path, map_location=device)
        load_model_weights(model, checkpoint, strict=True)
        print(f"Loaded best model from epoch {checkpoint.get('epoch', -1) + 1} for final validation export.")

    model.eval()
    val_loss_total, val_steps = 0.0, 0
    cm_total = torch.zeros(num_classes, num_classes, device=device)

    with torch.no_grad():
        for weak_img, _strong_img, damage, building, _event_name in tqdm(
            target_val_loader, desc="Final validation (global)"
        ):
            img = weak_img.to(device, non_blocking=True)
            damage = damage.to(device, non_blocking=True)
            building = building.to(device, non_blocking=True)

            logits = model(img, building_mask=building)
            preds = torch.argmax(logits, dim=1)
            valid_mask = building > 0

            loss_val = masked_cross_entropy_loss(logits, damage, valid_mask, class_weights)
            val_loss_total += loss_val.item()
            val_steps += 1

            cm_total += update_confusion_matrix(preds, damage, valid_mask, num_classes).to(device)

    metrics = compute_metrics_from_cm(cm_total, num_classes=num_classes, drop_background=True)

    row = {
        "best_epoch": best_epoch,
        "val_loss": val_loss_total / max(val_steps, 1),
        "accuracy": metrics["accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "kappa": metrics["kappa"],
        "miou_3class": metrics["miou_3class"],
        "miou_rare_2class": metrics["miou_rare_2class"],
    }
    for name in CLASS_NAMES:
        row[f"iou_{name}"] = metrics[f"iou_{name}"]
        row[f"precision_{name}"] = metrics[f"precision_{name}"]
        row[f"recall_{name}"] = metrics[f"recall_{name}"]
        row[f"f1_{name}"] = metrics[f"f1_{name}"]

    excel_path = Path(out_dir) / "final_validation_global_metrics.xlsx"
    pd.DataFrame([row]).to_excel(excel_path, index=False)
    print(f"Final validation (global) Excel saved: {excel_path}")
