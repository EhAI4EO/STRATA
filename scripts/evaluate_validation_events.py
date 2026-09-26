#!/usr/bin/env python
"""Evaluate a trained checkpoint on the target-domain validation split,
broken down by individual event.

Reproduces the notebook's event-based validation Excel export. This is
the "validation-event evaluation" referred to in the README -- it uses
the *same* target domain and validation split seen during training
(never the independent held-out event; see
``scripts/evaluate_independent_event.py`` for that).

Example:
    python scripts/evaluate_validation_events.py \\
        --config configs/budget_n100.yaml \\
        --checkpoint runs/main/optical_to_optical/STRATA_O2O_n100/models/best_model.pth
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluator import evaluate_by_event  # noqa: E402
from src.model import STRATA  # noqa: E402
from src.sampling import build_dataloaders  # noqa: E402
from src.trainer import TrainConfig  # noqa: E402
from src.utils import get_device, load_checkpoint, load_model_weights, load_yaml_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out-excel", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    raw = load_yaml_config(args.config)
    raw.pop("run_name", None)
    raw.pop("out_root", None)
    raw["out_dir"] = "."

    known_fields = set(TrainConfig.__dataclass_fields__.keys())
    config = TrainConfig(**{k: v for k, v in raw.items() if k in known_fields})

    device = get_device(config.gpu)

    _source_loader, _target_labeled_loader, _unlabeled, target_val_loader = build_dataloaders(
        source_root=config.source_root,
        target_root=config.target_root,
        batch_size=1,
        event_split_path=config.event_split_path,
        fixed_split_path=config.fixed_split_path,
        experiment_split_path=config.experiment_split_path,
        target_labels_per_class=config.target_labels_per_class,
        num_workers=config.num_workers,
    )

    model = STRATA(num_classes=config.num_classes).to(device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    load_model_weights(model, checkpoint, strict=True)
    model.eval()
    best_epoch = checkpoint.get("epoch", -1) + 1 if isinstance(checkpoint, dict) else -1

    import torch

    class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32, device=device)

    checkpoint_dir = Path(args.checkpoint).resolve().parent.parent
    out_excel = args.out_excel or str(checkpoint_dir / "final_validation_event_based_metrics.xlsx")

    evaluate_by_event(
        model=model,
        val_loader=target_val_loader,
        class_weights=class_weights,
        num_classes=config.num_classes,
        device=device,
        out_excel_path=out_excel,
        best_epoch=best_epoch,
    )


if __name__ == "__main__":
    main()
