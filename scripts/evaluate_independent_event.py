#!/usr/bin/env python
"""Quantitative pixel-wise and building-wise evaluation on an independent,
held-out disaster event (e.g. Hurricane Dorian) never seen during training.

Run ``scripts/prepare_independent_event.py`` first to produce the NPZ
patch directory this script consumes.

Example:
    python scripts/evaluate_independent_event.py \\
        --data-root data/independent_events/HURRICANE-DORIAN/HURRICANE-DORIAN_npz_512valid_as_256 \\
        --checkpoint runs/main/optical_to_optical/STRATA_O2O_full_AllData/models/best_model.pth \\
        --event-name HURRICANE-DORIAN \\
        --out-excel runs/HURRICANE_DORIAN_test_metrics/HURRICANE_DORIAN_test_patch_metrics.xlsx
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets import IndependentEventDataset  # noqa: E402
from src.evaluator import evaluate_independent_event  # noqa: E402
from src.model import STRATA  # noqa: E402
from src.utils import get_device, load_checkpoint, load_model_weights  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--event-name", type=str, default="INDEPENDENT_EVENT")
    parser.add_argument("--out-excel", type=str, required=True)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gpu", type=str, default="0")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(args.gpu)

    dataset = IndependentEventDataset(args.data_root)
    print(f"{args.event_name} test patches: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = STRATA(num_classes=args.num_classes).to(device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    load_model_weights(model, checkpoint, strict=True)
    model.eval()

    evaluate_independent_event(
        model=model,
        test_loader=loader,
        num_classes=args.num_classes,
        device=device,
        out_excel_path=args.out_excel,
        event_name=args.event_name,
    )


if __name__ == "__main__":
    main()
