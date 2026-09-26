#!/usr/bin/env python
"""Train one STRATA experiment from a YAML config.

Reproduces the notebook's ``run_experiment`` / ``train_and_eval`` entry
point. There is a single training mode in the original pipeline (joint
source + target supervised training); different experiments only differ
by the target label budget, so this one script covers every
``configs/budget_*.yaml`` file -- there is no separate
target-only/joint/fine-tuning script because the notebook never defines
those as distinct code paths.

Example:
    python scripts/train.py --config configs/budget_n100.yaml
    python scripts/train.py --config configs/budget_n100.yaml --out-dir /custom/path
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.trainer import TrainConfig, train_one_experiment  # noqa: E402
from src.utils import load_yaml_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config file.")
    parser.add_argument("--source-root", type=str, default=None, help="Override source_root from the config.")
    parser.add_argument("--target-root", type=str, default=None, help="Override target_root from the config.")
    parser.add_argument("--out-dir", type=str, default=None, help="Override the resolved output directory.")
    parser.add_argument("--splits-dir", type=str, default=None, help="Override splits_dir from the config.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from the config.")
    parser.add_argument("--gpu", type=str, default=None, help="Override the CUDA device id.")
    return parser.parse_args()


def main():
    args = parse_args()
    raw = load_yaml_config(args.config)

    if args.source_root is not None:
        raw["source_root"] = args.source_root
    if args.target_root is not None:
        raw["target_root"] = args.target_root
    if args.splits_dir is not None:
        raw["splits_dir"] = args.splits_dir
    if args.epochs is not None:
        raw["epochs"] = args.epochs
    if args.gpu is not None:
        raw["gpu"] = args.gpu

    run_name = raw.pop("run_name", "strata_run")
    out_root = raw.pop("out_root", "runs")
    out_dir = args.out_dir or str(Path(out_root) / "main" / "optical_to_optical" / run_name)
    raw["out_dir"] = out_dir

    known_fields = set(TrainConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in raw.items() if k in known_fields}
    unknown = set(raw.keys()) - known_fields
    if unknown:
        print(f"Warning: ignoring unknown config keys: {sorted(unknown)}")

    config = TrainConfig(**filtered)

    print("=" * 60)
    print("STRATA training run")
    print("=" * 60)
    for field_name in sorted(known_fields):
        print(f"{field_name}: {getattr(config, field_name)}")
    print("=" * 60)

    metrics = train_one_experiment(config)

    if metrics is not None:
        print("\nBest checkpoint metrics:")
        for k, v in metrics.items():
            if k != "iou_per_class":
                print(f"  {k}: {v}")
    print(f"\nOutputs saved under: {out_dir}")


if __name__ == "__main__":
    main()
