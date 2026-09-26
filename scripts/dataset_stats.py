#!/usr/bin/env python
"""Descriptive statistics for an NPZ patch directory: per-event damage-class
pixel counts, and (optionally) per-event building counts by majority
damage class.

Unifies two notebook utilities (a per-event pixel-count scan, and a
per-event connected-building count for the independent-event set) into
one script.

Example:
    python scripts/dataset_stats.py --data-root /path/to/ebd_npz
    python scripts/dataset_stats.py --data-root /path/to/HURRICANE-DORIAN_npz_512valid_as_256 --count-buildings
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets import get_event_name  # noqa: E402
from src.metrics import building_majority_map  # noqa: E402

CLASS_NAMES = {0: "background", 1: "intact", 2: "damaged", 3: "destroyed"}


def pixel_stats(data_root: Path) -> None:
    files_by_event = defaultdict(list)
    for f in sorted(data_root.glob("*.npz")):
        files_by_event[get_event_name(f)].append(f)

    print("\n" + "=" * 55)
    print("PER-EVENT DAMAGE-CLASS PIXEL STATISTICS (inside buildings)")
    print("=" * 55)

    for event, flist in files_by_event.items():
        counts = {1: 0, 2: 0, 3: 0}
        total_building_pixels = 0

        for fpath in tqdm(flist, desc=f"Scanning {event:<25}", leave=False):
            with np.load(fpath) as data:
                damage = data["damage_mask"]
                building = data["building_mask"]
            bld_pixels = damage[building > 0]
            total_building_pixels += len(bld_pixels)
            for c in (1, 2, 3):
                counts[c] += int(np.sum(bld_pixels == c))

        if total_building_pixels == 0:
            print(f"\n[{event}] No building pixels found.")
            continue

        print(f"\n{event} (files: {len(flist)}, building pixels: {total_building_pixels:,})")
        for c in (1, 2, 3):
            pct = 100.0 * counts[c] / total_building_pixels
            print(f"  {CLASS_NAMES[c]:>10} ({c}): {counts[c]:>12,} pixels | {pct:>7.4f}%")


def building_count_stats(data_root: Path, connectivity: int = 8) -> None:
    files_by_event = defaultdict(list)
    for f in sorted(data_root.glob("*.npz")):
        files_by_event[get_event_name(f)].append(f)

    print("\n" + "=" * 55)
    print("PER-EVENT BUILDING COUNTS BY MAJORITY DAMAGE CLASS")
    print("=" * 55)

    for event, flist in files_by_event.items():
        building_class_counts = {1: 0, 2: 0, 3: 0}

        for fpath in tqdm(flist, desc=f"Scanning {event:<25}", leave=False):
            with np.load(fpath) as data:
                damage = data["damage_mask"]
                building = data["building_mask"]

            majority = building_majority_map(damage, building, num_classes=4)
            # Count each connected building once: re-label and take one
            # majority value per connected component.
            from scipy import ndimage

            labeled, n = ndimage.label(building > 0)
            for i in range(1, n + 1):
                region = labeled == i
                vals = majority[region]
                vals = vals[vals > 0]
                if len(vals) == 0:
                    continue
                cls = int(np.bincount(vals, minlength=4).argmax())
                if cls in building_class_counts:
                    building_class_counts[cls] += 1

        print(f"\n{event}")
        for c in (1, 2, 3):
            print(f"  {CLASS_NAMES[c]:>10} ({c}) buildings: {building_class_counts[c]:,}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--count-buildings", action="store_true", help="Also compute per-building damage-class counts.")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    pixel_stats(data_root)
    if args.count_buildings:
        building_count_stats(data_root)


if __name__ == "__main__":
    main()
