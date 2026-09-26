#!/usr/bin/env python
"""Generate qualitative 512x512 prediction grids (RGB / GT / raw prediction /
raw TP-FP-FN / building-wise prediction / building-wise TP-FP-FN).

Reproduces and unifies two near-identical notebook cells: one used for the
label-budget validation split, one used for the independent Hurricane
Dorian test set. Every complete 512x512 group (4 co-located 256x256
patches) that contains at least one damaged/destroyed pixel in its ground
truth is rendered as one row of a grid image; grids are written in
batches of ``--samples-per-file`` rows.

Example (validation-linked, e.g. for Figure 3-style qualitative panels):
    python scripts/visualize_predictions.py \\
        --data-root /path/to/ebd_npz \\
        --checkpoint runs/.../STRATA_O2O_n250/models/best_model.pth \\
        --validation-files-npz splits/ebd_fixed_train_val_split.npz \\
        --out-dir runs/visualizations/STRATA_n250

Example (independent event, no validation subset -- every complete group is eligible):
    python scripts/visualize_predictions.py \\
        --data-root data/independent_events/HURRICANE-DORIAN/HURRICANE-DORIAN_npz_512valid_as_256 \\
        --checkpoint runs/.../STRATA_O2O_full_AllData/models/best_model.pth \\
        --out-dir runs/visualizations/HURRICANE-DORIAN_TEST_512
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets import BuildingDamageDataset, get_event_name  # noqa: E402
from src.metrics import building_majority_map  # noqa: E402
from src.model import STRATA  # noqa: E402
from src.utils import get_device, load_checkpoint, load_model_weights  # noqa: E402

PATCH_SIZE = 256
MERGED_SIZE = 512
POSITIONS = ((0, 0), (0, PATCH_SIZE), (PATCH_SIZE, 0), (PATCH_SIZE, PATCH_SIZE))

COLOR_INTACT = np.array([70, 220, 90], dtype=np.uint8)
COLOR_DAMAGE = np.array([255, 170, 40], dtype=np.uint8)
COLOR_DESTROYED = np.array([255, 70, 70], dtype=np.uint8)
COLOR_TP = np.array([190, 190, 190], dtype=np.uint8)
COLOR_FP = np.array([255, 230, 40], dtype=np.uint8)
COLOR_FN = np.array([50, 130, 255], dtype=np.uint8)

COLUMN_TITLES = (
    "Optical RGB",
    "Ground Truth Map",
    "Raw Prediction (Pixel-wise)",
    "Raw Evaluation Map\nTP=Gray | FP=Yellow | FN=Blue",
    "Building-wise Prediction Map",
    "Building Evaluation Map\nTP=Gray | FP=Yellow | FN=Blue",
)


def to_rgb(image) -> np.ndarray:
    image = image.detach().cpu().numpy() if torch.is_tensor(image) else np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (1, 2, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    image = image.astype(np.float32)
    if image.max() > 1:
        image /= 255.0
    image = np.clip(image, 0, 1)
    image = np.power(image, 0.85)  # mild gamma boost, display only
    return np.clip(image * 1.08, 0, 1)


def mask_to_2d(mask) -> np.ndarray:
    mask = mask.detach().cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    return mask


def colorize_damage(mask: np.ndarray) -> np.ndarray:
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    color[mask == 1] = COLOR_INTACT
    color[mask == 2] = COLOR_DAMAGE
    color[mask == 3] = COLOR_DESTROYED
    return color


def colorize_eval(mask: np.ndarray) -> np.ndarray:
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    color[mask == 1] = COLOR_TP
    color[mask == 2] = COLOR_FP
    color[mask == 3] = COLOR_FN
    return color


def make_raw_eval_mask(gt: np.ndarray, prediction: np.ndarray, building: np.ndarray) -> np.ndarray:
    valid = building > 0
    output = np.zeros_like(gt, dtype=np.uint8)
    output[valid & (gt == prediction) & (gt > 0)] = 1
    output[valid & (gt == 1) & np.isin(prediction, (2, 3))] = 2
    output[valid & np.isin(gt, (2, 3)) & ((~np.isin(prediction, (2, 3))) | (gt != prediction))] = 3
    return output


def make_building_eval_mask(gt_majority: np.ndarray, pred_majority: np.ndarray, building: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(building > 0)
    output = np.zeros_like(gt_majority, dtype=np.uint8)
    for object_id in range(1, count + 1):
        region = labels == object_id
        gt_class = gt_majority[region][gt_majority[region] > 0]
        pred_class = pred_majority[region][pred_majority[region] > 0]
        gt_class = np.bincount(gt_class, minlength=4).argmax() if len(gt_class) else 0
        pred_class = np.bincount(pred_class, minlength=4).argmax() if len(pred_class) else 0
        if gt_class == pred_class and gt_class > 0:
            output[region] = 1
        elif gt_class == 1 and pred_class in (2, 3):
            output[region] = 2
        elif gt_class in (2, 3) and pred_class != gt_class:
            output[region] = 3
    return output


def parse_patch_name(path: str):
    stem = Path(path).stem
    base_name, y_text, x_text = stem.rsplit("_", 2)
    return base_name, int(y_text), int(x_text)


def make_complete_groups(files, restrict_to=None):
    grouped = {}
    for path in files:
        try:
            base_name, y, x = parse_patch_name(path)
        except ValueError:
            continue
        block_y, block_x = (y // MERGED_SIZE) * MERGED_SIZE, (x // MERGED_SIZE) * MERGED_SIZE
        grouped.setdefault((base_name, block_y, block_x), {})[(y - block_y, x - block_x)] = path

    required = set(POSITIONS)
    complete = []
    for (base_name, block_y, block_x), patches in grouped.items():
        if not required.issubset(patches):
            continue
        if restrict_to is not None:
            matched = sum(1 for p in patches.values() if p in restrict_to)
            if matched == 0:
                continue
        else:
            matched = 4
        complete.append(
            {
                "base_name": base_name,
                "block_y": block_y,
                "block_x": block_x,
                "patches": {pos: patches[pos] for pos in required},
                "matched_count": matched,
            }
        )
    complete.sort(key=lambda g: (g["base_name"], g["block_y"], g["block_x"]))
    return complete


def build_package(model, dataset, file_to_index, group, device):
    rgb = np.zeros((MERGED_SIZE, MERGED_SIZE, 3), dtype=np.float32)
    damage = np.zeros((MERGED_SIZE, MERGED_SIZE), dtype=np.uint8)
    building = np.zeros((MERGED_SIZE, MERGED_SIZE), dtype=np.uint8)
    prediction = np.zeros((MERGED_SIZE, MERGED_SIZE), dtype=np.uint8)
    event_name = get_event_name(group["base_name"])

    for (local_y, local_x), path in group["patches"].items():
        idx = file_to_index[path]
        sample = dataset[idx]
        image, damage_t, building_t = sample[0], sample[2], sample[3]

        rgb_patch = to_rgb(image)
        damage_patch = mask_to_2d(damage_t).astype(np.uint8)
        building_patch = (mask_to_2d(building_t) > 0).astype(np.uint8)

        with torch.inference_mode():
            image_batch = image.unsqueeze(0).to(device)
            building_batch = building_t.unsqueeze(0).to(device)
            logits = model(image_batch, building_mask=building_batch)
            pred_patch = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
        pred_patch = pred_patch * building_patch

        ys, xs = slice(local_y, local_y + PATCH_SIZE), slice(local_x, local_x + PATCH_SIZE)
        rgb[ys, xs] = rgb_patch
        damage[ys, xs] = damage_patch
        building[ys, xs] = building_patch
        prediction[ys, xs] = pred_patch

    if not np.any((damage == 2) | (damage == 3)):
        return None

    gt_majority = building_majority_map(damage, building)
    pred_majority = building_majority_map(prediction, building)
    raw_eval = make_raw_eval_mask(damage, prediction, building)
    building_eval = make_building_eval_mask(gt_majority, pred_majority, building)

    return {
        "event": event_name,
        "file": group["base_name"],
        "block_y": group["block_y"],
        "block_x": group["block_x"],
        "rgb": rgb,
        "gt_map": colorize_damage(gt_majority),
        "raw_pred_map": colorize_damage(prediction),
        "eval_raw_map": colorize_eval(raw_eval),
        "pred_map": colorize_damage(pred_majority),
        "eval_map": colorize_eval(building_eval),
    }


def save_grid(packages, out_path, dpi=600, jpg_quality=92):
    if not packages:
        return None
    rows, columns = len(packages), len(COLUMN_TITLES)
    fig, axes = plt.subplots(rows, columns, figsize=(3.1 * columns, 3.35 * rows), squeeze=False)
    fig.subplots_adjust(top=0.985, bottom=0.01, left=0.01, right=0.995, wspace=0.025, hspace=0.18)

    for column, title in enumerate(COLUMN_TITLES):
        axes[0, column].set_title(title, fontsize=9, fontweight="bold", pad=10)

    keys = ("rgb", "gt_map", "raw_pred_map", "eval_raw_map", "pred_map", "eval_map")
    for row, package in enumerate(packages):
        row_title = f"{package['event']} | {package['file']} | y={package['block_y']} x={package['block_x']}"
        for column, key in enumerate(keys):
            axes[row, column].imshow(package[key])
            axes[row, column].axis("off")
            if column == 0:
                axes[row, column].text(
                    0, -0.045, row_title, transform=axes[row, column].transAxes,
                    fontsize=6.5, fontweight="bold", va="top", ha="left",
                )

    fig.savefig(
        out_path, dpi=dpi, format="jpg", bbox_inches=None, pad_inches=0,
        pil_kwargs={"quality": jpg_quality, "optimize": True, "progressive": True},
    )
    plt.close(fig)
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument(
        "--validation-files-npz", type=str, default=None,
        help="Optional splits/*.npz with a 'val_files' array; restricts grids to "
             "groups that overlap the validation set. Omit for a test-only directory "
             "(e.g. an independent event) where every complete group is eligible.",
    )
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--samples-per-file", type=int, default=6)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--tag", type=str, default="STRATA")
    parser.add_argument("--gpu", type=str, default="0")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(args.gpu)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = STRATA(num_classes=args.num_classes).to(device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    load_model_weights(model, checkpoint, strict=True)
    model.eval()

    dataset = BuildingDamageDataset(args.data_root, is_source=False, augment=False)
    all_files = list(dataset.files)
    file_to_index = {path: idx for idx, path in enumerate(all_files)}

    restrict_to = None
    if args.validation_files_npz is not None:
        data = np.load(args.validation_files_npz, allow_pickle=True)
        restrict_to = set(data["val_files"])

    groups = make_complete_groups(all_files, restrict_to=restrict_to)
    print(f"Complete 512 groups: {len(groups)}")

    saved_grids, packages, chunk_start = [], [], 0
    for i, group in enumerate(tqdm(groups, desc="Building prediction packages")):
        package = build_package(model, dataset, file_to_index, group, device)
        if package is not None:
            packages.append(package)

        if len(packages) == args.samples_per_file or i == len(groups) - 1:
            if packages:
                out_path = out_dir / f"{args.tag}_eval_512_{chunk_start}_{i}.jpg"
                saved = save_grid(packages, out_path, dpi=args.dpi)
                if saved is not None:
                    saved_grids.append(str(saved))
            packages = []
            chunk_start = i + 1

    print(f"\nSaved {len(saved_grids)} grid file(s):")
    for p in saved_grids:
        print(" ", p)


if __name__ == "__main__":
    main()
