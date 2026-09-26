#!/usr/bin/env python
"""Download one EBD event from Figshare and prepare 256x256 NPZ patches for
independent-event testing.

Reproduces the "512-block-validated" patch extraction actually consumed
by the notebook's independent-event evaluation cells (the notebook also
contains an earlier, per-256-patch-filtered variant that is superseded by
this one and is not reproduced here -- see the README audit notes).

Label harmonization (must match the main EBD source/target NPZ patches):
    Original EBD damage_mask:  0=background, 1=no damage, 2=minor damage,
                                3=major damage, 4=destroyed
    Harmonized STRATA labels:  0=Background, 1=Intact (no damage + minor),
                                2=Damaged (major), 3=Destroyed

A 512x512 block is kept only if, within its building footprint, all three
foreground classes {1, 2, 3} are present; if so, all four of its 256x256
sub-patches are saved (not filtered individually), which is what makes the
"complete 512 group" reconstruction in
``scripts/visualize_predictions.py`` and the building-wise evaluation in
``scripts/evaluate_independent_event.py`` possible.

Requires the extracted EBD event folder to contain ``images/`` and
``masks/`` subfolders with files named ``..._pre_disaster.png`` /
``..._post_disaster.png``, matching the official EBD release layout
(https://doi.org/10.6084/m9.figshare.25285009).

Example:
    python scripts/prepare_independent_event.py \\
        --event-name HURRICANE-DORIAN \\
        --figshare-article-id 25285009 \\
        --local-root data/independent_events/HURRICANE-DORIAN
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PATCH_SIZE = 256
BLOCK_SIZE = 512
REQUIRED_DAMAGE_CLASSES = {1, 2, 3}


def get_figshare_files(article_id: int, page_size: int = 100) -> list:
    files, page = [], 1
    while True:
        url = f"https://api.figshare.com/v2/articles/{article_id}/files"
        r = requests.get(url, params={"page": page, "page_size": page_size}, timeout=60)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        files.extend(batch)
        if len(batch) < page_size:
            break
        page += 1
    return files


def find_event_file(article_id: int, event_name: str) -> dict:
    files = get_figshare_files(article_id)
    matches = [f for f in files if event_name.upper() in f["name"].upper()]
    if not matches:
        available = "\n".join(f"  - {f['name']}" for f in files)
        raise FileNotFoundError(f"No Figshare file found for event '{event_name}'. Available files:\n{available}")
    if len(matches) > 1:
        print(f"Multiple matches for '{event_name}'; using the first: {matches[0]['name']}")
    return matches[0]


def download_file(url: str, out_path: Path, chunk_size: int = 1024 * 1024) -> None:
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"File already exists: {out_path} ({out_path.stat().st_size / 1024**3:.2f} GB)")
        return
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        with open(out_path, "wb") as f, tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    marker = extract_dir / ".extracted_done"
    if marker.exists():
        print(f"Already extracted: {extract_dir}")
        return
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)
    marker.write_text("done")


def convert_damage_to_3class_plus_background(damage: np.ndarray) -> np.ndarray:
    """Harmonize the raw 5-class EBD damage mask into the 4-class STRATA scheme."""
    new_damage = np.zeros_like(damage, dtype=np.uint8)
    new_damage[(damage == 1) | (damage == 2)] = 1  # no-damage + minor -> Intact
    new_damage[damage == 3] = 2                     # major -> Damaged
    new_damage[damage == 4] = 3                     # destroyed -> Destroyed
    return new_damage


def block_has_required_classes(dmg_block: np.ndarray, bld_block: np.ndarray, required_classes: set) -> bool:
    if np.sum(bld_block) == 0:
        return False
    present = set(np.unique(dmg_block[bld_block > 0]).astype(int).tolist())
    return required_classes.issubset(present)


def prepare_event_512valid_as_256(extract_dir: Path, save_root: Path) -> Path:
    post_masks = sorted(extract_dir.glob("**/masks/*_post_disaster.png"))
    if not post_masks:
        raise FileNotFoundError(f"No post-disaster masks found under: {extract_dir}")
    print(f"Total post-disaster masks found: {len(post_masks)}")

    saved_512_blocks = saved_256_patches = 0
    skipped = Counter()
    original_damage_values, converted_damage_values = Counter(), Counter()

    for pmask_path in tqdm(post_masks, desc="Preparing 512-valid NPZ patches"):
        pimg_path = Path(str(pmask_path).replace(f"{os.sep}masks{os.sep}", f"{os.sep}images{os.sep}"))
        pre_mask_path = Path(str(pmask_path).replace("_post_disaster.png", "_pre_disaster.png"))

        img = cv2.imread(str(pimg_path), cv2.IMREAD_COLOR)
        damage = cv2.imread(str(pmask_path), cv2.IMREAD_GRAYSCALE)
        building = cv2.imread(str(pre_mask_path), cv2.IMREAD_GRAYSCALE)

        if img is None or damage is None or building is None:
            skipped["missing_file"] += 1
            continue
        if img.shape[:2] != damage.shape or damage.shape != building.shape:
            skipped["bad_shape"] += 1
            continue

        for v in np.unique(damage):
            original_damage_values[int(v)] += 1

        building = (building > 0).astype(np.uint8)
        new_damage = convert_damage_to_3class_plus_background(damage)

        for v in np.unique(new_damage):
            converted_damage_values[int(v)] += 1

        h, w = img.shape[:2]
        base_name = pmask_path.stem.replace("_post_disaster", "")

        for y512 in range(0, h, BLOCK_SIZE):
            for x512 in range(0, w, BLOCK_SIZE):
                img_block = img[y512 : y512 + BLOCK_SIZE, x512 : x512 + BLOCK_SIZE]
                dmg_block = new_damage[y512 : y512 + BLOCK_SIZE, x512 : x512 + BLOCK_SIZE]
                bld_block = building[y512 : y512 + BLOCK_SIZE, x512 : x512 + BLOCK_SIZE]

                if img_block.shape[0] != BLOCK_SIZE or img_block.shape[1] != BLOCK_SIZE:
                    skipped["incomplete_512"] += 1
                    continue
                if np.sum(bld_block) == 0:
                    skipped["no_building"] += 1
                    continue
                if not block_has_required_classes(dmg_block, bld_block, REQUIRED_DAMAGE_CLASSES):
                    skipped["missing_required_classes"] += 1
                    continue

                for dy in (0, PATCH_SIZE):
                    for dx in (0, PATCH_SIZE):
                        y, x = y512 + dy, x512 + dx
                        img_patch = img[y : y + PATCH_SIZE, x : x + PATCH_SIZE]
                        dmg_patch = new_damage[y : y + PATCH_SIZE, x : x + PATCH_SIZE]
                        bld_patch = building[y : y + PATCH_SIZE, x : x + PATCH_SIZE]

                        if img_patch.shape[0] != PATCH_SIZE or img_patch.shape[1] != PATCH_SIZE:
                            continue

                        save_path = save_root / f"{base_name}_{y}_{x}.npz"
                        np.savez_compressed(
                            save_path,
                            image=img_patch.astype(np.uint8),
                            building_mask=bld_patch.astype(np.uint8),
                            damage_mask=dmg_patch.astype(np.uint8),
                        )
                        saved_256_patches += 1

                saved_512_blocks += 1

    print("\n" + "=" * 50)
    print("Preprocessing finished.")
    print(f"Saved valid 512 blocks: {saved_512_blocks} | Saved 256 patches: {saved_256_patches}")
    print(f"Skipped: {dict(skipped)}")
    print(f"Original damage values observed: {dict(sorted(original_damage_values.items()))}")
    print(f"Converted damage values observed: {dict(sorted(converted_damage_values.items()))}")
    if saved_512_blocks == 0:
        print("WARNING: no valid 512 block was saved (no block contained all of classes {1,2,3} inside buildings).")
    print("=" * 50)

    return save_root


def check_complete_512_groups(npz_dir: Path) -> None:
    """Sanity check: every saved patch should belong to a complete 4-patch 512 group."""
    files = sorted(npz_dir.glob("*.npz"))
    grouped = defaultdict(dict)

    for path in files:
        stem = path.stem
        base_name, y_str, x_str = stem.rsplit("_", 2)
        y, x = int(y_str), int(x_str)
        block_y, block_x = (y // BLOCK_SIZE) * BLOCK_SIZE, (x // BLOCK_SIZE) * BLOCK_SIZE
        grouped[(base_name, block_y, block_x)][(y - block_y, x - block_x)] = path

    required = {(0, 0), (0, PATCH_SIZE), (PATCH_SIZE, 0), (PATCH_SIZE, PATCH_SIZE)}
    complete = sum(1 for patches in grouped.values() if required.issubset(patches.keys()))
    print(f"Sanity check: {len(files)} files, {complete} complete 512 groups, {len(grouped) - complete} incomplete.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-name", type=str, required=True, help='e.g. "HURRICANE-DORIAN"')
    parser.add_argument("--figshare-article-id", type=int, default=25285009, help="EBD Figshare article id.")
    parser.add_argument("--local-root", type=str, required=True, help="Local working directory for this event.")
    return parser.parse_args()


def main():
    args = parse_args()
    local_root = Path(args.local_root)
    zip_path = local_root / f"{args.event_name}.zip"
    extract_dir = local_root / "extracted"
    save_root = local_root / f"{args.event_name}_npz_512valid_as_256"

    local_root.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    if save_root.exists():
        shutil.rmtree(save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    event_file = find_event_file(args.figshare_article_id, args.event_name)
    print(f"Selected Figshare file: {event_file['name']} ({event_file.get('size', 0) / 1024**3:.2f} GB)")
    download_file(event_file["download_url"], zip_path)
    extract_zip(zip_path, extract_dir)

    prepared_dir = prepare_event_512valid_as_256(extract_dir, save_root)
    check_complete_512_groups(prepared_dir)

    print(f"\nPrepared independent-event dataset directory: {prepared_dir}")


if __name__ == "__main__":
    main()
