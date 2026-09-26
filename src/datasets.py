"""NPZ-backed datasets for STRATA building-damage segmentation.

Reproduces the ``BuildingDamageDataset`` used in the original research
notebook (weak/strong augmentation branches, NPZ memory-mapped caching).
Each NPZ patch is expected to contain three arrays:

    image          (H, W) or (H, W, C), uint8      -- post-event VHR optical patch
    damage_mask    (H, W), uint8                   -- harmonized 4-class label
    building_mask  (H, W), uint8/float              -- binary building footprint

Label encoding (harmonized STRATA scheme):
    0 = Background
    1 = Intact
    2 = Damaged
    3 = Destroyed
"""

from __future__ import annotations

import os
import random
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Pixel-level helpers
# ---------------------------------------------------------------------------

def normalize_image(img: np.ndarray) -> np.ndarray:
    """Scale a uint8 image to [0, 1]."""
    return img.astype(np.float32) / 255.0


def safe_copy(x: np.ndarray) -> np.ndarray:
    """Return a contiguous copy, required after numpy slicing/flipping views."""
    return np.ascontiguousarray(x)


def ensure_channel(img: np.ndarray) -> np.ndarray:
    """Ensure a trailing channel dimension: (H, W) -> (H, W, 1)."""
    if img.ndim == 2:
        img = img[:, :, None]
    return img


def random_flip(img: np.ndarray, mask: np.ndarray, bld: np.ndarray):
    """Joint horizontal/vertical flip applied identically to image and masks."""
    if random.random() < 0.5:
        img, mask, bld = img[:, ::-1], mask[:, ::-1], bld[:, ::-1]
    if random.random() < 0.5:
        img, mask, bld = img[::-1, :], mask[::-1, :], bld[::-1, :]
    return img, mask, bld


def random_rotate(img: np.ndarray, mask: np.ndarray, bld: np.ndarray):
    """Joint random 90-degree rotation applied identically to image and masks."""
    k = random.randint(0, 3)
    return np.rot90(img, k), np.rot90(mask, k), np.rot90(bld, k)


def gaussian_noise(img: np.ndarray) -> np.ndarray:
    """Mild additive Gaussian noise, applied with probability 0.3."""
    if random.random() < 0.3:
        img = img + np.random.normal(0, 0.02, img.shape)
    return np.clip(img, 0, 1)


def color_jitter(img: np.ndarray) -> np.ndarray:
    """Mild global brightness scaling, applied with probability 0.5."""
    if random.random() < 0.5:
        img = img * (0.8 + 0.4 * random.random())
    return np.clip(img, 0, 1)


def get_event_name(path: PathLike) -> str:
    """Extract the event identifier from an NPZ filename.

    Expected filename pattern: ``{EVENT-NAME}_{id}_{y}_{x}.npz``, e.g.
    ``EARTHQUAKE-TURKEY_018187_256_256.npz``.
    """
    return os.path.basename(str(path)).split("_")[0]


# ---------------------------------------------------------------------------
# NPZ cache
# ---------------------------------------------------------------------------

class NPZCache:
    """A small LRU cache of memory-mapped ``np.load`` handles.

    Keeps up to ``max_size`` open NPZ handles to avoid repeatedly touching
    the filesystem for files revisited within an epoch.
    """

    def __init__(self, max_size: int = 1000):
        self.cache: "OrderedDict[str, np.lib.npyio.NpzFile]" = OrderedDict()
        self.max_size = max_size

    def get(self, path: PathLike):
        path = str(path)
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]

        data = np.load(path, mmap_mode="r")

        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)

        self.cache[path] = data
        return data


# ---------------------------------------------------------------------------
# Labeled dataset
# ---------------------------------------------------------------------------

class BuildingDamageDataset(Dataset):
    """Labeled building-damage segmentation dataset backed by NPZ patches.

    Returns, per sample, a weakly-augmented view (used as the "clean"
    training input) and a strongly-augmented view (currently unused by the
    active training loop but kept for parity with the original notebook and
    for future experimentation), together with the damage mask, building
    mask, and source event name.

    Args:
        root_dir: directory containing ``*.npz`` patch files.
        is_source: whether this dataset represents the labeled source
            domain. Retained for interface parity with the original
            notebook; it does not change any preprocessing behavior.
        augment: whether to apply the weak/strong augmentation pipeline.
    """

    def __init__(self, root_dir: PathLike, is_source: bool = True, augment: bool = True):
        self.root_dir = Path(root_dir)
        self.files: List[str] = sorted(
            str(p) for p in self.root_dir.glob("*.npz")
        )
        random.shuffle(self.files)

        self.is_source = is_source
        self.augment = augment
        self.cache = NPZCache(max_size=1000)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        data = self.cache.get(self.files[idx])

        img = data["image"]
        damage = data["damage_mask"]
        building = data["building_mask"]

        img = ensure_channel(img)
        img = normalize_image(img)

        # --- weak augmentation (flip + 90-degree rotation) ---
        weak_img, weak_damage, weak_building = img.copy(), damage.copy(), building.copy()

        if self.augment:
            weak_img, weak_damage, weak_building = random_flip(
                weak_img, weak_damage, weak_building
            )
            weak_img, weak_damage, weak_building = random_rotate(
                weak_img, weak_damage, weak_building
            )

        # --- strong augmentation (photometric only, on top of the weak view) ---
        strong_img = weak_img.copy()

        if self.augment:
            has_rare_class = np.any((damage == 2) | (damage == 3))
            if has_rare_class:
                strong_img = gaussian_noise(strong_img)
                strong_img = color_jitter(strong_img)
            else:
                if random.random() < 0.5:
                    strong_img = gaussian_noise(strong_img)
                if random.random() < 0.3:
                    strong_img = color_jitter(strong_img)

        weak_img = safe_copy(weak_img)
        strong_img = safe_copy(strong_img)
        weak_damage = safe_copy(weak_damage)
        weak_building = safe_copy(weak_building)

        weak_img = np.transpose(weak_img, (2, 0, 1))
        strong_img = np.transpose(strong_img, (2, 0, 1))

        weak_img_t = torch.from_numpy(weak_img).float()
        strong_img_t = torch.from_numpy(strong_img).float()
        weak_damage_t = torch.from_numpy(weak_damage).long()
        weak_building_t = torch.from_numpy(weak_building).float()

        event_name = get_event_name(self.files[idx])

        return weak_img_t, strong_img_t, weak_damage_t, weak_building_t, event_name


# ---------------------------------------------------------------------------
# Unlabeled dataset
# ---------------------------------------------------------------------------

class BuildingDamageDatasetUnlabeled(Dataset):
    """Unlabeled NPZ dataset intended for pseudo-label / consistency training.

    NOTE: This class is preserved from the original notebook for interface
    parity, but the active STRATA training pipeline (:mod:`src.trainer`)
    never instantiates an unlabeled loader — ``build_dataloaders`` always
    returns ``target_unlabeled_loader = None``. There is no pseudo-labeling
    step in the reported results. Kept here only so downstream code that
    experiments with semi-supervised extensions has a ready-made starting
    point.
    """

    def __init__(self, root_dir: PathLike):
        self.root_dir = Path(root_dir)
        self.files: List[str] = sorted(str(p) for p in self.root_dir.glob("*.npz"))
        random.shuffle(self.files)
        self.cache = NPZCache(max_size=300)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        data = self.cache.get(self.files[idx])

        img = data["image"]
        building = data["building_mask"]

        img = ensure_channel(img)
        img = normalize_image(img)

        dummy_mask = building.copy()

        img_w, bld_w = img.copy(), building.copy()
        img_w, _, bld_w = random_flip(img_w, dummy_mask, bld_w)

        # Strong branch keeps the same spatial transform as the weak branch
        # (only photometric augmentation differs) so that pixel-aligned
        # pseudo-labels would remain valid if this path were ever activated.
        img_s, bld_s = img_w.copy(), bld_w.copy()
        img_s = gaussian_noise(img_s)
        if img_s.shape[-1] == 3:
            img_s = color_jitter(img_s)
        building_out = bld_s

        img_w = safe_copy(img_w)
        img_s = safe_copy(img_s)
        building_out = safe_copy(building_out)

        img_w_t = torch.from_numpy(np.transpose(img_w, (2, 0, 1))).float()
        img_s_t = torch.from_numpy(np.transpose(img_s, (2, 0, 1))).float()
        building_t = torch.from_numpy(building_out).float()

        return img_w_t, img_s_t, building_t


# ---------------------------------------------------------------------------
# Independent-event test dataset (e.g. Hurricane Dorian)
# ---------------------------------------------------------------------------

class IndependentEventDataset(Dataset):
    """Reads an entire independent-event NPZ directory as a flat test set.

    No split, shuffling, or augmentation is applied -- every patch under
    ``root_dir`` is used. Named ``PakistanTestNPZDataset`` /
    ``EventTestNPZDataset`` in the original notebook (leftover naming from
    an earlier, unrelated event); unified here under one name since the
    two were functionally identical.
    """

    def __init__(self, root_dir: PathLike):
        self.root_dir = Path(root_dir)
        self.files: List[str] = sorted(str(p) for p in self.root_dir.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in: {root_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        file_path = self.files[idx]

        with np.load(file_path) as data:
            img = data["image"]
            damage = data["damage_mask"]
            building = data["building_mask"]

        img = ensure_channel(img)
        img = normalize_image(img)
        img = safe_copy(img)
        damage = safe_copy(damage)
        building = safe_copy(building)

        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW

        img_t = torch.from_numpy(img).float()
        damage_t = torch.from_numpy(damage.astype(np.uint8)).long()
        building_t = torch.from_numpy(building.astype(np.float32)).float()

        event_name = get_event_name(file_path)

        return img_t, damage_t, building_t, event_name, file_path
