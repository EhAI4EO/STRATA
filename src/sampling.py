"""Source/target event splits, per-class label-budget sampling, and dataloaders.

Reproduces the split logic actually used by the final pipeline
(``create_event_based_split`` in the notebook -- the earlier
``create_same_domain_split`` random-ratio split was defined but never
called, and is intentionally not ported here).

Split hierarchy
----------------
1. **Event-based source/target split** (:func:`create_event_based_split`):
   patches from ``TARGET_EVENTS`` become the target domain; everything
   else becomes the source domain. Saved once to an ``.npz`` file and
   reused across runs.
2. **Fixed train/val split of the target domain** (:func:`create_fixed_split`):
   a 70/30 (``val_ratio=0.3``) split of the target files, saved once and
   shared across every label-budget experiment so that validation never
   sees a different set of patches across budgets.
3. **Per-class label budget** (:func:`split_per_class_nested`): from the
   target *training* files, up to ``n_per_class`` files containing each
   foreground class are drawn as "labeled"; the remainder are "unlabeled"
   (unused by the active training loop, which never instantiates an
   unlabeled loader).
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .datasets import BuildingDamageDataset, get_event_name

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Event-based source/target split
# ---------------------------------------------------------------------------

def create_event_based_split(root_dir: PathLike, save_path: PathLike, target_events: Set[str]) -> None:
    """Partition all NPZ files under ``root_dir`` into source/target by event name."""
    root_dir = Path(root_dir)
    all_files = sorted(str(p) for p in root_dir.glob("*.npz"))

    source_files, target_files = [], []
    for f in all_files:
        if get_event_name(f) in target_events:
            target_files.append(f)
        else:
            source_files.append(f)

    np.savez(
        save_path,
        source_files=source_files,
        target_files=target_files,
        target_events=list(target_events),
    )
    print(f"Saved event-based split to: {save_path}")
    print(f"Source samples: {len(source_files)} | Target samples: {len(target_files)}")


def load_event_based_split(split_path: PathLike) -> Tuple[List[str], List[str]]:
    data = np.load(split_path, allow_pickle=True)
    return list(data["source_files"]), list(data["target_files"])


# ---------------------------------------------------------------------------
# Fixed target-domain train/val split
# ---------------------------------------------------------------------------

def create_fixed_split(files: Sequence[str], val_ratio: float, save_path: PathLike, seed: int = 42) -> None:
    """Save a single, fixed train/val split of ``files`` to be reused across budgets."""
    rng = random.Random(seed)
    files = sorted(files)
    rng.shuffle(files)

    n_val = int(len(files) * val_ratio)
    val_files = files[:n_val]
    train_files = files[n_val:]

    np.savez(save_path, train_files=train_files, val_files=val_files)
    print(f"Saved fixed train/val split to: {save_path}")


def load_fixed_split(split_path: PathLike) -> Tuple[List[str], List[str]]:
    data = np.load(split_path, allow_pickle=True)
    return list(data["train_files"]), list(data["val_files"])


# ---------------------------------------------------------------------------
# Per-class label budget
# ---------------------------------------------------------------------------

def split_per_class_nested(
    train_files: Sequence[str], n_per_class: int, seed: int = 42
) -> Tuple[List[str], List[str]]:
    """Draw up to ``n_per_class`` files per foreground class (1, 2, 3).

    A file is eligible for a class if that class appears anywhere in its
    damage mask. Because a single patch can satisfy more than one class,
    the resulting labeled set size is not simply ``3 * n_per_class`` --
    files are deduplicated via a set union across the three per-class
    draws.

    Returns:
        ``(labeled_files, unlabeled_files)``, both sorted.
    """
    rng = random.Random(seed)
    class_dict: Dict[int, List[str]] = defaultdict(list)

    for f in sorted(train_files):
        with np.load(f) as data:
            mask = data["damage_mask"]
        for cls in np.unique(mask):
            if cls != 0:
                class_dict[int(cls)].append(f)

    labeled_set: Set[str] = set()
    for cls in sorted(class_dict.keys()):
        flist = sorted(set(class_dict[cls]))
        rng.shuffle(flist)
        labeled_set.update(flist[: min(len(flist), n_per_class)])

    labeled_files = sorted(labeled_set)
    unlabeled_files = sorted(set(train_files) - labeled_set)
    return labeled_files, unlabeled_files


def split_target_data_fixed(
    files: Sequence[str], n_per_class: int, fixed_split_path: PathLike
) -> Tuple[List[str], List[str], List[str]]:
    """Combine the fixed train/val split with a per-class label-budget draw
    from the *train* portion.

    Returns:
        ``(labeled_files, unlabeled_files, val_files)``.
    """
    train_files, val_files = load_fixed_split(fixed_split_path)
    labeled_files, unlabeled_files = split_per_class_nested(train_files, n_per_class, seed=42)
    return labeled_files, unlabeled_files, val_files


# ---------------------------------------------------------------------------
# Sample weighting (used by WeightedRandomSampler)
# ---------------------------------------------------------------------------

def compute_rare_class_sample_weights(files: Sequence[str]) -> torch.DoubleTensor:
    """Per-file sampling weight: 6.0 if the file contains class 3 (destroyed),
    else 5.0 if it contains class 2 (damaged), else 1.0.

    Used for both the source and the target-labeled ``WeightedRandomSampler``
    (identical logic in the original notebook, unified here into one
    function).
    """
    weights = []
    for f in files:
        with np.load(f) as data:
            classes = set(np.unique(data["damage_mask"]).tolist())

        w = 1.0
        if 2 in classes:
            w = max(w, 5.0)
        if 3 in classes:
            w = max(w, 6.0)
        weights.append(w)

    return torch.DoubleTensor(weights)


# ---------------------------------------------------------------------------
# Dataloader construction
# ---------------------------------------------------------------------------

def build_dataloaders(
    source_root: PathLike,
    target_root: PathLike,
    batch_size: int,
    event_split_path: PathLike,
    fixed_split_path: PathLike,
    experiment_split_path: PathLike,
    target_labels_per_class: int,
    num_workers: int = 12,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], DataLoader]:
    """Build the source, target-labeled, target-unlabeled (always ``None``),
    and target-validation dataloaders for one training run.

    Splits are created on first use and reused (loaded from disk) on
    subsequent calls so that every label-budget experiment shares the same
    source/target event assignment and the same validation set.

    Args:
        source_root: directory of source-domain NPZ patches.
        target_root: directory of target-domain NPZ patches (may be the
            same directory as ``source_root`` for the optical-to-optical
            same-domain setup).
        batch_size: source-batch size; the target-labeled batch uses
            ``max(1, batch_size // 2)``.
        event_split_path: path to the cached source/target event split.
        fixed_split_path: path to the cached target train/val split.
        experiment_split_path: path to the cached labeled/unlabeled split
            for this specific ``target_labels_per_class`` budget.
        target_labels_per_class: per-class label budget for the target domain.
        num_workers: dataloader worker count.
        pin_memory: dataloader pin_memory flag.

    Returns:
        ``(source_loader, target_labeled_loader, target_unlabeled_loader, target_val_loader)``.
        ``target_unlabeled_loader`` is always ``None`` -- the active
        pipeline does not perform pseudo-label / semi-supervised training.
    """
    source_dataset = BuildingDamageDataset(source_root, is_source=True, augment=True)
    target_all_dataset = BuildingDamageDataset(target_root, is_source=False, augment=False)

    event_split_path = Path(event_split_path)
    if not event_split_path.exists():
        raise FileNotFoundError(
            f"Event split not found at {event_split_path}. "
            "Create it once with create_event_based_split(...)."
        )
    source_files, target_files = load_event_based_split(event_split_path)
    source_dataset.files = source_files
    target_all_dataset.files = target_files

    fixed_split_path = Path(fixed_split_path)
    if not fixed_split_path.exists():
        print("Creating fixed target train/val split...")
        create_fixed_split(target_all_dataset.files, val_ratio=0.3, save_path=fixed_split_path, seed=42)

    experiment_split_path = Path(experiment_split_path)
    if experiment_split_path.exists():
        data = np.load(experiment_split_path, allow_pickle=True)
        labeled_files = list(data["labeled"])
        unlabeled_files = list(data["unlabeled"])
        val_files = list(data["val"])
    else:
        random.seed(42)
        np.random.seed(42)
        labeled_files, unlabeled_files, val_files = split_target_data_fixed(
            target_all_dataset.files, target_labels_per_class, fixed_split_path=fixed_split_path
        )
        np.savez(experiment_split_path, labeled=labeled_files, unlabeled=unlabeled_files, val=val_files)

    print("===== SPLIT SANITY CHECK =====")
    print("source:", len(source_dataset.files))
    print("target_all:", len(target_all_dataset.files))
    print("labeled:", len(labeled_files))
    print("val:", len(val_files))
    print("source ∩ val:", len(set(source_dataset.files) & set(val_files)))
    print("source ∩ labeled:", len(set(source_dataset.files) & set(labeled_files)))
    print("labeled ∩ val:", len(set(labeled_files) & set(val_files)))
    print("===============================")

    target_labeled_dataset = BuildingDamageDataset(target_root, is_source=False, augment=True)
    target_labeled_dataset.files = labeled_files

    target_val_dataset = BuildingDamageDataset(target_root, is_source=False, augment=False)
    target_val_dataset.files = val_files

    source_weights = compute_rare_class_sample_weights(source_dataset.files)
    source_sampler = WeightedRandomSampler(
        weights=source_weights, num_samples=len(source_weights), replacement=True
    )
    source_loader = DataLoader(
        source_dataset,
        batch_size=batch_size,
        sampler=source_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    target_weights = compute_rare_class_sample_weights(target_labeled_dataset.files)
    target_sampler = WeightedRandomSampler(
        weights=target_weights, num_samples=len(target_weights), replacement=True
    )
    target_labeled_loader = DataLoader(
        target_labeled_dataset,
        batch_size=max(1, batch_size // 2),
        sampler=target_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    target_val_loader = DataLoader(
        target_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    target_unlabeled_loader = None
    return source_loader, target_labeled_loader, target_unlabeled_loader, target_val_loader
