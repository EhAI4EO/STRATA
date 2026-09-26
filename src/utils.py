"""Small shared utilities: seeding, device selection, checkpoint I/O.

NOTE on seeding: the original notebook only seeds the *data split*
construction (``random.seed(42)`` / ``np.random.seed(42)`` inside
``build_dataloaders``). It does not seed model initialization, dataloader
shuffling, or the training loop itself. :func:`set_seed` below is an
**addition** made during this refactor (not present in the original code)
to make full runs reproducible end-to-end; it is called once at the start
of :func:`~src.trainer.train_one_experiment`. This is flagged here and in
the README as a deliberate deviation from "reproduce exactly," made
because the requirements explicitly ask for a documented random-seed
policy.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch

PathLike = Union[str, Path]


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + all CUDA devices).

    Addition beyond the original notebook -- see module docstring.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(gpu: str = "0") -> torch.device:
    """Return ``cuda`` if available, else ``cpu``.

    ``gpu`` is accepted for interface parity with the original notebook
    (which set ``CUDA_VISIBLE_DEVICES``); pass e.g. ``"0"`` or ``"0,1"``.
    """
    import os

    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(path: PathLike, **payload: Any) -> None:
    """Save a checkpoint dict to ``path``, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: PathLike, map_location: Union[str, torch.device] = "cpu") -> Dict[str, Any]:
    """Load a checkpoint dict, safe across devices (``weights_only=False`` to
    allow non-tensor metadata such as the stored metrics dict, matching the
    original notebook's checkpoints)."""
    return torch.load(str(path), map_location=map_location, weights_only=False)


def load_model_weights(model: torch.nn.Module, checkpoint: Dict[str, Any], strict: bool = True) -> None:
    """Load a model's ``state_dict`` from a checkpoint dict, tolerating both
    a raw state_dict and a dict with a ``"model"`` key (as saved by
    :func:`~src.trainer.train_one_experiment`)."""
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state, strict=strict)


def load_yaml_config(path: PathLike) -> Dict[str, Any]:
    """Load a YAML config, resolving a single-level ``extends: <file>.yaml``
    key relative to the same directory (used by ``configs/budget_*.yaml``
    to inherit from ``configs/base.yaml``)."""
    import yaml

    path = Path(path)
    with open(path, "r") as f:
        config = yaml.safe_load(f) or {}

    base_name = config.pop("extends", None)
    if base_name is not None:
        base_path = path.parent / base_name
        base_config = load_yaml_config(base_path)
        base_config.update(config)
        return base_config

    return config
