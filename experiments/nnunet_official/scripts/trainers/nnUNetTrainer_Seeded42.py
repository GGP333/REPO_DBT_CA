"""
Canonical nnU-Net trainer (Isensee defaults) with explicit seed=42.

Keeps everything from the default `nnUNetTrainer`:
    - SGD + Nesterov momentum 0.99
    - lr=1e-2, wd=3e-5
    - 1000 epochs
    - Planner-decided batch size
    - Deep supervision, mirror TTA, full augmentation pipeline

The only difference vs. `nnUNetTrainer` is that random/numpy/torch are seeded
to 42 before initialization. Use this if you want the canonical nnU-Net of
Isensee et al. but still want bit-reproducibility with seed=42 (matches the
paper's "fixed random seed" claim in paragraph 56).
"""
from __future__ import annotations

import random

import numpy as np
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class nnUNetTrainer_Seeded42(nnUNetTrainer):
    PAPER_SEED = 42

    def __init__(self, plans, configuration, fold, dataset_json, device=None):
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        _set_global_seed(self.PAPER_SEED)
        super().__init__(plans, configuration, fold, dataset_json, device=device)
