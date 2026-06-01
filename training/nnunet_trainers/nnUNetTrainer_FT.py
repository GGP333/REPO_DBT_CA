"""
Fine-tuning trainer for sim->real transfer on DBT (Dataset005_DBTRealFT).

Used together with `nnUNetv2_train ... -pretrained_weights <D003 checkpoint>`:
the network is initialized from the synthetic-pretrained model (Dataset003)
and fine-tuned on the 10 real development cases.

Deviations from the canonical nnUNetTrainer:
    - Initial LR:  1e-3  (10x lower than the 1e-2 default -> standard fine-tuning)
    - Epochs:      150   (vs 1000; warm-start converges fast on 10 cases)
    - Seed:        42 fixed (python/numpy/torch)

Everything else stays canonical (SGD-Nesterov, Dice+CE deep supervision,
mirror TTA, full augmentation pipeline). checkpoint_best.pth is selected by
EMA pseudo-Dice on the fold's validation split, so no manual epoch picking is
needed.
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


class nnUNetTrainer_FT(nnUNetTrainer):
    PAPER_SEED = 42
    FT_LR = 1e-3
    FT_EPOCHS = 150

    def __init__(self, plans, configuration, fold, dataset_json, device=None):
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        _set_global_seed(self.PAPER_SEED)
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.initial_lr = self.FT_LR
        self.num_epochs = self.FT_EPOCHS
