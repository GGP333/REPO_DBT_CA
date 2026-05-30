"""
nnUNetTrainer subclass standardized to the BMC manuscript training protocol.

What is overridden (paragraph 54 of the manuscript):
    - Optimizer:        AdamW (vs. SGD-Nesterov in the default)
    - Initial LR:       1e-3 (vs. 1e-2)
    - Weight decay:     1e-5 (vs. 3e-5)
    - Epochs:           500  (vs. 1000)
    - Warm-up:          5 epochs of linear warm-up before poly decay
    - Batch size:       1 forced (overrides whatever the planner picked)
    - Seed:             42 fixed (numpy + torch + python random)

What is intentionally kept canonical (Isensee defaults):
    - Patch-based sampling with foreground oversampling 0.33 (matches the paper)
    - Deep supervision with per-stage weighted Dice+CE loss
    - Mirror TTA at validation/inference
    - The full batchgenerators augmentation pipeline
    - nnUNetv2's planner-derived patch size and channel widths

NOTE: this trainer deviates from the "vanilla" nnU-Net of Isensee et al. and
exists ONLY to satisfy the paper's "all models trained with consistent
experimental setup" statement. If you want the canonical nnU-Net (the one
recommended by Isensee), use the default `nnUNetTrainer` instead.
"""
from __future__ import annotations

import random

import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class PolyLRSchedulerWithWarmup(_LRScheduler):
    """Linear warm-up for `warmup_epochs` then polynomial decay."""

    def __init__(self, optimizer, initial_lr: float, max_epochs: int,
                 warmup_epochs: int = 5, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        if current_step < self.warmup_epochs:
            # linear warm-up: start near 0, reach initial_lr at warmup_epochs
            new_lr = self.initial_lr * (current_step + 1) / max(1, self.warmup_epochs)
        else:
            # poly decay over the remaining (max_epochs - warmup_epochs) epochs
            t = current_step - self.warmup_epochs
            T = max(1, self.max_epochs - self.warmup_epochs)
            new_lr = self.initial_lr * (1 - t / T) ** self.exponent
        new_lr = max(new_lr, 0.0)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

        self._last_lr = [g["lr"] for g in self.optimizer.param_groups]

    def get_last_lr(self):
        return self._last_lr


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class nnUNetTrainer_PaperStandardized(nnUNetTrainer):
    PAPER_SEED = 42
    PAPER_BATCH_SIZE = 1
    PAPER_LR = 1e-3
    PAPER_WD = 1e-5
    PAPER_EPOCHS = 500
    PAPER_WARMUP = 5

    def __init__(self, plans, configuration, fold, dataset_json, device=None):
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        _set_global_seed(self.PAPER_SEED)
        super().__init__(plans, configuration, fold, dataset_json, device=device)

        # Hyperparameters dictated by the manuscript (paragraph 54)
        self.initial_lr = self.PAPER_LR
        self.weight_decay = self.PAPER_WD
        self.num_epochs = self.PAPER_EPOCHS

        # Force batch size = 1 by mutating the configuration the trainer reads from
        cfg = self.configuration_manager.configuration
        cfg["batch_size"] = self.PAPER_BATCH_SIZE

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
        )
        lr_scheduler = PolyLRSchedulerWithWarmup(
            optimizer,
            initial_lr=self.initial_lr,
            max_epochs=self.num_epochs,
            warmup_epochs=self.PAPER_WARMUP,
            exponent=0.9,
        )
        return optimizer, lr_scheduler
