"""
Canonical Isensee nnU-Net trainer (lr=1e-2, SGD-Nesterov 0.99, wd=3e-5,
deep supervision, etc.) + seed 42 + **early stopping on EMA pseudo-Dice
plateau**.

Stopping rule:
    if current_epoch >= MIN_EPOCHS and
       (current EMA pseudo-Dice has not improved by > MIN_DELTA in the last
        PATIENCE epochs):
        stop.

We never stop before MIN_EPOCHS so that the EMA has time to stabilize. We
operate on `ema_fg_dice` (what nnunetv2 already tracks for `checkpoint_best`),
so "best by val Dice" semantics are preserved.

The reported model is always `checkpoint_best.pth`, exactly like with the
no-early-stopping trainer. The only thing that changes is how many wasted
epochs you avoid after the plateau.
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


class nnUNetTrainer_Seeded42_ES(nnUNetTrainer):
    PAPER_SEED = 42

    # Early-stopping knobs
    ES_PATIENCE = 50          # epochs without EMA improvement to trigger stop
    ES_MIN_DELTA = 1e-4       # minimum improvement to count as "better"
    ES_MIN_EPOCHS = 100       # never stop before this epoch

    def __init__(self, plans, configuration, fold, dataset_json, device=None):
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        _set_global_seed(self.PAPER_SEED)
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self._es_best_ema = -float("inf")
        self._es_stale_epochs = 0

    def on_epoch_end(self):
        super().on_epoch_end()  # also advances self.current_epoch by 1

        current_ema = float(self.logger.get_value("ema_fg_dice", step=-1))
        if current_ema > self._es_best_ema + self.ES_MIN_DELTA:
            self._es_best_ema = current_ema
            self._es_stale_epochs = 0
        else:
            self._es_stale_epochs += 1

        # self.current_epoch was already incremented by super().on_epoch_end
        # so "current_epoch" here is actually the NEXT one to run.
        completed_epoch = self.current_epoch - 1

        if (completed_epoch >= self.ES_MIN_EPOCHS
                and self._es_stale_epochs >= self.ES_PATIENCE):
            self.print_to_log_file(
                f"Early stopping at epoch {completed_epoch}: "
                f"EMA pseudo-Dice has not improved by > {self.ES_MIN_DELTA} "
                f"in the last {self.ES_PATIENCE} epochs "
                f"(best EMA = {self._es_best_ema:.4f}). "
                f"Saving checkpoint_final.pth and stopping."
            )
            # Force the outer for-loop in run_training to exit at the next check
            self.num_epochs = self.current_epoch
