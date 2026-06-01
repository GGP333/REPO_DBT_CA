"""
Fine-tuning trainer with Focal-Tversky loss for sim->real transfer on DBT.

Same warm-start fine-tuning as nnUNetTrainer_FT (LR 1e-3, seed 42), but the
Dice term of the loss is replaced by a Tversky term with alpha < beta so that
FALSE NEGATIVES are penalized harder than false positives. This pushes recall,
which is exactly what the failing real cases need (the missed tumors have
recall ~0). Optional focal exponent (gamma) emphasizes hard examples.

    Tversky index TI = tp / (tp + alpha*fp + beta*fn)
    loss = CE + (1 - TI)^gamma     (gamma=1 -> plain Tversky)

alpha=0.3, beta=0.7 is the standard recall-favoring setting.
"""
from __future__ import annotations

import random
from typing import Callable

import numpy as np
import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.utilities.helpers import softmax_helper_dim1


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FocalTverskyLoss(nn.Module):
    """Tversky loss mirroring SoftDiceLoss but with alpha/beta and focal gamma."""

    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True,
                 smooth: float = 1., ddp: bool = True, alpha: float = 0.3, beta: float = 0.7,
                 gamma: float = 1.0):
        super().__init__()
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        axes = [0] + list(range(2, len(shp_x))) if self.batch_dice else list(range(2, len(shp_x)))
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = tp
        denominator = tp + self.alpha * fp + self.beta * fn
        ti = (nominator + self.smooth) / (torch.clip(denominator + self.smooth, 1e-8))

        if not self.do_bg:
            ti = ti[1:] if self.batch_dice else ti[:, 1:]
        loss = torch.pow(1.0 - ti, self.gamma)
        return loss.mean()


class nnUNetTrainer_FT_Tversky(nnUNetTrainer):
    PAPER_SEED = 42
    FT_LR = 1e-3
    FT_EPOCHS = 80
    TVERSKY_ALPHA = 0.3
    TVERSKY_BETA = 0.7
    TVERSKY_GAMMA = 1.0  # set >1 for focal emphasis on hard cases

    def __init__(self, plans, configuration, fold, dataset_json, device=None):
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        _set_global_seed(self.PAPER_SEED)
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.initial_lr = self.FT_LR
        self.num_epochs = self.FT_EPOCHS

    def _build_loss(self):
        # CE + Tversky (recall-weighted), do_bg=False like the canonical trainer
        loss = DC_and_CE_loss(
            {'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5,
             'do_bg': False, 'ddp': self.is_ddp,
             'alpha': self.TVERSKY_ALPHA, 'beta': self.TVERSKY_BETA, 'gamma': self.TVERSKY_GAMMA},
            {}, weight_ce=1, weight_dice=1,
            ignore_label=self.label_manager.ignore_label, dice_class=FocalTverskyLoss)

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss
