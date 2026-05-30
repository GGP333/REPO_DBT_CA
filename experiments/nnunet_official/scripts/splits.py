"""
Reproduce the repo's nnUNet train/val/test split exactly.

Mirrors `src/models/nnunet/train_unet_dbt.py::split_records` with the same
defaults used during the original experiments:
    train_ratio = 0.8
    val_ratio   = 0.2
    test_ratio  = 0.1
    seed        = 42

Special case (Dataset_Both_RealWorld):
    test = first 50% of real_dbt_* after shuffle (seed=42)
    train/val = (synth dbt_* + remaining real_dbt_*) shuffled, 80/20.
"""
from __future__ import annotations

import random
from typing import Dict, List


def _normalize_ratios(tr: float, va: float, te: float):
    total = tr + va + te
    if total > 1.0:
        tr /= total
        va /= total
        te /= total
    return tr, va, te


def compute_split(
    study_ids: List[str],
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
) -> Dict[str, List[str]]:
    """Return {"train": [...], "val": [...], "test": [...]} sorted alphabetically."""
    rng = random.Random(int(seed))
    real_ids = sorted([s for s in study_ids if s.startswith("real_dbt_")])
    synth_ids = sorted([s for s in study_ids if not s.startswith("real_dbt_")])

    if real_ids:
        rng.shuffle(real_ids)
        n_real = len(real_ids)
        n_test = max(1, n_real // 2)
        test_ids = set(real_ids[:n_test])
        real_for_trainval = real_ids[n_test:]

        trainval_ids = synth_ids + real_for_trainval
        rng.shuffle(trainval_ids)
        n_trainval = len(trainval_ids)

        tv_total = train_ratio + val_ratio
        adj_train = train_ratio / tv_total if tv_total > 0 else 0.8

        n_train = max(1, int(round(n_trainval * adj_train)))
        n_val = max(1, n_trainval - n_train)

        train_ids = set(trainval_ids[:n_train])
        val_ids = set(trainval_ids[n_train:n_train + n_val])
    else:
        train_ratio, val_ratio, test_ratio = _normalize_ratios(train_ratio, val_ratio, test_ratio)
        ids = sorted(study_ids)
        rng.shuffle(ids)
        n = len(ids)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        n_test = max(0, int(round(n * test_ratio)))
        while n_train + n_val + n_test > n:
            if n_train >= n_val and n_train > 1:
                n_train -= 1
            elif n_val >= n_test and n_val > 1:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1
            else:
                break
        train_ids = set(ids[:n_train])
        val_ids = set(ids[n_train:n_train + n_val])
        test_ids = set(ids[n_train + n_val:n_train + n_val + n_test])

    return {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
    }
