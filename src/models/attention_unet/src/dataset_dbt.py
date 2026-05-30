# src/dataset_dbt.py
from __future__ import annotations
import os
import sys
import math
import random
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from io_utils import discover_studies, get_logger, StudyRecord

# Importar augmentaciones desde Models/shared/
_shared_dir = str(Path(__file__).resolve().parents[2] / "shared")
if _shared_dir not in sys.path:
    sys.path.insert(0, _shared_dir)
from augmentations import apply_augmentations

def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def robust_minmax(x: np.ndarray, p1: float = 1.0, p99: float = 99.0) -> Tuple[np.ndarray, float, float]:
    """
    Normalización por percentiles (robusta a outliers). Si hay muchos ceros,
    ignora ceros si existen valores > 0.
    """
    x = x.astype(np.float32, copy=False)
    flat = x.ravel()
    nonzero = flat[flat != 0]
    ref = nonzero if nonzero.size > 0 else flat
    lo = np.percentile(ref, p1)
    hi = np.percentile(ref, p99)
    if hi <= lo:
        hi = ref.max() if ref.size else 1.0
        lo = ref.min() if ref.size else 0.0
    x = (x - lo) / (hi - lo + 1e-6)
    x = np.clip(x, 0.0, 1.0)
    return x, float(lo), float(hi)

@dataclass
class Item:
    image: torch.Tensor  # (1,Z,H,W), float32
    mask: torch.Tensor   # (1,Z,H,W), float32 {0,1}
    study_id: str
    img_path: str
    mask_path: str
    shape: Tuple[int, int, int]  # (Z,H,W)
    norm_bounds: Tuple[float, float]  # (lo, hi)

class DBTVolumeDataset(Dataset):
    """
    Dataset 3D que empareja volúmenes img/mask en (Z,H,W), normaliza a [0,1],
    y retorna tensores (1,Z,H,W). Soporta augmentación y pre-cómputo de foreground.
    """
    def __init__(self, records: List[StudyRecord], logger: Optional[logging.Logger] = None,
                 augment: bool = False, cfg=None):
        self.records = records
        self.logger = logger or get_logger("dbt_dataset")
        self.augment = augment
        self.cfg = cfg
        # Pre-computar flag de foreground para sampling
        self.has_foreground: List[bool] = []
        for rec in records:
            try:
                m = np.load(rec.mask_path)
                self.has_foreground.append(bool((m > 0).any()))
            except Exception:
                self.has_foreground.append(False)
        n_fg = sum(self.has_foreground)
        self.logger.info(f"Dataset: {len(records)} estudios, {n_fg} con foreground ({n_fg/max(1,len(records))*100:.1f}%)")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Item:
        rec = self.records[idx]
        img = np.load(rec.img_path)  # (Z,H,W) esperado
        mask = np.load(rec.mask_path)

        if img.ndim != 3:
            self.logger.warning(f"[{rec.study_id}] Imagen con ndim={img.ndim}, intentando squeeze.")
            img = np.squeeze(img)
        if mask.ndim != 3:
            self.logger.warning(f"[{rec.study_id}] Máscara con ndim={mask.ndim}, intentando squeeze.")
            mask = np.squeeze(mask)

        if img.shape != mask.shape:
            z = min(img.shape[0], mask.shape[0])
            h = min(img.shape[1], mask.shape[1])
            w = min(img.shape[2], mask.shape[2])
            self.logger.warning(f"[{rec.study_id}] Shape mismatch img{img.shape} vs mask{mask.shape}. Cropping centrado a {(z,h,w)}.")
            def center_crop(x, target):
                sz, sh, sw = x.shape
                cz = (sz - target[0])//2
                ch = (sh - target[1])//2
                cw = (sw - target[2])//2
                return x[cz:cz+target[0], ch:ch+target[1], cw:cw+target[2]]
            img = center_crop(img, (z,h,w))
            mask = center_crop(mask, (z,h,w))

        img, lo, hi = robust_minmax(img)
        mask = (mask > 0).astype(np.float32)

        # Augmentación on-the-fly (solo en entrenamiento)
        if self.augment:
            img, mask = apply_augmentations(img, mask, self.cfg)

        # Expandir a (1,Z,H,W)
        img_t = torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0).float()
        mask_t = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0).float()

        return Item(
            image=img_t,
            mask=mask_t,
            study_id=rec.study_id,
            img_path=rec.img_path,
            mask_path=rec.mask_path,
            shape=img.shape,
            norm_bounds=(lo, hi),
        )


def build_foreground_sampler(dataset: DBTVolumeDataset, oversample_ratio: float = 0.5) -> WeightedRandomSampler:
    """
    Crea un WeightedRandomSampler que sobrerepresenta volúmenes con foreground.
    oversample_ratio=0.5 significa que ~50% de los samples tendrán foreground.
    """
    n = len(dataset)
    n_fg = sum(dataset.has_foreground)
    n_bg = n - n_fg

    if n_fg == 0 or n_bg == 0:
        return None

    w_fg = oversample_ratio / n_fg
    w_bg = (1.0 - oversample_ratio) / n_bg
    weights = [w_fg if fg else w_bg for fg in dataset.has_foreground]

    return WeightedRandomSampler(weights, num_samples=n, replacement=True)

def _pad_to_shape(x: torch.Tensor, target_shape: Tuple[int, int, int]) -> torch.Tensor:
    if x.ndim == 3:
        x = x.unsqueeze(0)
    _, Z, H, W = x.shape
    Zt, Ht, Wt = target_shape
    pad = (0, Wt - W, 0, Ht - H, 0, Zt - Z)
    pad = tuple(max(0, p) for p in pad)
    return torch.nn.functional.pad(x, pad)

def collate_pad(batch: List[Item]) -> Dict[str, Any]:
    Zmax = max(it.image.shape[1] for it in batch)
    Hmax = max(it.image.shape[2] for it in batch)
    Wmax = max(it.image.shape[3] for it in batch)
    def round_up(x, m=8):
        return ((x + m - 1) // m) * m
    Zt, Ht, Wt = round_up(Zmax, 8), round_up(Hmax, 8), round_up(Wmax, 8)
    imgs = torch.stack([_pad_to_shape(it.image, (Zt,Ht,Wt)) for it in batch], dim=0)
    masks = torch.stack([_pad_to_shape(it.mask, (Zt,Ht,Wt)) for it in batch], dim=0)
    study_ids = [it.study_id for it in batch]
    shapes = [it.shape for it in batch]
    norm_bounds = [it.norm_bounds for it in batch]
    paths = [(it.img_path, it.mask_path) for it in batch]
    return {"image": imgs, "mask": masks, "study_ids": study_ids, "shapes": shapes, "norm_bounds": norm_bounds, "paths": paths}

def make_dataloaders(cfg) -> Tuple[DataLoader, DataLoader, List[StudyRecord], List[StudyRecord]]:
    logger = get_logger("dbt_dataset")
    set_global_seeds(int(cfg.seed))

    records = discover_studies(cfg.dataset_root, logger=logger)
    if not records:
        raise RuntimeError("No se encontraron estudios válidos.")

    ids = sorted([r.study_id for r in records])
    random.Random(int(cfg.seed)).shuffle(ids)
    n_total = len(ids)
    n_val = max(1, int(round(n_total * float(cfg.val_ratio))))
    val_ids = set(ids[:n_val])
    train_records = [r for r in records if r.study_id not in val_ids]
    val_records = [r for r in records if r.study_id in val_ids]

    if not train_records or not val_records:
        logger.warning(f"Split resultó en train={len(train_records)} val={len(val_records)}.")

    use_aug = bool(getattr(cfg, "augmentation", False))
    tr_ds = DBTVolumeDataset(train_records, logger=logger, augment=use_aug, cfg=cfg)
    va_ds = DBTVolumeDataset(val_records, logger=logger, augment=False, cfg=cfg)

    fg_ratio = float(getattr(cfg, "foreground_oversample", 0.0))
    sampler = None
    if fg_ratio > 0:
        sampler = build_foreground_sampler(tr_ds, oversample_ratio=fg_ratio)
        if sampler:
            logger.info(f"Foreground oversampling activado: ratio={fg_ratio}")

    tr_loader = DataLoader(
        tr_ds, batch_size=int(cfg.batch_size), shuffle=(sampler is None),
        sampler=sampler, num_workers=int(cfg.num_workers), pin_memory=True, collate_fn=collate_pad,
    )
    va_loader = DataLoader(
        va_ds, batch_size=int(cfg.batch_size), shuffle=False,
        num_workers=int(cfg.num_workers), pin_memory=True, collate_fn=collate_pad,
    )
    return tr_loader, va_loader, train_records, val_records
