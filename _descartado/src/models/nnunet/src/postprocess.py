# src/postprocess.py
"""
Post-procesamiento para nnU-Net:
  - Filtrado de componentes conectados pequeños
  - Selección automática de umbral min_size en validación
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    from scipy.ndimage import label as scipy_label
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


def remove_small_components(
    binary_mask: np.ndarray,
    min_size: int = 0,
) -> np.ndarray:
    """
    Elimina componentes conectados 3D menores a min_size voxeles.

    Parameters
    ----------
    binary_mask : ndarray (Z, H, W) uint8 o bool
    min_size : int
        Componentes con menos voxeles que esto se eliminan.
        Si 0, no filtra nada.

    Returns
    -------
    ndarray (Z, H, W) uint8 filtrada
    """
    if min_size <= 0 or not _SCIPY_AVAILABLE:
        return binary_mask.astype(np.uint8)

    mask = binary_mask.astype(bool)
    if not mask.any():
        return binary_mask.astype(np.uint8)

    labeled, n_components = scipy_label(mask)
    out = np.zeros_like(mask, dtype=np.uint8)

    for comp_id in range(1, n_components + 1):
        comp_mask = labeled == comp_id
        if comp_mask.sum() >= min_size:
            out[comp_mask] = 1

    return out


def keep_largest_component(binary_mask: np.ndarray) -> np.ndarray:
    """
    Mantiene solo el componente conectado más grande.
    Útil cuando se espera una sola lesión por volumen.
    """
    if not _SCIPY_AVAILABLE:
        return binary_mask.astype(np.uint8)

    mask = binary_mask.astype(bool)
    if not mask.any():
        return binary_mask.astype(np.uint8)

    labeled, n_components = scipy_label(mask)
    if n_components <= 1:
        return mask.astype(np.uint8)

    # Encontrar el componente más grande
    sizes = []
    for comp_id in range(1, n_components + 1):
        sizes.append((labeled == comp_id).sum())

    largest_id = np.argmax(sizes) + 1
    out = (labeled == largest_id).astype(np.uint8)
    return out


def _dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice score rápido entre dos máscaras binarias."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0 if not pred.any() and not gt.any() else 0.0
    return float(2.0 * inter / denom)


def find_optimal_min_size(
    predictions: list,
    ground_truths: list,
    candidates: Optional[list] = None,
) -> Tuple[int, float]:
    """
    Encuentra el min_size óptimo para remove_small_components
    evaluando el Dice sobre un conjunto de validación.

    Parameters
    ----------
    predictions : list[ndarray]
        Máscaras binarias predichas (Z,H,W).
    ground_truths : list[ndarray]
        Máscaras GT (Z,H,W).
    candidates : list[int], optional
        Valores de min_size a probar.

    Returns
    -------
    (best_min_size, best_dice)
    """
    if candidates is None:
        candidates = [0, 10, 25, 50, 100, 200, 500, 1000]

    best_min_size = 0
    best_dice = -1.0

    for ms in candidates:
        dices = []
        for pred, gt in zip(predictions, ground_truths):
            filtered = remove_small_components(pred, min_size=ms)
            dices.append(_dice_score(filtered, gt))
        mean_dice = float(np.mean(dices))

        if mean_dice > best_dice:
            best_dice = mean_dice
            best_min_size = ms

    return best_min_size, best_dice

