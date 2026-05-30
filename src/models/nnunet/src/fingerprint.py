# src/fingerprint.py
"""
Dataset Fingerprint: analiza todos los volúmenes .npy del dataset
y extrae estadísticas para la auto-configuración de nnU-Net.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from io_utils import StudyRecord, get_logger


def compute_fingerprint(
    records: List[StudyRecord],
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Escanea todos los volúmenes del dataset y calcula un fingerprint estadístico.

    Parameters
    ----------
    records : list[StudyRecord]
        Lista de estudios descubiertos (img + mask).
    spacing : tuple
        Espaciado físico (Z, Y, X). Por defecto (1,1,1) para datos ya resampleados.
    logger : Logger, optional

    Returns
    -------
    dict  con claves:
        - shapes           : lista de (Z,H,W) por estudio
        - median_shape     : (Z,H,W) mediana del dataset
        - min_shape        : (Z,H,W) mínima
        - max_shape        : (Z,H,W) máxima
        - spacing          : (sz,sy,sx)
        - foreground_ratios: lista de float (ratio fg/total por estudio)
        - median_fg_ratio  : float
        - intensity_stats  : dict con mean, std, p0.5, p99.5 globales
        - num_cases        : int
    """
    logger = logger or get_logger("fingerprint")

    if not records:
        raise ValueError("No hay estudios para analizar.")

    shapes: List[Tuple[int, int, int]] = []
    fg_ratios: List[float] = []
    all_means: List[float] = []
    all_stds: List[float] = []
    all_lo: List[float] = []
    all_hi: List[float] = []

    logger.info(f"Analizando {len(records)} volúmenes para fingerprint...")

    for i, rec in enumerate(records):
        img = np.load(rec.img_path)
        mask = np.load(rec.mask_path)

        # Compactar dims extra
        if img.ndim == 4:
            img = img.squeeze()
        if mask.ndim == 4:
            mask = mask.squeeze()

        if img.ndim != 3:
            logger.warning(f"[{rec.study_id}] img ndim={img.ndim}, saltando.")
            continue

        shapes.append(tuple(img.shape))  # (Z, H, W)

        # Foreground ratio
        fg_voxels = int((mask > 0).sum())
        total_voxels = int(mask.size)
        fg_ratios.append(fg_voxels / max(1, total_voxels))

        # Estadísticas de intensidad
        flat = img.astype(np.float64)
        all_means.append(float(flat.mean()))
        all_stds.append(float(flat.std()))
        all_lo.append(float(np.percentile(flat, 0.5)))
        all_hi.append(float(np.percentile(flat, 99.5)))

        if (i + 1) % 20 == 0 or (i + 1) == len(records):
            logger.info(f"  [{i+1}/{len(records)}] procesados")

    shapes_arr = np.array(shapes)  # (N, 3)

    median_shape = tuple(int(x) for x in np.median(shapes_arr, axis=0))
    min_shape = tuple(int(x) for x in shapes_arr.min(axis=0))
    max_shape = tuple(int(x) for x in shapes_arr.max(axis=0))

    fingerprint: Dict[str, Any] = {
        "num_cases": len(shapes),
        "spacing": list(spacing),
        "shapes": [list(s) for s in shapes],
        "median_shape": list(median_shape),
        "min_shape": list(min_shape),
        "max_shape": list(max_shape),
        "foreground_ratios": fg_ratios,
        "median_fg_ratio": float(np.median(fg_ratios)),
        "intensity_stats": {
            "global_mean": float(np.mean(all_means)),
            "global_std": float(np.mean(all_stds)),
            "global_p0.5": float(np.mean(all_lo)),
            "global_p99.5": float(np.mean(all_hi)),
        },
    }

    logger.info(
        f"Fingerprint: {len(shapes)} casos | "
        f"median_shape={median_shape} | "
        f"fg_ratio={fingerprint['median_fg_ratio']:.6f}"
    )

    return fingerprint


def save_fingerprint(fingerprint: Dict[str, Any], out_dir: str) -> str:
    """Guarda el fingerprint como JSON. Devuelve la ruta al archivo."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fp_path = out_path / "fingerprint.json"
    with open(fp_path, "w", encoding="utf-8") as f:
        json.dump(fingerprint, f, indent=2)
    return str(fp_path)

