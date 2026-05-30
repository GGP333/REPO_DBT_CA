#!/usr/bin/env python3
"""
Preprocesado SIMPLE de stacks TIFF de DBT:
- Carga el volumen TIFF
- Normalización de intensidades (z-score o minmax)
- Resize XY a tamaño fijo (512x256 por defecto)
- Guarda como .npy

NO hace: segmentación de seno, crop, histogram matching global
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import SimpleITK as sitk


# ----------------------------- Utilidades ----------------------------- #
def _natural_key(path: Path) -> List[object]:
    """Ordena nombres con partes numéricas (slice_1, slice_2, slice_10)."""
    import re
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", path.name)
    ]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ----------------------------- E/S de imagen ----------------------------- #
def load_tiff_stack(folder: Path) -> np.ndarray:
    """Carga un stack de imágenes TIFF en orden z -> numpy (z, y, x)."""
    if not folder.is_dir():
        raise FileNotFoundError(f"input_dir no existe o no es carpeta: {folder}")
    tiffs = sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in {".tif", ".tiff"}],
        key=_natural_key
    )
    if not tiffs:
        raise FileNotFoundError(f"No se encontraron .tif/.tiff en {folder}")

    try:
        import tifffile
    except ImportError as exc:
        raise RuntimeError("Instala 'tifffile' para leer stacks TIFF") from exc

    slices = [tifffile.imread(str(p)) for p in tiffs]
    arr = np.stack(slices, axis=0)
    return arr.astype(np.float32)


def ensure_volume_3d(vol: np.ndarray) -> np.ndarray:
    """Garantiza shape (z,y,x). Si llega (1,z,y,x) o (z,y,x,1), compacta."""
    if vol.ndim == 4:
        if vol.shape[0] == 1:
            vol = vol[0]
        elif vol.shape[-1] == 1:
            vol = vol[..., 0]
    if vol.ndim != 3:
        raise ValueError(f"Volumen esperado 3D, shape recibido {vol.shape}")
    return vol


def infer_mask_dir_from_input(input_dir: Path) -> Path:
    """Busca una carpeta mask_* hermana de input_dir (que suele ser img_*)."""
    parent = input_dir.parent
    candidates = [p for p in parent.iterdir() if p.is_dir() and p.name.lower().startswith("mask")]
    if not candidates:
        raise FileNotFoundError(f"No se encontró carpeta de máscara (mask_*) junto a {input_dir}")
    if len(candidates) > 1:
        raise ValueError(f"Se encontraron múltiples carpetas de máscara en {parent}: {candidates}")
    return candidates[0]


# ----------------------------- Normalización ----------------------------- #
def normalize_volume(
    volume: np.ndarray,
    clip_percentiles: Tuple[float, float] = (0.5, 99.5),
    mode: str = "zscore",
) -> np.ndarray:
    """
    Clipping + normalización.
    - zscore: (x - mean) / std
    - minmax: (x - min) / (max - min) -> [0, 1]
    - none: solo clipping
    """
    v = volume.astype(np.float32)
    
    # Clipping por percentiles
    lo, hi = np.percentile(v, clip_percentiles)
    if lo < hi:
        v = np.clip(v, lo, hi)

    if mode == "zscore":
        mean = float(v.mean())
        std = float(v.std())
        if std > 0:
            v = (v - mean) / std
        else:
            v = v - mean
    elif mode == "minmax":
        vmin = float(v.min())
        vmax = float(v.max())
        rng = vmax - vmin
        if rng > 0:
            v = (v - vmin) / rng
        else:
            v = v * 0
    elif mode != "none":
        raise ValueError("normalize debe ser zscore, minmax o none")

    return v


# ----------------------------- Resize XY ----------------------------- #
def resize_xy(
    volume: np.ndarray,
    target_size_xy: Tuple[int, int],
    interpolator: int = sitk.sitkLinear,
) -> np.ndarray:
    """
    Redimensiona el plano XY a target_size_xy conservando el número de slices Z.
    """
    if volume.ndim != 3:
        raise ValueError(f"Se esperaba un volumen 3D (z,y,x); shape recibido {volume.shape}")

    vz, vy, vx = volume.shape
    xt, yt = int(target_size_xy[0]), int(target_size_xy[1])

    img = sitk.GetImageFromArray(volume)
    # Spacing dummy (1,1,1) para el input
    img.SetSpacing((1.0, 1.0, 1.0))

    # Output spacing escalado para cubrir toda la extensión de la imagen
    # original y así hacer un downsample real (no un crop).
    output_spacing = (float(vx) / xt, float(vy) / yt, 1.0)

    resampled = sitk.Resample(
        img,
        size=(xt, yt, vz),
        transform=sitk.Transform(),
        interpolator=interpolator,
        outputOrigin=img.GetOrigin(),
        outputSpacing=output_spacing,
        outputDirection=img.GetDirection(),
        defaultPixelValue=0.0,
    )
    return sitk.GetArrayFromImage(resampled)


# ----------------------------- Config ----------------------------- #
@dataclass
class Params:
    input_dir: str
    output_dir: str
    mask_dir: str
    original_shape: Tuple[int, int, int]
    target_size_xy: Tuple[int, int]
    clip_percentiles: Tuple[float, float]
    normalize: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preprocesado SIMPLE: normalización + resize XY (sin segmentación/crop)"
    )
    p.add_argument("--input_dir", required=True, type=Path, 
                   help="Carpeta con slices .tif/.tiff")
    p.add_argument("--output_dir", required=True, type=Path,
                   help="Carpeta destino para volúmenes procesados")
    p.add_argument("--mask_dir", type=Path,
                   help="Carpeta con máscara GT. Si se omite, busca mask_* junto a input_dir")
    p.add_argument("--target_size_xy", nargs=2, type=int, default=[512, 256],
                   help="Tamaño final en voxeles (x y)")
    p.add_argument("--clip_percentiles", nargs=2, type=float, default=[0.5, 99.5],
                   help="Percentiles para clipping")
    p.add_argument("--normalize", choices=["zscore", "minmax", "none"], default="zscore",
                   help="Modo de normalización")
    return p.parse_args()


# ----------------------------- Main ----------------------------- #
def main() -> None:
    args = parse_args()

    target_size_xy = tuple(int(x) for x in args.target_size_xy)
    clip_percentiles = tuple(float(x) for x in args.clip_percentiles)

    # Crear carpeta de salida
    ensure_dir(args.output_dir)

    # Cargar imagen
    print(f"[INFO] Cargando stack TIFF desde {args.input_dir}")
    vol = load_tiff_stack(args.input_dir)
    vol = ensure_volume_3d(vol)
    original_shape = vol.shape
    print(f"[INFO] Shape original (z,y,x): {original_shape}")

    # Cargar máscara GT
    mask_dir = args.mask_dir or infer_mask_dir_from_input(args.input_dir)
    print(f"[INFO] Cargando máscara GT desde {mask_dir}")
    gt_mask = load_tiff_stack(mask_dir)
    gt_mask = ensure_volume_3d(gt_mask).astype(np.uint8)
    if gt_mask.shape != vol.shape:
        raise ValueError(f"Shape de máscara {gt_mask.shape} no coincide con imagen {vol.shape}")

    # Normalizar
    print(f"[INFO] Normalizando con clip={clip_percentiles}, modo={args.normalize}")
    vol_norm = normalize_volume(vol, clip_percentiles, mode=args.normalize)

    # Resize XY
    print(f"[INFO] Redimensionando XY a {target_size_xy}")
    vol_final = resize_xy(vol_norm, target_size_xy, interpolator=sitk.sitkLinear)
    gt_mask_final = resize_xy(
        gt_mask.astype(np.float32), target_size_xy, interpolator=sitk.sitkNearestNeighbor
    )
    gt_mask_final = (gt_mask_final > 0.5).astype(np.uint8)

    print(f"[INFO] Shape final: {vol_final.shape}")

    # Guardar
    study_id = args.input_dir.parent.name  # dbt_XXX
    img_out = args.output_dir / f"{study_id}_img.npy"
    mask_out = args.output_dir / f"{study_id}_mask.npy"

    np.save(img_out, vol_final.astype(np.float32))
    np.save(mask_out, gt_mask_final.astype(np.uint8))
    print(f"[INFO] Guardado: {img_out}")
    print(f"[INFO] Guardado: {mask_out}")

    # Guardar parámetros
    params = Params(
        input_dir=str(args.input_dir),
        output_dir=str(args.output_dir),
        mask_dir=str(mask_dir),
        original_shape=original_shape,
        target_size_xy=target_size_xy,
        clip_percentiles=clip_percentiles,
        normalize=args.normalize,
    )
    with open(args.output_dir / "preprocess_params.json", "w", encoding="utf-8") as f:
        json.dump(asdict(params), f, indent=2)


if __name__ == "__main__":
    main()

