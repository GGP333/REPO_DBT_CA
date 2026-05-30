#!/usr/bin/env python3
"""
Resumen de dimensiones para datasets crudos (TIFF) y preprocesados (NPY).

Ejemplos:
  python scripts/summarize_dataset_dimensions.py --dataset_root Dataset
  python scripts/summarize_dataset_dimensions.py --preprocessed_root Dataset_Preprocessed
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ShapeStats:
    count: int
    min_shape: Tuple[int, int, int]
    median_shape: Tuple[int, int, int]
    max_shape: Tuple[int, int, int]
    unique_shapes: int


@dataclass(frozen=True)
class ShapeSummary:
    total_cases: int
    missing_img: int
    missing_mask: int
    mismatched_img_mask: int
    stats: Optional[ShapeStats]


def _natural_key(text: str) -> List[object]:
    import re

    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", text)]


def _safe_shape(shape: Tuple[int, ...]) -> Tuple[int, int, int]:
    if len(shape) == 2:
        return (1, int(shape[0]), int(shape[1]))
    if len(shape) == 3:
        return (int(shape[0]), int(shape[1]), int(shape[2]))
    raise ValueError(f"Shape no soportado: {shape}")


def _tiff_shape(path: Path) -> Tuple[int, int, int]:
    try:
        import tifffile
    except ImportError as exc:
        raise RuntimeError("Instala 'tifffile' para leer TIFF") from exc

    with tifffile.TiffFile(str(path)) as tif:
        if not tif.series:
            raise ValueError(f"TIFF sin series: {path}")
        shape = tif.series[0].shape
    return _safe_shape(shape)


def _npy_shape(path: Path) -> Tuple[int, int, int]:
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 3:
        raise ValueError(f"NPY esperado 3D, shape recibido {arr.shape} ({path})")
    return (int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2]))


def _summarize_shapes(shapes: List[Tuple[int, int, int]]) -> Optional[ShapeStats]:
    if not shapes:
        return None
    arr = np.array(shapes, dtype=np.int64)
    min_shape = tuple(int(x) for x in arr.min(axis=0))
    max_shape = tuple(int(x) for x in arr.max(axis=0))
    median_shape = tuple(int(x) for x in np.median(arr, axis=0))
    unique_shapes = len({tuple(s) for s in shapes})
    return ShapeStats(
        count=len(shapes),
        min_shape=min_shape,
        median_shape=median_shape,
        max_shape=max_shape,
        unique_shapes=unique_shapes,
    )


def _print_summary(title: str, summary: ShapeSummary) -> None:
    print(f"\n[{title}]")
    print(
        "  casos=%d | img_faltante=%d | mask_faltante=%d | img_mask_mismatch=%d"
        % (summary.total_cases, summary.missing_img, summary.missing_mask, summary.mismatched_img_mask)
    )
    if summary.stats is None:
        print("  sin datos de shape")
        return
    stats = summary.stats
    print(
        "  shapes z,y,x | min=%s | mediana=%s | max=%s | unicos=%d"
        % (stats.min_shape, stats.median_shape, stats.max_shape, stats.unique_shapes)
    )


def _find_subsets(dataset_root: Path) -> List[Path]:
    subsets: List[Path] = []
    if any(p.is_dir() and _is_study_dir(p.name) for p in dataset_root.iterdir()):
        subsets.append(dataset_root)
    for child in sorted([p for p in dataset_root.iterdir() if p.is_dir()], key=lambda p: _natural_key(p.name)):
        if any(p.is_dir() and _is_study_dir(p.name) for p in child.iterdir()):
            subsets.append(child)
    return subsets


def _is_study_dir(name: str) -> bool:
    """Detecta carpetas de estudio: dbt_* o real_dbt_*"""
    return name.startswith("dbt_") or name.startswith("real_dbt_")


def _summarize_raw_dataset(subset_root: Path) -> ShapeSummary:
    study_dirs = sorted(
        [p for p in subset_root.iterdir() if p.is_dir() and _is_study_dir(p.name)],
        key=lambda p: _natural_key(p.name),
    )
    total = len(study_dirs)
    missing_img = 0
    missing_mask = 0
    mismatched = 0
    shapes: List[Tuple[int, int, int]] = []

    for sd in study_dirs:
        img_dir = next((p for p in sd.iterdir() if p.is_dir() and p.name.startswith("img_")), None)
        mask_dir = next((p for p in sd.iterdir() if p.is_dir() and p.name.startswith("mask_")), None)
        if img_dir is None:
            missing_img += 1
            continue
        img_files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {".tif", ".tiff"}])
        if not img_files:
            missing_img += 1
            continue
        img_shape = _tiff_shape(img_files[0])
        shapes.append(img_shape)

        if mask_dir is None:
            missing_mask += 1
            continue
        mask_files = sorted([p for p in mask_dir.iterdir() if p.suffix.lower() in {".tif", ".tiff"}])
        if not mask_files:
            missing_mask += 1
            continue
        mask_shape = _tiff_shape(mask_files[0])
        if mask_shape != img_shape:
            mismatched += 1

    return ShapeSummary(
        total_cases=total,
        missing_img=missing_img,
        missing_mask=missing_mask,
        mismatched_img_mask=mismatched,
        stats=_summarize_shapes(shapes),
    )


def _summarize_preprocessed(root: Path) -> Dict[str, ShapeSummary]:
    summaries: Dict[str, ShapeSummary] = {}
    img_paths = sorted([p for p in root.rglob("*_img.npy") if p.is_file()], key=lambda p: _natural_key(str(p)))
    if not img_paths:
        summaries["(root)"] = ShapeSummary(0, 0, 0, 0, None)
        return summaries

    buckets: Dict[str, List[Path]] = {}
    for img in img_paths:
        rel = img.relative_to(root)
        key = rel.parts[0] if len(rel.parts) > 1 else "(root)"
        buckets.setdefault(key, []).append(img)

    for bucket, imgs in buckets.items():
        total = len(imgs)
        missing_img = 0
        missing_mask = 0
        mismatched = 0
        shapes: List[Tuple[int, int, int]] = []

        for img in imgs:
            if not img.exists():
                missing_img += 1
                continue
            case_id = img.name[:-8]
            mask = img.parent / f"{case_id}_mask.npy"
            img_shape = _npy_shape(img)
            shapes.append(img_shape)
            if not mask.exists():
                missing_mask += 1
                continue
            mask_shape = _npy_shape(mask)
            if mask_shape != img_shape:
                mismatched += 1

        summaries[bucket] = ShapeSummary(
            total_cases=total,
            missing_img=missing_img,
            missing_mask=missing_mask,
            mismatched_img_mask=mismatched,
            stats=_summarize_shapes(shapes),
        )

    return summaries


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resumen general de dimensiones para Dataset y Dataset_Preprocessed.")
    p.add_argument("--dataset_root", type=Path, default=Path("Dataset"), help="Ruta a Dataset (raw).")
    p.add_argument(
        "--preprocessed_root",
        type=Path,
        default=Path("Dataset_Preprocessed"),
        help="Ruta a Dataset_Preprocessed (npy).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root
    preprocessed_root = args.preprocessed_root

    if dataset_root.exists():
        subsets = _find_subsets(dataset_root)
        if not subsets:
            print(f"[WARN] No se encontraron sub-datasets dentro de {dataset_root}")
        for subset in subsets:
            name = subset.name if subset != dataset_root else "(root)"
            _print_summary(f"RAW {name}", _summarize_raw_dataset(subset))
    else:
        print(f"[WARN] dataset_root no existe: {dataset_root}")

    if preprocessed_root.exists():
        summaries = _summarize_preprocessed(preprocessed_root)
        for key in sorted(summaries.keys(), key=_natural_key):
            _print_summary(f"PRE {key}", summaries[key])
    else:
        print(f"[WARN] preprocessed_root no existe: {preprocessed_root}")


if __name__ == "__main__":
    main()

