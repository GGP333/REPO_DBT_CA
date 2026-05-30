#!/usr/bin/env python3
"""
Construye un histograma/CDF de referencia GLOBAL usando SOLO voxels dentro de la
`tissue_mask` (máscara de seno/tejido) a partir de todos los estudios en Dataset/.

Salida:
- Un .npz con:
  - bin_centers: (bins,) float32
  - ref_cdf:     (bins,) float32 en [0,1]

Uso típico:
  python scripts/build_global_hist_ref.py \
    --dataset_root Dataset \
    --out global_hist_ref_tissue.npz
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple, TypeVar

import numpy as np

# Reutilizamos funciones del pipeline principal
try:
    from preprocess_dbt import (  # type: ignore
        build_breast_mask,
        choose_best_mask,
        crop_slices_from_mask,
        ensure_dir,
        ensure_volume_3d,
        load_spacing_from_metadata,
        load_tiff_stack,
    )
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "No se pudo importar preprocess_dbt.py. Ejecuta este script desde la carpeta 'scripts/' "
        "o asegúrate de que scripts/ esté en PYTHONPATH."
    ) from exc


T = TypeVar("T")


def _find_study_img_dir(study_dir: Path) -> Optional[Path]:
    for p in sorted(study_dir.iterdir(), key=lambda x: x.name):
        if p.is_dir() and p.name.startswith("img_"):
            return p
    return None


def _infer_study_metadata_path(study_dir: Path) -> Optional[Path]:
    for name in ("metadata.json", "metadata.yaml", "metadata.yml"):
        meta = study_dir / name
        if meta.exists():
            return meta
    return None


def _maybe_tqdm(it: Iterable[T], *, total: int, desc: str) -> Iterator[T]:
    """
    Barra de progreso: usa tqdm si está disponible; si no, usa una barra simple.
    """
    try:
        from tqdm import tqdm  # type: ignore

        yield from tqdm(it, total=total, desc=desc, unit="study")
        return
    except Exception:
        pass

    # Fallback sin dependencias (barra simple)
    width = 28
    count = 0

    def _render(n: int) -> None:
        frac = 0.0 if total <= 0 else min(1.0, n / total)
        filled = int(round(frac * width))
        bar = "#" * filled + "-" * (width - filled)
        pct = int(round(frac * 100))
        sys.stderr.write(f"\r{desc} [{bar}] {n}/{total} ({pct}%)")
        sys.stderr.flush()

    _render(0)
    for x in it:
        yield x
        count += 1
        _render(count)
    sys.stderr.write("\n")


def _pass1_worker(
    study_dir_str: str,
    *,
    fallback_spacing: Tuple[float, float, float],
    clip_percentiles: Tuple[float, float],
    margin_mm: float,
    mask_method: str,
    mask_percentile: float,
    closing_mm: float,
    dilation_mm: float,
) -> Optional[Tuple[float, float]]:
    """
    Worker picklable (para multiprocessing):
    devuelve (lo, hi) percentiles dentro de tejido, o None si no aplica.
    """
    sd = Path(study_dir_str)
    img_dir = _find_study_img_dir(sd)
    if img_dir is None:
        return None

    meta_path = _infer_study_metadata_path(sd)
    spacing = load_spacing_from_metadata(meta_path) or fallback_spacing

    vol = ensure_volume_3d(load_tiff_stack(img_dir))
    if mask_method == "auto":
        tissue_mask, _ = choose_best_mask(
            vol,
            spacing,
            percentile=mask_percentile,
            closing_mm=closing_mm,
            dilation_mm=dilation_mm,
        )
    else:
        tissue_mask = build_breast_mask(
            vol,
            spacing,
            method=mask_method,
            percentile=mask_percentile,
            closing_mm=closing_mm,
            dilation_mm=dilation_mm,
        )

    crop_sl = crop_slices_from_mask(tissue_mask, spacing, margin_mm)
    vol_c = vol[crop_sl]
    tissue_c = tissue_mask[crop_sl]
    if not (tissue_c > 0).any():
        return None

    vox = vol_c[tissue_c > 0].astype(np.float32)
    lo, hi = np.percentile(vox, clip_percentiles)
    lo_f, hi_f = float(lo), float(hi)
    if lo_f >= hi_f:
        return None
    return lo_f, hi_f


def _pass2_worker(
    study_dir_str: str,
    *,
    fallback_spacing: Tuple[float, float, float],
    bins: int,
    global_lo: float,
    global_hi: float,
    margin_mm: float,
    mask_method: str,
    mask_percentile: float,
    closing_mm: float,
    dilation_mm: float,
) -> Optional[np.ndarray]:
    """
    Worker picklable: devuelve histograma normalizado (bins,) o None.
    """
    sd = Path(study_dir_str)
    img_dir = _find_study_img_dir(sd)
    if img_dir is None:
        return None

    meta_path = _infer_study_metadata_path(sd)
    spacing = load_spacing_from_metadata(meta_path) or fallback_spacing

    vol = ensure_volume_3d(load_tiff_stack(img_dir))
    if mask_method == "auto":
        tissue_mask, _ = choose_best_mask(
            vol,
            spacing,
            percentile=mask_percentile,
            closing_mm=closing_mm,
            dilation_mm=dilation_mm,
        )
    else:
        tissue_mask = build_breast_mask(
            vol,
            spacing,
            method=mask_method,
            percentile=mask_percentile,
            closing_mm=closing_mm,
            dilation_mm=dilation_mm,
        )

    crop_sl = crop_slices_from_mask(tissue_mask, spacing, margin_mm)
    vol_c = vol[crop_sl]
    tissue_c = tissue_mask[crop_sl]
    if not (tissue_c > 0).any():
        return None

    vox = vol_c[tissue_c > 0].astype(np.float32)
    vox = np.clip(vox, float(global_lo), float(global_hi))
    hist, _ = np.histogram(vox, bins=int(bins), range=(float(global_lo), float(global_hi)))
    s = float(hist.sum())
    if s <= 0:
        return None
    return (hist.astype(np.float64) / s).astype(np.float64)


def build_global_reference_cdf(
    *,
    dataset_root: Path,
    fallback_spacing: Tuple[float, float, float],
    bins: int,
    clip_percentiles: Tuple[float, float],
    margin_mm: float,
    mask_method: str,
    mask_percentile: float,
    closing_mm: float,
    dilation_mm: float,
    max_studies: int = 0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Promedia histogramas normalizados por-estudio (para no sesgar por tamaño de tejido).

    Pass 1: estima rango global robusto (mediana de lo/hi percentiles por estudio en tejido).
    Pass 2: acumula histogramas normalizados por estudio dentro de ese rango.
    """
    if not dataset_root.exists():
        raise FileNotFoundError(f"--dataset_root no existe: {dataset_root}")

    studies = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
    dbt_like = [p for p in studies if p.name.startswith("dbt_")]
    if dbt_like:
        studies = dbt_like
    if not studies:
        raise RuntimeError(f"No se encontraron estudios en {dataset_root}")

    # Muestreo (opcional)
    if max_studies and max_studies > 0 and len(studies) > max_studies:
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(len(studies), size=int(max_studies), replace=False)
        studies = [studies[int(i)] for i in sorted(idx)]
    study_strs = [str(p) for p in studies]

    # Pass 1: rango global
    los: List[float] = []
    his: List[float] = []
    used = 0
    if int(getattr(build_global_reference_cdf, "_jobs", 1)) > 1:
        jobs = int(getattr(build_global_reference_cdf, "_jobs", 1))
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            fn = partial(
                _pass1_worker,
                fallback_spacing=fallback_spacing,
                clip_percentiles=clip_percentiles,
                margin_mm=float(margin_mm),
                mask_method=str(mask_method),
                mask_percentile=float(mask_percentile),
                closing_mm=float(closing_mm),
                dilation_mm=float(dilation_mm),
            )
            it = ex.map(fn, study_strs)
            for out in _maybe_tqdm(it, total=len(study_strs), desc="Pass 1/2: rango"):
                if out is None:
                    continue
                lo_f, hi_f = out
                los.append(float(lo_f))
                his.append(float(hi_f))
                used += 1
    else:
        for sd_str in _maybe_tqdm(study_strs, total=len(study_strs), desc="Pass 1/2: rango"):
            out = _pass1_worker(
                sd_str,
                fallback_spacing=fallback_spacing,
                clip_percentiles=clip_percentiles,
                margin_mm=float(margin_mm),
                mask_method=str(mask_method),
                mask_percentile=float(mask_percentile),
                closing_mm=float(closing_mm),
                dilation_mm=float(dilation_mm),
            )
            if out is None:
                continue
            lo_f, hi_f = out
            los.append(float(lo_f))
            his.append(float(hi_f))
            used += 1

    if used < 3:
        raise RuntimeError(f"No hay suficientes estudios válidos con tejido para rango global (used={used}).")

    global_lo = float(np.median(np.array(los, dtype=np.float32)))
    global_hi = float(np.median(np.array(his, dtype=np.float32)))
    if not (global_lo < global_hi):
        raise RuntimeError(f"Rango global inválido: lo={global_lo}, hi={global_hi}")

    # Pass 2: hist promedio por-estudio
    acc = np.zeros(int(bins), dtype=np.float64)
    n = 0
    # Edges determinísticos para que bin_centers sean consistentes
    edges = np.linspace(global_lo, global_hi, int(bins) + 1, dtype=np.float32)

    if int(getattr(build_global_reference_cdf, "_jobs", 1)) > 1:
        jobs = int(getattr(build_global_reference_cdf, "_jobs", 1))
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            fn = partial(
                _pass2_worker,
                fallback_spacing=fallback_spacing,
                bins=int(bins),
                global_lo=float(global_lo),
                global_hi=float(global_hi),
                margin_mm=float(margin_mm),
                mask_method=str(mask_method),
                mask_percentile=float(mask_percentile),
                closing_mm=float(closing_mm),
                dilation_mm=float(dilation_mm),
            )
            it = ex.map(fn, study_strs)
            for h in _maybe_tqdm(it, total=len(study_strs), desc="Pass 2/2: hist"):
                if h is None:
                    continue
                acc += h
                n += 1
    else:
        for sd_str in _maybe_tqdm(study_strs, total=len(study_strs), desc="Pass 2/2: hist"):
            h = _pass2_worker(
                sd_str,
                fallback_spacing=fallback_spacing,
                bins=int(bins),
                global_lo=float(global_lo),
                global_hi=float(global_hi),
                margin_mm=float(margin_mm),
                mask_method=str(mask_method),
                mask_percentile=float(mask_percentile),
                closing_mm=float(closing_mm),
                dilation_mm=float(dilation_mm),
            )
            if h is None:
                continue
            acc += h
            n += 1

    if n < 3:
        raise RuntimeError(f"No hay suficientes estudios válidos para histograma global (n={n}).")

    ref_hist = acc / float(n)
    ref_cdf = np.cumsum(ref_hist).astype(np.float32)
    ref_cdf = np.clip(ref_cdf, 0.0, 1.0)
    ref_cdf[-1] = 1.0
    bin_centers = ((edges[:-1] + edges[1:]) * 0.5).astype(np.float32)
    return bin_centers, ref_cdf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Construye histograma/CDF global de referencia (tissue_mask).")
    p.add_argument("--dataset_root", type=Path, required=True, help="Ruta a Dataset/ (con subcarpetas dbt_*).")
    p.add_argument("--out", type=Path, required=True, help="Ruta de salida .npz (cache).")

    p.add_argument("--bins", type=int, default=256, help="Número de bins.")
    p.add_argument("--clip_percentiles", nargs=2, type=float, default=[0.5, 99.5], help="Percentiles (lo hi) dentro de tejido.")
    p.add_argument("--fallback_spacing", nargs=3, type=float, default=[0.085, 0.085, 1.0], help="Spacing (sx sy sz) si no hay metadata.")
    p.add_argument("--margin_mm", type=float, default=5.0, help="Margen mm del crop (igual que preprocess).")

    p.add_argument("--mask_method", choices=["otsu", "percentile", "auto"], default="auto", help="Método de máscara de tejido.")
    p.add_argument("--mask_percentile", type=float, default=40.0, help="Percentil si mask_method=percentile.")
    p.add_argument("--closing_mm", type=float, default=3.0, help="Radio mm closing 3D.")
    p.add_argument("--mask_dilation_mm", type=float, default=5.0, help="Radio mm dilatación 3D.")

    p.add_argument("--max_studies", type=int, default=0, help="Máximo de estudios para estimación (0=todos).")
    p.add_argument("--seed", type=int, default=0, help="Semilla para muestreo si max_studies>0.")
    p.add_argument("--force", action="store_true", help="Recalcula aunque exista --out.")
    p.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Procesos en paralelo para el cómputo por-estudio (1 = secuencial).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists() and not bool(args.force):
        print(f"[INFO] Cache ya existe, no se recalcula: {args.out}")
        print("[INFO] Usa --force para recalcular.")
        return

    clip_percentiles = (float(args.clip_percentiles[0]), float(args.clip_percentiles[1]))
    fallback_spacing = (float(args.fallback_spacing[0]), float(args.fallback_spacing[1]), float(args.fallback_spacing[2]))

    jobs = int(args.jobs)
    if jobs == 0:
        jobs = max(1, os.cpu_count() or 1)
    elif jobs < 1:
        jobs = 1
    # Hack simple: pasamos jobs a build_global_reference_cdf sin cambiar su firma pública
    setattr(build_global_reference_cdf, "_jobs", jobs)

    bin_centers, ref_cdf = build_global_reference_cdf(
        dataset_root=args.dataset_root,
        fallback_spacing=fallback_spacing,
        bins=int(args.bins),
        clip_percentiles=clip_percentiles,
        margin_mm=float(args.margin_mm),
        mask_method=str(args.mask_method),
        mask_percentile=float(args.mask_percentile),
        closing_mm=float(args.closing_mm),
        dilation_mm=float(args.mask_dilation_mm),
        max_studies=int(args.max_studies),
        seed=int(args.seed),
    )

    ensure_dir(args.out.parent)
    np.savez_compressed(args.out, bin_centers=bin_centers, ref_cdf=ref_cdf)
    print(f"[INFO] Guardado histograma global en: {args.out}")


if __name__ == "__main__":
    main()


