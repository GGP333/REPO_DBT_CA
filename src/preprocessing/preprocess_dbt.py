#!/usr/bin/env python3
"""
Preprocesado de stacks TIFF de DBT:
- Mantiene el spacing nativo durante la segmentación y recorte (sin remuestreo
  isotrópico; se preserva Z).
- Elige la mejor máscara (Otsu vs percentil) y aplica limpieza morfológica.
- Recorte por bounding box con margen, respetando el borde de la imagen.
- Clipping/normalización de intensidades.
- Redimensiona el plano XY al tamaño objetivo mediante interpolación (lineal
  para la imagen, vecino más cercano para la máscara) y ajusta el spacing XY
  resultante.
- Guarda volúmenes procesados como .npy en `Dataset_Preprocessed/<case_relpath>/`
  y opcionalmente genera previews.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk


# ----------------------------- Utilidades generales ----------------------------- #
def _natural_key(path: Path) -> List[object]:
    """Ordena nombres con partes numéricas (slice_1, slice_2, slice_10)."""
    import re

    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", path.name)
    ]


def parse_triplet(values: Sequence[str], expected_len: int = 3) -> Tuple[float, ...]:
    if len(values) != expected_len:
        raise argparse.ArgumentTypeError(f"Se esperaban {expected_len} valores, got {len(values)}")
    try:
        return tuple(float(v) for v in values)  # type: ignore[return-value]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Valores deben ser numéricos") from exc


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ----------------------------- E/S de imagen ----------------------------- #
def load_spacing_from_metadata(meta_path: Optional[Path]) -> Optional[Tuple[float, float, float]]:
    if not meta_path:
        return None
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata no encontrada: {meta_path}")

    if meta_path.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("pyyaml es requerido para leer YAML") from exc
        with open(meta_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    else:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    for key in ("spacing", "spacing_mm", "pixel_spacing"):
        if key in data:
            vals = data[key]
            if isinstance(vals, dict):
                vals = [vals[k] for k in sorted(vals.keys())]
            return tuple(float(x) for x in vals[:3])  # type: ignore[return-value]
    return None


def load_tiff_stack(folder: Path) -> np.ndarray:
    """Carga un stack de imágenes TIFF en orden z -> numpy (z, y, x)."""
    if not folder.is_dir():
        raise FileNotFoundError(f"input_dir no existe o no es carpeta: {folder}")
    tiffs = sorted([p for p in folder.iterdir() if p.suffix.lower() in {".tif", ".tiff"}], key=_natural_key)
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
    """
    Garantiza shape (z,y,x). Si llega (1,z,y,x) o (z,y,x,1), compacta la dimensión extra.
    """
    if vol.ndim == 4:
        if vol.shape[0] == 1:
            vol = vol[0]
        elif vol.shape[-1] == 1:
            vol = vol[..., 0]
    if vol.ndim != 3:
        raise ValueError(f"Volumen esperado 3D, shape recibido {vol.shape}")
    return vol


def infer_mask_dir_from_input(input_dir: Path) -> Path:
    """
    Busca una carpeta `mask_*` hermana de `input_dir` (que suele ser `img_*`).
    """
    parent = input_dir.parent
    candidates = [p for p in parent.iterdir() if p.is_dir() and p.name.lower().startswith("mask")]
    if not candidates:
        raise FileNotFoundError(f"No se encontró carpeta de máscara (mask_*) junto a {input_dir}")
    if len(candidates) > 1:
        raise ValueError(f"Se encontraron múltiples carpetas de máscara en {parent}: {candidates}")
    return candidates[0]


def save_image(path: Path, volume: np.ndarray, spacing: Tuple[float, float, float]) -> None:
    """Guarda volumen como NIfTI (.nii/.nii.gz) o TIFF apilado."""
    ensure_dir(path.parent)
    if path.suffix.lower() in {".nii", ".gz"} or path.name.endswith(".nii.gz"):
        img = sitk.GetImageFromArray(volume)
        img.SetSpacing(spacing)
        sitk.WriteImage(img, str(path))
    elif path.suffix.lower() in {".tif", ".tiff"}:
        try:
            import tifffile
        except ImportError as exc:
            raise RuntimeError("Instala 'tifffile' para escribir TIFF") from exc
        tifffile.imwrite(str(path), volume.astype(np.float32))
    else:
        raise ValueError(f"Formato no soportado para {path}")


# ----------------------------- Procesado espacial ----------------------------- #
def np_to_sitk(volume: np.ndarray, spacing: Tuple[float, float, float]) -> sitk.Image:
    img = sitk.GetImageFromArray(volume)
    img.SetSpacing(spacing)
    return img


def histogram_match(
    volume: np.ndarray,
    reference: np.ndarray,
    bins: int = 256,
    match_points: int = 10,
) -> np.ndarray:
    """
    Ajusta el histograma de `volume` al de `reference` usando SimpleITK.
    Devuelve un nuevo array float32.
    """
    flt = sitk.HistogramMatchingImageFilter()
    flt.SetNumberOfHistogramLevels(int(bins))
    flt.SetNumberOfMatchPoints(int(match_points))
    flt.ThresholdAtMeanIntensityOn()
    out = flt.Execute(
        sitk.GetImageFromArray(volume.astype(np.float32)),
        sitk.GetImageFromArray(reference.astype(np.float32)),
    )
    return sitk.GetArrayFromImage(out).astype(np.float32)


# ----------------------------- Histogram matching (global, masked) ----------------------------- #
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


def _compute_global_reference_cdf_from_dataset(
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
    Calcula un CDF de referencia (promedio por-estudio) sobre voxels de tejido (tissue_mask)
    para todos/una muestra de estudios dentro de `dataset_root` (espera estructura dbt_*/img_*).

    Devuelve (bin_centers, ref_cdf) donde ref_cdf está en [0,1].
    """
    if not dataset_root.exists():
        raise FileNotFoundError(f"hist_match_global_root no existe: {dataset_root}")

    # Buscar estudios recursivamente (dbt_* o real_dbt_*) con carpeta img_*
    def _is_study_dir(p: Path) -> bool:
        return p.name.startswith("dbt_") or p.name.startswith("real_dbt_")

    studies = sorted(
        [p for p in dataset_root.rglob("*") if p.is_dir() and _is_study_dir(p) and _find_study_img_dir(p) is not None],
        key=lambda p: p.as_posix(),
    )

    if not studies:
        raise RuntimeError(f"No se encontraron estudios en {dataset_root}")

    # Muestreo (opcional) para acelerar
    if max_studies and max_studies > 0 and len(studies) > max_studies:
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(len(studies), size=int(max_studies), replace=False)
        studies = [studies[int(i)] for i in sorted(idx)]

    # Pass 1: estimar rango global robusto a partir de percentiles por-estudio dentro de tejido
    los: List[float] = []
    his: List[float] = []
    used = 0
    for sd in studies:
        img_dir = _find_study_img_dir(sd)
        if img_dir is None:
            continue
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
            continue
        vox = vol_c[tissue_c > 0].astype(np.float32)
        lo, hi = np.percentile(vox, clip_percentiles)
        if float(lo) < float(hi):
            los.append(float(lo))
            his.append(float(hi))
            used += 1

    if used < 3:
        raise RuntimeError(
            f"No se pudo estimar histograma global (solo {used} estudios válidos con tejido)."
        )

    global_lo = float(np.median(np.array(los, dtype=np.float32)))
    global_hi = float(np.median(np.array(his, dtype=np.float32)))
    if not (global_lo < global_hi):
        raise RuntimeError(f"Rango global inválido para hist_match: lo={global_lo}, hi={global_hi}")

    # Pass 2: promedio por-estudio del histograma (para no sesgar por tamaño del seno)
    acc = np.zeros(int(bins), dtype=np.float64)
    n = 0
    for sd in studies:
        img_dir = _find_study_img_dir(sd)
        if img_dir is None:
            continue
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
            continue
        vox = vol_c[tissue_c > 0].astype(np.float32)
        vox = np.clip(vox, global_lo, global_hi)
        hist, edges = np.histogram(vox, bins=int(bins), range=(global_lo, global_hi))
        s = float(hist.sum())
        if s <= 0:
            continue
        acc += (hist.astype(np.float64) / s)
        n += 1

    if n < 3:
        raise RuntimeError(f"No se pudo construir histograma global (solo {n} estudios válidos).")

    ref_hist = acc / float(n)
    ref_cdf = np.cumsum(ref_hist).astype(np.float32)
    # Asegura monotonía y límite superior
    ref_cdf = np.clip(ref_cdf, 0.0, 1.0)
    ref_cdf[-1] = 1.0

    bin_centers = ((edges[:-1] + edges[1:]) * 0.5).astype(np.float32)
    return bin_centers, ref_cdf


def _apply_hist_match_cdf(
    volume: np.ndarray,
    tissue_mask: np.ndarray,
    *,
    bin_centers: np.ndarray,
    ref_cdf: np.ndarray,
) -> np.ndarray:
    """
    Histogram matching por CDF usando voxels dentro de `tissue_mask` para estimar
    el CDF fuente, y aplica el mapeo a TODO el volumen.
    """
    if volume.shape != tissue_mask.shape or not (tissue_mask > 0).any():
        return volume.astype(np.float32)

    v = volume.astype(np.float32)
    src_vox = v[tissue_mask > 0].astype(np.float32)
    # Clip al rango cubierto por los bins
    lo = float(bin_centers[0])
    hi = float(bin_centers[-1])
    src_vox = np.clip(src_vox, lo, hi)

    # Histograma fuente en los mismos bins (usando edges derivados de centros)
    # Reconstruimos edges aproximados desde centers (bins uniformes por construcción)
    if bin_centers.size < 2:
        return v
    step = float(bin_centers[1] - bin_centers[0])
    edges = np.concatenate(
        [
            (bin_centers[:1] - step / 2.0),
            (bin_centers + step / 2.0),
        ]
    )
    hist, _ = np.histogram(src_vox, bins=edges)
    s = float(hist.sum())
    if s <= 0:
        return v
    src_cdf = (np.cumsum(hist.astype(np.float32)) / s).astype(np.float32)
    src_cdf = np.clip(src_cdf, 0.0, 1.0)
    src_cdf[-1] = 1.0

    # Mapeo: x -> p = CDF_src(x) -> x' = invCDF_ref(p)
    x = np.clip(v, lo, hi)
    p = np.interp(x.ravel(), bin_centers, src_cdf).astype(np.float32)
    x_m = np.interp(p, ref_cdf, bin_centers).astype(np.float32)
    return x_m.reshape(v.shape).astype(np.float32)


# ----------------------------- Máscara y recorte ----------------------------- #
def _preprocess_for_mask(
    volume: np.ndarray,
    spacing: Tuple[float, float, float],
    clip_percentiles: Tuple[float, float] = (1.0, 99.0),
    smooth_sigma_mm: float = 0.6,
) -> np.ndarray:
    """
    Normaliza (clip + min-max) y aplica un suavizado ligero para estabilizar el
    umbral y las componentes conectadas posteriores.
    """
    v = volume.astype(np.float32)
    lo, hi = np.percentile(v, clip_percentiles)
    if lo < hi:
        v = np.clip(v, lo, hi)
        rng = hi - lo
        if rng > 0:
            v = (v - lo) / rng

    img = sitk.GetImageFromArray(v)
    img.SetSpacing(spacing)
    if smooth_sigma_mm > 0:
        img = sitk.SmoothingRecursiveGaussian(img, sigma=smooth_sigma_mm)
    return sitk.GetArrayFromImage(img)


def build_breast_mask(
    volume: np.ndarray,
    spacing: Tuple[float, float, float],
    method: str = "otsu",
    percentile: float = 40.0,
    closing_mm: float = 3.0,
    dilation_mm: float = 5.0,
    clip_percentiles: Tuple[float, float] = (1.0, 99.0),
    smooth_sigma_mm: float = 0.6,
) -> np.ndarray:
    if method not in {"otsu", "percentile"}:
        raise ValueError("mask_method debe ser 'otsu' o 'percentile'")

    # Paso 0: normalizar + suavizar
    pre = _preprocess_for_mask(
        volume,
        spacing,
        clip_percentiles=clip_percentiles,
        smooth_sigma_mm=smooth_sigma_mm,
    )

    # Paso 1: umbral (Otsu o percentil)
    if method == "otsu":
        img = np_to_sitk(pre, spacing)
        mask_img = sitk.OtsuThreshold(img, 0, 1, 200)
    else:
        thr = np.percentile(pre, percentile)
        mask_np = (pre >= thr).astype(np.uint8)
        mask_img = sitk.GetImageFromArray(mask_np)
        mask_img.SetSpacing(spacing)

    # Paso 2: componentes conectadas -> nos quedamos con la más grande
    cc = sitk.ConnectedComponent(mask_img)
    relabeled = sitk.RelabelComponent(cc, sortByObjectSize=True)
    largest = sitk.BinaryThreshold(relabeled, lowerThreshold=1, upperThreshold=1, insideValue=1, outsideValue=0)

    # Paso 3: closing 3D para rellenar huecos
    closing_radius = [
        max(1, int(round(closing_mm / spacing[i]))) for i in range(3)
    ]
    closed = sitk.BinaryMorphologicalClosing(largest, closing_radius, sitk.sitkBall)

    # Paso 4: dilatación para margen anatómico
    dilation_radius = [
        max(1, int(round(dilation_mm / spacing[i]))) for i in range(3)
    ]
    dilated = sitk.BinaryDilate(closed, dilation_radius, sitk.sitkBall)
    return sitk.GetArrayFromImage(dilated).astype(np.uint8)


def keep_largest_component_per_slice(mask: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Para cada corte a lo largo de `axis` (por defecto Z), conserva solo la mayor
    componente conectada 2D. Devuelve una máscara del mismo shape.
    """
    if axis != 0:
        raise ValueError("Solo se soporta axis=0 (Z) en el pipeline actual.")

    result = np.zeros_like(mask, dtype=np.uint8)
    for z in range(mask.shape[0]):
        slice_mask = mask[z]
        if slice_mask.max() == 0:
            continue
        img = sitk.GetImageFromArray(slice_mask.astype(np.uint8))
        cc = sitk.ConnectedComponent(img)
        relabeled = sitk.RelabelComponent(cc, sortByObjectSize=True)
        largest = sitk.BinaryThreshold(
            relabeled, lowerThreshold=1, upperThreshold=1, insideValue=1, outsideValue=0
        )
        result[z] = sitk.GetArrayFromImage(largest)
    return result


def replicate_xy_from_z_union(mask: np.ndarray) -> np.ndarray:
    """
    Proyecta la huella en XY como la unión de todos los cortes Z y la replica
    en cada slice. Útil cuando se desea garantizar presencia de máscara en
    X/Y siempre que haya tejido en algún corte axial.
    """
    xy = (mask.sum(axis=0) > 0).astype(mask.dtype)
    return np.repeat(xy[None, ...], mask.shape[0], axis=0)


def dilate_per_slice_xy(mask: np.ndarray, spacing: Tuple[float, float, float], dilation_mm: float) -> np.ndarray:
    # eliminado: no usamos dilatación 2D por corte en la versión actual
    return mask


def rebuild_xy_from_z(mask: np.ndarray) -> np.ndarray:
    """
    Reconstruye soporte en X/Y a partir de la huella axial (Z):
    toma la unión en Z y la replica en todos los cortes, garantizando
    presencia de máscara en X/Y donde exista en cualquier slice.
    """
    xy_support = (mask.sum(axis=0) > 0).astype(np.uint8)
    return np.repeat(xy_support[None, ...], mask.shape[0], axis=0)



def choose_best_mask(
    volume: np.ndarray,
    spacing: Tuple[float, float, float],
    percentile: float,
    closing_mm: float,
    dilation_mm: float,
    clip_percentiles: Tuple[float, float] = (1.0, 99.0),
    smooth_sigma_mm: float = 0.6,
) -> Tuple[np.ndarray, str]:
    """
    Genera máscara con Otsu y percentil y selecciona la que tenga ratio de
    foreground razonable. Preferimos máscaras con ratio entre 1% y 60% del
    volumen; si ambas están en rango, elegimos la de mayor ratio.
    """

    def _score(mask: np.ndarray) -> float:
        ratio = float(mask.mean())
        if ratio <= 0:
            return 0.0
        if 0.01 <= ratio <= 0.6:
            return ratio
        # penaliza máscaras demasiado pequeñas o grandes
        return ratio * 0.2

    mask_otsu = build_breast_mask(
        volume,
        spacing,
        method="otsu",
        percentile=percentile,
        closing_mm=closing_mm,
        dilation_mm=dilation_mm,
        clip_percentiles=clip_percentiles,
        smooth_sigma_mm=smooth_sigma_mm,
    )
    mask_pct = build_breast_mask(
        volume,
        spacing,
        method="percentile",
        percentile=percentile,
        closing_mm=closing_mm,
        dilation_mm=dilation_mm,
        clip_percentiles=clip_percentiles,
        smooth_sigma_mm=smooth_sigma_mm,
    )

    score_otsu = _score(mask_otsu)
    score_pct = _score(mask_pct)

    if score_otsu == 0 and score_pct == 0:
        return mask_otsu, "otsu"  # ambas fallaron, devolvemos Otsu por defecto
    if score_pct > score_otsu:
        return mask_pct, "percentile"
    return mask_otsu, "otsu"


def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None
    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0)
    return (zmin, zmax), (ymin, ymax), (xmin, xmax)


def crop_with_margin(
    volume: np.ndarray,
    mask: np.ndarray,
    spacing: Tuple[float, float, float],
    margin_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    slices = crop_slices_from_mask(mask, spacing, margin_mm)
    return volume[slices], mask[slices]


def crop_slices_from_mask(
    mask: np.ndarray, spacing: Tuple[float, float, float], margin_mm: float
) -> Tuple[slice, slice, slice]:
    bbox = bbox_from_mask(mask)
    if bbox is None:
        # Si no hay máscara, devolvemos volumen completo
        return (
            slice(0, mask.shape[0]),
            slice(0, mask.shape[1]),
            slice(0, mask.shape[2]),
        )

    margin_vox = [int(round(margin_mm / spacing[i])) for i in range(3)]
    (zmin, zmax), (ymin, ymax), (xmin, xmax) = bbox
    z0 = max(0, zmin - margin_vox[2])
    z1 = min(mask.shape[0], zmax + margin_vox[2] + 1)
    y0 = max(0, ymin - margin_vox[1])
    y1 = min(mask.shape[1], ymax + margin_vox[1] + 1)
    x0 = max(0, xmin - margin_vox[0])
    x1 = min(mask.shape[2], xmax + margin_vox[0] + 1)
    return (slice(z0, z1), slice(y0, y1), slice(x0, x1))


# ----------------------------- Normalización ----------------------------- #
def clip_and_normalize(
    volume: np.ndarray,
    clip_percentiles: Tuple[float, float],
    mode: str = "zscore",
    mask: Optional[np.ndarray] = None,
    background_value: Optional[float] = None,
) -> np.ndarray:
    """
    Clipping + normalización.

    Si `mask` se provee, calcula percentiles/estadísticos SOLO dentro de `mask>0`
    y NO modifica los voxels fuera de la máscara (mantiene la imagen). Si se
    quiere forzar fondo a un valor fijo, pasar `background_value` (e.g. 0.0).
    """
    v = volume.astype(np.float32)
    use_mask = mask is not None and mask.shape == v.shape and (mask > 0).any()
    apply_background = bool(use_mask and background_value is not None)

    if mode == "none":
        out = v
        if apply_background:
            out = out.copy()
            out[mask <= 0] = float(background_value)
        return out

    if use_mask:
        vox = v[mask > 0]
        lo, hi = np.percentile(vox, clip_percentiles)
    else:
        lo, hi = np.percentile(v, clip_percentiles)

    if lo < hi:
        v = np.clip(v, lo, hi)

    if mode == "zscore":
        if use_mask:
            vox = v[mask > 0]
            mean = float(vox.mean())
            std = float(vox.std())
        else:
            mean = float(v.mean())
            std = float(v.std())
        if std > 0:
            v = (v - mean) / std
        else:
            v = v - mean
    elif mode == "minmax":
        if use_mask:
            vox = v[mask > 0]
            vmin = float(vox.min())
            vmax = float(vox.max())
        else:
            vmin = float(v.min())
            vmax = float(v.max())
        rng = vmax - vmin
        if rng > 0:
            v = (v - vmin) / rng
        else:
            v = v * 0
    else:
        raise ValueError("normalize debe ser zscore, minmax o none")

    if apply_background:
        v = v.copy()
        v[mask <= 0] = float(background_value)
    return v


def resample_xy_to_spacing(
    volume: np.ndarray,
    spacing: Tuple[float, float, float],
    target_spacing_xy: Tuple[float, float],
    interpolator: int,
    default_value: float = 0.0,
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Remuestrea SOLO en XY para alcanzar `target_spacing_xy=(sx,sy)` manteniendo Z.
    Devuelve (volumen_resampleado, output_spacing).
    """
    if volume.ndim != 3:
        raise ValueError(f"Se esperaba un volumen 3D (z,y,x); shape recibido {volume.shape}")
    vz, vy, vx = volume.shape
    tsx, tsy = float(target_spacing_xy[0]), float(target_spacing_xy[1])
    if tsx <= 0 or tsy <= 0:
        raise ValueError("target_spacing_xy debe ser positivo (sx sy)")

    img = sitk.GetImageFromArray(volume)
    img.SetSpacing(spacing)

    # Nuevo tamaño preservando tamaño físico aprox (en XY)
    new_vx = max(1, int(round((vx * float(spacing[0])) / tsx)))
    new_vy = max(1, int(round((vy * float(spacing[1])) / tsy)))
    out_spacing = (tsx, tsy, float(spacing[2]))

    resampled = sitk.Resample(
        img,
        size=(new_vx, new_vy, vz),
        transform=sitk.Transform(),
        interpolator=interpolator,
        outputOrigin=img.GetOrigin(),
        outputSpacing=out_spacing,
        outputDirection=img.GetDirection(),
        defaultPixelValue=float(default_value),
    )
    return sitk.GetArrayFromImage(resampled), out_spacing


# ----------------------------- Redimensionado XY ----------------------------- #
def resize_xy(
    volume: np.ndarray,
    spacing: Tuple[float, float, float],
    target_size_xy: Tuple[int, int],
    interpolator: int = sitk.sitkLinear,
    default_value: float = 0.0,
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Redimensiona el plano XY a `target_size_xy` conservando el número de cortes
    Z. Devuelve el volumen remuestreado y el nuevo spacing (sx, sy, sz).
    """
    if volume.ndim != 3:
        raise ValueError(f"Se esperaba un volumen 3D (z,y,x); shape recibido {volume.shape}")

    vz, vy, vx = volume.shape
    xt, yt = (int(target_size_xy[0]), int(target_size_xy[1]))

    img = sitk.GetImageFromArray(volume)
    img.SetSpacing(spacing)

    new_spacing = (
        float(spacing[0] * vx / xt),
        float(spacing[1] * vy / yt),
        float(spacing[2]),
    )

    resampled = sitk.Resample(
        img,
        size=(xt, yt, vz),
        transform=sitk.Transform(),
        interpolator=interpolator,
        outputOrigin=img.GetOrigin(),
        outputSpacing=new_spacing,
        outputDirection=img.GetDirection(),
        defaultPixelValue=float(default_value),
    )
    return sitk.GetArrayFromImage(resampled), new_spacing


# ----------------------------- Previews QA ----------------------------- #
def save_previews(volume: np.ndarray, mask: np.ndarray, out_dir: Path, prefix: str) -> None:
    try:
        import imageio
    except ImportError as exc:
        raise RuntimeError("Instala 'imageio' para generar previews") from exc

    ensure_dir(out_dir)
    zc, yc, xc = [s // 2 for s in volume.shape]
    planes = {
        "axial": (volume[zc], mask[zc]),
        "coronal": (volume[:, yc, :], mask[:, yc, :]),
        "sagittal": (volume[:, :, xc], mask[:, :, xc]),
    }
    for name, (img, msk) in planes.items():
        norm = img
        if norm.max() > norm.min():
            norm = (norm - norm.min()) / (norm.max() - norm.min())
        overlay = (np.clip(norm, 0, 1) * 255).astype(np.uint8)
        overlay_mask = (msk > 0).astype(np.uint8) * 255
        imageio.imwrite(out_dir / f"{prefix}_{name}_img.png", overlay)
        imageio.imwrite(out_dir / f"{prefix}_{name}_mask.png", overlay_mask)


# ----------------------------- Config y CLI ----------------------------- #
@dataclass
class Params:
    input_dir: str
    output_dir: str
    mask_dir: Optional[str]
    spacing: Tuple[float, float, float]
    output_spacing: Tuple[float, float, float]
    target_size_xy: Tuple[int, int]
    margin_mm: float
    closing_mm: float
    dilation_mm: float
    clip_percentiles: Tuple[float, float]
    normalize: str
    mask_method: str
    mask_percentile: float
    format: str
    preview_slices: bool
    target_spacing_xy: Optional[Tuple[float, float]] = None
    metadata_file: Optional[str] = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocesado DBT TIFF -> máscara, crop, normalización y resize XY (sin remuestreo isotrópico).")
    p.add_argument("--input_dir", required=True, type=Path, help="Carpeta con slices .tif/.tiff (ordenados por nombre).")
    p.add_argument("--output_dir", required=True, type=Path, help="Carpeta base destino para volúmenes procesados.")
    p.add_argument(
        "--output_subdir",
        type=str,
        default="Dataset_Preprocessed",
        help="Subcarpeta bajo output_dir para los casos (default: Dataset_Preprocessed).",
    )
    p.add_argument(
        "--case_relpath",
        type=str,
        help="Ruta relativa del caso dentro del output_subdir (ej: Dataset_Both/dbt_001).",
    )
    p.add_argument("--mask_dir", type=Path, help="Carpeta con la máscara de hallazgo (.tif/.tiff). Si se omite, se busca mask_* junto a input_dir.")
    p.add_argument("--spacing", nargs=3, type=float, help="Espaciado original (sx sy sz) en mm si no hay metadata.")
    p.add_argument("--metadata", type=Path, help="JSON/YAML con 'spacing' o 'spacing_mm'.")
    p.add_argument("--target_size_xy", nargs=2, type=int, default=[512, 256], help="Tamaño final en voxeles (x y) para el resize interpolado en XY.")
    p.add_argument(
        "--target_spacing_xy",
        nargs=2,
        type=float,
        help="Si se provee, remuestrea XY a este spacing (sx sy, mm) y luego aplica resize a target_size_xy (sin pad/crop).",
    )
    p.add_argument("--margin_mm", type=float, default=5.0, help="Margen en mm alrededor del seno para recorte.")
    p.add_argument("--closing_mm", type=float, default=3.0, help="Radio (mm) para closing morfológico 3D de la máscara.")
    p.add_argument("--mask_dilation_mm", type=float, default=5.0, help="Radio (mm) para dilatación de la máscara.")
    p.add_argument("--mask_method", choices=["otsu", "percentile", "auto"], default="auto", help="Método de umbral para máscara de seno.")
    p.add_argument("--mask_percentile", type=float, default=40.0, help="Percentil para umbral si mask_method=percentile.")
    p.add_argument("--clip_percentiles", nargs=2, type=float, default=[0.5, 99.5], help="Percentiles para clipping antes de normalizar.")
    p.add_argument("--normalize", choices=["zscore", "minmax", "none"], default="zscore", help="Modo de normalización.")
    p.add_argument("--format", choices=["npy"], default="npy", help="Formato de salida (fijado a npy).")
    p.add_argument("--preview_slices", action="store_true", help="Guardar previews PNG de cortes medios.")
    p.add_argument("--preview_dir", type=Path, help="Destino para previews (por defecto output_dir/previews).")
    p.add_argument("--hist_match_ref", type=Path, help="Ruta a volumen de referencia (npy o carpeta TIFF) para histogram matching opcional.")
    p.add_argument("--hist_match_bins", type=int, default=256, help="Niveles de histograma para matching.")
    p.add_argument("--hist_match_points", type=int, default=10, help="Puntos de emparejamiento para matching.")
    p.add_argument(
        "--hist_match_global_root",
        type=Path,
        help="Si se provee, calcula (o carga) un histograma de referencia GLOBAL usando SOLO voxels dentro de la tissue_mask de todos los estudios en este root (p.ej. Dataset/).",
    )
    p.add_argument(
        "--hist_match_global_cache",
        type=Path,
        help="Ruta a .npz para cachear el histograma global (bin_centers + ref_cdf). Si se omite, usa output_dir/global_hist_ref_tissue.npz.",
    )
    p.add_argument(
        "--hist_match_global_max_studies",
        type=int,
        default=0,
        help="Máximo de estudios para estimar el histograma global (0 = todos). Útil para acelerar.",
    )
    p.add_argument(
        "--hist_match_global_seed",
        type=int,
        default=0,
        help="Semilla para muestreo aleatorio cuando max_studies > 0.",
    )
    p.add_argument(
        "--hist_match_global_force_recompute",
        action="store_true",
        help="Recalcula el histograma global aunque exista el cache.",
    )
    return p.parse_args()


# ----------------------------- Pipeline principal ----------------------------- #
def main() -> None:
    args = parse_args()

    spacing = load_spacing_from_metadata(args.metadata)
    if spacing is None:
        if args.spacing is None:
            raise ValueError("Debe proveer --spacing o metadata con spacing.")
        spacing = tuple(float(s) for s in args.spacing)  # type: ignore[assignment]

    print(f"[INFO] Se mantiene spacing original {spacing}; remuestreo isotrópico desactivado")

    target_size_xy = tuple(int(x) for x in args.target_size_xy)  # type: ignore[assignment]
    clip_percentiles = tuple(float(x) for x in args.clip_percentiles)  # type: ignore[assignment]

    study_id = args.input_dir.name
    base_out = args.output_dir
    case_relpath = Path(args.case_relpath) if args.case_relpath else Path(study_id)
    if case_relpath.is_absolute():
        raise ValueError("--case_relpath debe ser una ruta relativa, no absoluta.")
    out_dir = base_out / args.output_subdir / case_relpath
    ensure_dir(out_dir)

    print(f"[INFO] Cargando stack TIFF desde {args.input_dir}")
    vol = load_tiff_stack(args.input_dir)
    vol = ensure_volume_3d(vol)
    print(f"[INFO] Volumen original shape (z,y,x): {vol.shape}, spacing: {spacing}")

    # Cargar máscara de hallazgo (GT)
    mask_dir = args.mask_dir or infer_mask_dir_from_input(args.input_dir)
    print(f"[INFO] Cargando máscara GT desde {mask_dir}")
    gt_mask = load_tiff_stack(mask_dir)
    gt_mask = ensure_volume_3d(gt_mask).astype(np.uint8)
    if gt_mask.shape != vol.shape:
        raise ValueError(f"Shape de máscara {gt_mask.shape} no coincide con imagen {vol.shape}")

    print("[INFO] Sin resample: el volumen se procesa en spacing nativo")
    vol_native = vol

    print(f"[INFO] Construyendo máscara de seno con método {args.mask_method}")
    if args.mask_method == "auto":
        tissue_mask_native, chosen = choose_best_mask(
            vol_native,
            spacing,
            percentile=args.mask_percentile,
            closing_mm=args.closing_mm,
            dilation_mm=args.mask_dilation_mm,
        )
        print(f"[INFO] Método seleccionado para máscara: {chosen}")
    else:
        tissue_mask_native = build_breast_mask(
            vol_native,
            spacing,
            method=args.mask_method,
            percentile=args.mask_percentile,
            closing_mm=args.closing_mm,
            dilation_mm=args.mask_dilation_mm,
        )

    # IMPORTANTE: la máscara se usa SOLO para calcular el recorte (bbox+margen).
    # No se enmascara el volumen: mantenemos la imagen original (vol_native) intacta.

    print(f"[INFO] Recorte usando bbox + margen {args.margin_mm} mm")
    crop_slices = crop_slices_from_mask(tissue_mask_native, spacing, args.margin_mm)
    vol_crop = vol_native[crop_slices]
    gt_mask_crop = gt_mask[crop_slices]
    tissue_mask_crop = tissue_mask_native[crop_slices].astype(np.uint8)
    print(f"[INFO] Shape tras crop: {vol_crop.shape}")

    # Matching de histograma opcional
    if args.hist_match_ref:
        print(f"[INFO] Histogram matching usando referencia {args.hist_match_ref}")
        ref_path = args.hist_match_ref
        if ref_path.is_dir():
            ref_vol = load_tiff_stack(ref_path)
        elif ref_path.suffix.lower() in {".npy"}:
            ref_vol = np.load(ref_path)
        elif ref_path.suffix.lower() in {".tif", ".tiff"}:
            # si es un único tiff apilado
            try:
                import tifffile
            except ImportError as exc:
                raise RuntimeError("Instala 'tifffile' para leer TIFF de referencia") from exc
            ref_vol = tifffile.imread(str(ref_path))
        else:
            raise ValueError("hist_match_ref debe ser carpeta de TIFF o archivo .npy/.tif/.tiff")

        ref_vol = ref_vol.astype(np.float32)
        ref_vol = ensure_volume_3d(ref_vol)
        vol_crop = histogram_match(
            vol_crop,
            ref_vol,
            bins=args.hist_match_bins,
            match_points=args.hist_match_points,
        )

    # Matching de histograma GLOBAL (promedio en tissue_mask) opcional
    if args.hist_match_global_root:
        cache_path = args.hist_match_global_cache or (base_out / "global_hist_ref_tissue.npz")
        if cache_path.exists() and not bool(args.hist_match_global_force_recompute):
            data = np.load(cache_path)
            bin_centers = data["bin_centers"].astype(np.float32)
            ref_cdf = data["ref_cdf"].astype(np.float32)
            print(f"[INFO] Cargado histograma global desde cache: {cache_path}")
        else:
            print(f"[INFO] Calculando histograma global (tissue_mask) desde {args.hist_match_global_root} ...")
            bin_centers, ref_cdf = _compute_global_reference_cdf_from_dataset(
                dataset_root=args.hist_match_global_root,
                fallback_spacing=spacing,
                bins=int(args.hist_match_bins),
                clip_percentiles=clip_percentiles,
                margin_mm=float(args.margin_mm),
                mask_method=str(args.mask_method),
                mask_percentile=float(args.mask_percentile),
                closing_mm=float(args.closing_mm),
                dilation_mm=float(args.mask_dilation_mm),
                max_studies=int(args.hist_match_global_max_studies),
                seed=int(args.hist_match_global_seed),
            )
            ensure_dir(cache_path.parent)
            np.savez_compressed(cache_path, bin_centers=bin_centers, ref_cdf=ref_cdf)
            print(f"[INFO] Histograma global guardado en: {cache_path}")

        print("[INFO] Aplicando histogram matching global (CDF) usando tissue_mask del caso actual")
        vol_crop = _apply_hist_match_cdf(
            vol_crop,
            tissue_mask_crop.astype(np.uint8),
            bin_centers=bin_centers,
            ref_cdf=ref_cdf,
        )

    # 1) Estandarización de escala (opcional): resample a spacing XY fijo + resize a target_size_xy
    if args.target_spacing_xy:
        target_spacing_xy = (float(args.target_spacing_xy[0]), float(args.target_spacing_xy[1]))
        print(f"[INFO] Remuestreando XY a spacing fijo {target_spacing_xy} y luego resize a {target_size_xy} (sin crop/pad)")
        vol_rs, rs_spacing = resample_xy_to_spacing(
            vol_crop,
            spacing,
            target_spacing_xy,
            interpolator=sitk.sitkLinear,
            default_value=0.0,
        )
        gt_rs, _ = resample_xy_to_spacing(
            gt_mask_crop.astype(np.uint8),
            spacing,
            target_spacing_xy,
            interpolator=sitk.sitkNearestNeighbor,
            default_value=0,
        )
        tissue_rs, _ = resample_xy_to_spacing(
            tissue_mask_crop.astype(np.uint8),
            spacing,
            target_spacing_xy,
            interpolator=sitk.sitkNearestNeighbor,
            default_value=0,
        )

        # Resize interpolado a tamaño fijo (sin crop/pad)
        vol_rs, output_spacing = resize_xy(
            vol_rs,
            rs_spacing,
            target_size_xy,
            interpolator=sitk.sitkLinear,  # bilinear en XY (Z se mantiene)
            default_value=0.0,
        )
        gt_rs, _ = resize_xy(
            gt_rs.astype(np.uint8),
            rs_spacing,
            target_size_xy,
            interpolator=sitk.sitkNearestNeighbor,
            default_value=0,
        )
        tissue_rs, _ = resize_xy(
            tissue_rs.astype(np.uint8),
            rs_spacing,
            target_size_xy,
            interpolator=sitk.sitkNearestNeighbor,
            default_value=0,
        )

        print(f"[INFO] Clipping {clip_percentiles} y normalización (estadísticos dentro de mask; sin enmascarar imagen) {args.normalize}")
        vol_final = clip_and_normalize(vol_rs, clip_percentiles, mode=args.normalize, mask=tissue_rs, background_value=None)
        gt_mask_final = (gt_rs > 0).astype(np.uint8)
        tissue_mask_final = (tissue_rs > 0).astype(np.uint8)
        print(f"[INFO] Spacing final tras resample XY + resize: {output_spacing}")
    else:
        # 2) Pipeline anterior: normaliza y luego resize XY a tamaño fijo
        print(f"[INFO] Clipping {clip_percentiles} y normalización (estadísticos dentro de mask; sin enmascarar imagen) {args.normalize}")
        vol_norm = clip_and_normalize(vol_crop, clip_percentiles, mode=args.normalize, mask=tissue_mask_crop, background_value=None)

        print(f"[INFO] Redimensionando XY a {target_size_xy} con interpolación (Z se mantiene)")
        vol_final, output_spacing = resize_xy(
            vol_norm,
            spacing,
            target_size_xy,
            interpolator=sitk.sitkLinear,
            default_value=0.0,
        )
        gt_mask_final, _ = resize_xy(
            gt_mask_crop.astype(np.uint8),
            spacing,
            target_size_xy,
            interpolator=sitk.sitkNearestNeighbor,
            default_value=0,
        )
        tissue_mask_final, _ = resize_xy(
            tissue_mask_crop.astype(np.uint8),
            spacing,
            target_size_xy,
            interpolator=sitk.sitkNearestNeighbor,
            default_value=0,
        )
        gt_mask_final = (gt_mask_final > 0).astype(np.uint8)
        tissue_mask_final = (tissue_mask_final > 0).astype(np.uint8)
        print(f"[INFO] Spacing final tras resize XY: {output_spacing}")

    # Salvar resultados en NPY
    img_out = out_dir / f"{study_id}_img.npy"
    mask_out = out_dir / f"{study_id}_mask.npy"
    tissue_out = out_dir / f"{study_id}_tissue_mask.npy"
    np.save(img_out, vol_final.astype(np.float32))
    np.save(mask_out, gt_mask_final.astype(np.uint8))
    np.save(tissue_out, tissue_mask_final.astype(np.uint8))
    print(f"[INFO] Guardado volumen en {img_out}")
    print(f"[INFO] Guardado máscara en {mask_out}")
    print(f"[INFO] Guardado tissue_mask en {tissue_out}")

    params = Params(
        input_dir=str(args.input_dir),
        output_dir=str(out_dir),
        spacing=tuple(spacing),
        output_spacing=tuple(output_spacing),
        target_size_xy=target_size_xy,
        target_spacing_xy=tuple(float(x) for x in args.target_spacing_xy) if args.target_spacing_xy else None,
        margin_mm=args.margin_mm,
        closing_mm=args.closing_mm,
        dilation_mm=args.mask_dilation_mm,
        clip_percentiles=clip_percentiles,
        normalize=args.normalize,
        mask_method=args.mask_method,
        mask_percentile=args.mask_percentile,
        format=args.format,
        preview_slices=bool(args.preview_slices),
        mask_dir=str(mask_dir) if mask_dir else None,
        metadata_file=str(args.metadata) if args.metadata else None,
    )
    with open(out_dir / "preprocess_params.json", "w", encoding="utf-8") as f:
        json.dump(asdict(params), f, indent=2)

    if args.preview_slices:
        prev_dir = args.preview_dir or (out_dir / "previews")
        save_previews(vol_final, gt_mask_final, prev_dir, study_id)
        print(f"[INFO] Previews guardados en {prev_dir}")


if __name__ == "__main__":
    main()

