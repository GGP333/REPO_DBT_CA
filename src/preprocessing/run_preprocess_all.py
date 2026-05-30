#!/usr/bin/env python3
"""
Orquesta preprocess_dbt.py sobre todos los estudios dbt_* en Dataset/.
Parámetros fijos dentro de este script; solo hay que ejecutar:

    python scripts/run_preprocess_all.py
"""
from __future__ import annotations

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]

# Parámetros del pipeline
DEFAULT_SPACING = (0.085, 0.085, 1.0)  # usado si no hay metadata
TARGET_SIZE_XY = (512, 256)
TARGET_SPACING_XY = (0.085, 0.085)  # spacing XY fijo (mm) para estandarizar escala
MARGIN_MM = 5.0
CLOSING_MM = 3.0
DILATION_MM = 5.0
MASK_PERCENTILE = 40.0
CLIP_PERCENTILES = (0.5, 99.5)
NORMALIZE = "zscore"  # zscore | minmax | none
MASK_METHOD = "auto"  # auto | otsu | percentile

# Histogram matching opcional: deja REF vacío para desactivar
HIST_MATCH_REF = ""  # Path a .npy, .tif/.tiff o carpeta con tiffs; "" -> desactivado
HIST_MATCH_BINS = 256
HIST_MATCH_POINTS = 10

# Histogram matching GLOBAL (promedio sobre tissue_mask de todos los estudios)
HIST_MATCH_GLOBAL = True
HIST_MATCH_GLOBAL_MAX_STUDIES = 0  # 0 = todos (puede ser lento la primera vez; luego usa cache)
HIST_MATCH_GLOBAL_SEED = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ejecuta preprocess_dbt.py para todos los estudios dbt_* en Dataset/.")
    p.add_argument(
        "--dataset_root",
        type=Path,
        default=REPO_ROOT / "Dataset",
        help="Carpeta base donde están los sub-datasets (default: REPO_ROOT/Dataset).",
    )
    p.add_argument(
        "--output_root",
        type=Path,
        default=REPO_ROOT,
        help="Carpeta base para salidas (default: REPO_ROOT).",
    )
    p.add_argument(
        "--output_subdir",
        type=str,
        default="Dataset_Preprocessed",
        help="Subcarpeta bajo output_root para los casos (default: Dataset_Preprocessed).",
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Número de estudios a procesar en paralelo (default: 1).",
    )
    p.add_argument(
        "--preview_slices",
        action="store_true",
        help="Genera PNGs de QA (axial/coronal/sagittal) por estudio.",
    )
    p.add_argument(
        "--preview_dir",
        type=Path,
        help="Carpeta destino para previews. Si se omite, cada estudio usa output_subdir/<case_relpath>/previews/.",
    )
    return p.parse_args()


def find_img_dir(study_dir: Path) -> Path | None:
    for p in sorted(study_dir.iterdir()):
        if p.is_dir() and p.name.startswith("img_"):
            return p
    return None


def build_meta_args(study_dir: Path) -> List[str]:
    for name in ("metadata.json", "metadata.yaml", "metadata.yml"):
        meta = study_dir / name
        if meta.exists():
            return ["--metadata", str(meta)]
    return ["--spacing", *[str(x) for x in DEFAULT_SPACING]]


def build_hist_args(dataset_root: Path, output_root: Path, output_subdir: str) -> List[str]:
    if not HIST_MATCH_REF:
        args: List[str] = []
    else:
        args = [
            "--hist_match_ref",
            str(HIST_MATCH_REF),
            "--hist_match_bins",
            str(HIST_MATCH_BINS),
            "--hist_match_points",
            str(HIST_MATCH_POINTS),
        ]

    if HIST_MATCH_GLOBAL:
        args += [
            "--hist_match_global_root",
            str(dataset_root),
            "--hist_match_global_cache",
            str(Path(output_root) / output_subdir / "global_hist_ref_tissue.npz"),
            "--hist_match_global_max_studies",
            str(HIST_MATCH_GLOBAL_MAX_STUDIES),
            "--hist_match_global_seed",
            str(HIST_MATCH_GLOBAL_SEED),
        ]

    return args


def run_study(
    study_dir: Path,
    *,
    dataset_root: Path,
    output_root: Path,
    output_subdir: str,
    preview_slices: bool,
    preview_dir: Path | None,
) -> None:
    study_id = study_dir.name
    img_dir = find_img_dir(study_dir)
    if img_dir is None:
        print(f"[WARN] {study_id}: sin carpeta img_*; se omite")
        return

    meta_args = build_meta_args(study_dir)
    hist_args = build_hist_args(dataset_root, output_root, output_subdir)
    case_relpath = study_dir.relative_to(dataset_root)

    cmd = [
        "python",
        str(REPO_ROOT / "scripts" / "preprocess_dbt.py"),
        "--input_dir",
        str(img_dir),
        "--output_dir",
        str(output_root),
        "--output_subdir",
        str(output_subdir),
        "--case_relpath",
        str(case_relpath),
        "--target_size_xy",
        str(TARGET_SIZE_XY[0]),
        str(TARGET_SIZE_XY[1]),
        "--target_spacing_xy",
        str(TARGET_SPACING_XY[0]),
        str(TARGET_SPACING_XY[1]),
        "--margin_mm",
        str(MARGIN_MM),
        "--closing_mm",
        str(CLOSING_MM),
        "--mask_dilation_mm",
        str(DILATION_MM),
        "--mask_method",
        MASK_METHOD,
        "--mask_percentile",
        str(MASK_PERCENTILE),
        "--clip_percentiles",
        str(CLIP_PERCENTILES[0]),
        str(CLIP_PERCENTILES[1]),
        "--normalize",
        NORMALIZE,
        "--format",
        "npy",
        *meta_args,
        *hist_args,
    ]

    if preview_slices:
        cmd.append("--preview_slices")
        if preview_dir is not None:
            cmd.extend(["--preview_dir", str(preview_dir)])

    print(f"[INFO] Procesando {study_id} ...")
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    jobs = max(1, int(args.jobs))
    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"No se encuentra dataset_root: {dataset_root}")

    output_root = Path(args.output_root)
    output_subdir = str(args.output_subdir)

    # Detectar estudios dbt_* y real_dbt_*
    studies = sorted(
        [
            p for p in dataset_root.rglob("*")
            if p.is_dir()
            and (p.name.startswith("dbt_") or p.name.startswith("real_dbt_"))
            and find_img_dir(p) is not None
        ],
        key=lambda p: p.as_posix(),
    )
    if not studies:
        print(f"[WARN] No se encontraron estudios dbt_* en {dataset_root}")
        return

    if jobs == 1:
        for study_dir in studies:
            run_study(
                study_dir,
                dataset_root=dataset_root,
                output_root=output_root,
                output_subdir=output_subdir,
                preview_slices=bool(args.preview_slices),
                preview_dir=args.preview_dir,
            )
    else:
        print(f"[INFO] Paralelizando con jobs={jobs} (un proceso por estudio)")
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futures = {
                ex.submit(
                    run_study,
                    sd,
                    dataset_root=dataset_root,
                    output_root=output_root,
                    output_subdir=output_subdir,
                    preview_slices=bool(args.preview_slices),
                    preview_dir=args.preview_dir,
                ): sd
                for sd in studies
            }
            for fut in as_completed(futures):
                sd = futures[fut]
                try:
                    fut.result()
                except Exception as exc:
                    raise RuntimeError(f"Falló el preprocesado para {sd.name}") from exc

    print(f"[INFO] Listo. Salidas en {output_root / output_subdir}/<case_relpath>")


if __name__ == "__main__":
    main()

