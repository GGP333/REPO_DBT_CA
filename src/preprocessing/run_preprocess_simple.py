#!/usr/bin/env python3
"""
Ejecuta preprocess_simple.py sobre todos los estudios dbt_* / real_dbt_* en un dataset.

Uso:
    python scripts/run_preprocess_simple.py --dataset_root Dataset/Dataset_Hybrid --jobs 4
"""
from __future__ import annotations

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Parámetros del pipeline simple
TARGET_SIZE_XY = (512, 256)
CLIP_PERCENTILES = (0.5, 99.5)
NORMALIZE = "zscore"  # zscore | minmax | none


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ejecuta preprocess_simple.py para todos los estudios en un dataset."
    )
    p.add_argument(
        "--dataset_root",
        type=Path,
        required=True,
        help="Carpeta del dataset (ej: Dataset/Dataset_Hybrid)",
    )
    p.add_argument(
        "--output_root",
        type=Path,
        default=REPO_ROOT / "Dataset_Preprocessed",
        help="Carpeta base para salidas",
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Número de estudios a procesar en paralelo",
    )
    p.add_argument(
        "--normalize",
        choices=["zscore", "minmax", "none"],
        default=NORMALIZE,
        help="Modo de normalización",
    )
    return p.parse_args()


def find_img_dir(study_dir: Path) -> Path | None:
    for p in sorted(study_dir.iterdir()):
        if p.is_dir() and p.name.startswith("img_"):
            return p
    return None


def is_study_dir(name: str) -> bool:
    return name.startswith("dbt_") or name.startswith("real_dbt_")


def run_study(
    study_dir: Path,
    *,
    output_root: Path,
    dataset_name: str,
    normalize: str,
) -> None:
    study_id = study_dir.name
    img_dir = find_img_dir(study_dir)
    if img_dir is None:
        print(f"[WARN] {study_id}: sin carpeta img_*; se omite")
        return

    # Carpeta de salida: output_root/dataset_name/study_id/
    out_dir = output_root / dataset_name / study_id

    cmd = [
        "python",
        str(REPO_ROOT / "scripts" / "preprocess_simple.py"),
        "--input_dir", str(img_dir),
        "--output_dir", str(out_dir),
        "--target_size_xy", str(TARGET_SIZE_XY[0]), str(TARGET_SIZE_XY[1]),
        "--clip_percentiles", str(CLIP_PERCENTILES[0]), str(CLIP_PERCENTILES[1]),
        "--normalize", normalize,
    ]

    print(f"[INFO] Procesando {study_id} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] {study_id}:\n{result.stderr}")
        raise RuntimeError(f"Falló preprocesado de {study_id}")
    print(f"[OK] {study_id}")


def main() -> None:
    args = parse_args()
    jobs = max(1, int(args.jobs))
    dataset_root = Path(args.dataset_root)
    
    if not dataset_root.exists():
        raise FileNotFoundError(f"No se encuentra dataset_root: {dataset_root}")

    output_root = Path(args.output_root)
    dataset_name = dataset_root.name  # ej: Dataset_Hybrid
    normalize = args.normalize

    # Buscar estudios
    studies = sorted(
        [p for p in dataset_root.iterdir() if p.is_dir() and is_study_dir(p.name)],
        key=lambda p: p.name,
    )

    if not studies:
        print(f"[WARN] No se encontraron estudios dbt_*/real_dbt_* en {dataset_root}")
        return

    print(f"[INFO] Encontrados {len(studies)} estudios en {dataset_name}")
    print(f"[INFO] Salidas en: {output_root / dataset_name}/")

    if jobs == 1:
        for study_dir in studies:
            run_study(
                study_dir,
                output_root=output_root,
                dataset_name=dataset_name,
                normalize=normalize,
            )
    else:
        print(f"[INFO] Paralelizando con jobs={jobs}")
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futures = {
                ex.submit(
                    run_study,
                    sd,
                    output_root=output_root,
                    dataset_name=dataset_name,
                    normalize=normalize,
                ): sd
                for sd in studies
            }
            for fut in as_completed(futures):
                sd = futures[fut]
                try:
                    fut.result()
                except Exception as exc:
                    print(f"[ERROR] Falló {sd.name}: {exc}")

    print(f"\n[INFO] ¡Listo! Salidas en {output_root / dataset_name}/")


if __name__ == "__main__":
    main()

