#!/usr/bin/env python3
"""
Orquestador de entrenamientos.

Esta versión fue simplificada para ejecutar **solo nnUnet_original** y para ser
portable: no usa rutas absolutas hardcodeadas, sino rutas relativas al root del
repositorio (derivadas de `__file__`).

Uso típico (desde cualquier carpeta):
  python /ruta/al/repo/train_all_models.py

Opcionales útiles:
  python /ruta/al/repo/train_all_models.py --max-epochs 200 --num-workers 4
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = ROOT / "Dataset_Preprocessed" / "Dataset_Both_RealWorld"
MODELS = [
    {
        "name": "nnUnet_original_only_resize",
        "path": ROOT / "Models" / "nnUnet_original",
        "script": "train_unet_dbt.py",
    },
    {
        "name": "attention_unet",
        "path": ROOT / "Models" / "Attention U_Net",
        "script": "train_unet_dbt.py",
    },
    {
        "name": "unet_3d",
        "path": ROOT / "Models" / "U_Net BCE",
        "script": "train_unet_bce.py",
    },
]

# Salidas globales (fuera de las carpetas outputs internas)
GLOBAL_OUT = ROOT / "outputs_global"
GLOBAL_OUT.mkdir(parents=True, exist_ok=True)

# Overrides opcionales. Si no se proveen por CLI, se usa lo que diga el YAML del modelo.
MAX_EPOCHS: Optional[int] = None
NUM_WORKERS: Optional[int] = None
BATCH_SIZE: Optional[int] = None
VISUALIZATION_TOTAL = 70


def _next_available_dir(base: Path) -> Path:
    """
    Devuelve una carpeta disponible sin crearla.
    Si `base` ya existe, prueba con sufijos: base_1, base_2, ...
    """
    base = Path(base)
    if not base.exists():
        return base
    stem = base.name
    parent = base.parent
    max_i = 0
    try:
        for p in parent.iterdir():
            name = p.name
            if not name.startswith(stem + "_"):
                continue
            suf = name[len(stem) + 1 :]
            if suf.isdigit():
                max_i = max(max_i, int(suf))
    except Exception:
        # fallback defensivo: si no podemos listar, volvemos a búsqueda incremental
        max_i = 0

    i = max_i + 1
    cand = base.with_name(f"{stem}_{i}")
    while cand.exists():
        i += 1
        cand = base.with_name(f"{stem}_{i}")
    return cand


def _next_available_file(base: Path) -> Path:
    """
    Devuelve un path de archivo disponible sin crearlo.
    Si `base` ya existe, usa sufijos: name_1.ext, name_2.ext, ...
    """
    base = Path(base)
    if not base.exists():
        return base

    parent = base.parent
    stem = base.stem
    suffix = base.suffix  # incluye el "."

    max_i = 0
    try:
        for p in parent.iterdir():
            if not p.is_file():
                continue
            if p.suffix != suffix:
                continue
            name = p.stem  # sin extensión
            if not name.startswith(stem + "_"):
                continue
            suf = name[len(stem) + 1 :]
            if suf.isdigit():
                max_i = max(max_i, int(suf))
    except Exception:
        max_i = 0

    i = max_i + 1
    cand = parent / f"{stem}_{i}{suffix}"
    while cand.exists():
        i += 1
        cand = parent / f"{stem}_{i}{suffix}"
    return cand


def _natural_key(text: str) -> List[object]:
    import re

    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", text)]


def _print_dataset_samples(dataset_root: Path, *, n: int) -> None:
    """
    Muestra ejemplos de pares (img, mask) encontrados en Dataset_Preprocesed, sin spamear.
    Espera estructura por caso: <case>/<case>_img.npy y <case>/<case>_mask.npy
    """
    if n <= 0:
        return
    if not dataset_root.exists():
        print(f"[WARN] Dataset root no existe: {dataset_root}")
        return

    case_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()], key=lambda p: _natural_key(p.name))
    pairs: List[tuple[Path, Path]] = []
    for cd in case_dirs:
        cid = cd.name
        img = cd / f"{cid}_img.npy"
        msk = cd / f"{cid}_mask.npy"
        if img.exists() and msk.exists():
            pairs.append((img, msk))

    print(f"[INFO] Dataset: {dataset_root} -> {len(pairs)} casos con *_img.npy y *_mask.npy")
    for i, (img, msk) in enumerate(pairs[:n], start=1):
        print(f"[INFO] Sample {i}/{min(n, len(pairs))}:")
        print(f"  img:  {img}")
        print(f"  mask: {msk}")


def run_model(
    entry: Dict[str, object],
    *,
    max_epochs: Optional[int],
    num_workers: Optional[int],
    batch_size: Optional[int],
    dataset_root: Path,
    generate_viz: bool,
) -> Dict[str, Any]:
    name = str(entry["name"])
    model_dir = Path(entry["path"])  # type: ignore[arg-type]
    script = model_dir / str(entry.get("script", "train_unet_dbt.py"))
    config = model_dir / "configs" / "config.yaml"
    out_dir = _next_available_dir(GLOBAL_OUT / name.replace(" ", "_"))

    if not script.is_file():
        return {"model": name, "status": "missing_script", "detail": str(script)}
    if not config.is_file():
        return {"model": name, "status": "missing_config", "detail": str(config)}

    cmd: List[str] = [
        sys.executable,
        str(script),
        "--config", str(config),
        "--out-dir", str(out_dir),
        "--dataset-root", str(dataset_root),
    ]
    if max_epochs is not None:
        cmd += ["--max-epochs", str(int(max_epochs))]
    if num_workers is not None:
        cmd += ["--num-workers", str(int(num_workers))]
    if batch_size is not None:
        cmd += ["--batch-size", str(int(batch_size))]

    env = os.environ.copy()
    # Optimización de memoria CUDA
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Asegurar que los imports relativos encuentren la carpeta src del modelo
    env["PYTHONPATH"] = f"{model_dir / 'src'}" + (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    print(f"[INFO] Entrenando {name} -> {out_dir}")
    try:
        subprocess.run(cmd, check=True, env=env, cwd=model_dir)
        status = "ok"
    except subprocess.CalledProcessError as e:
        status = "error"
        print(f"[ERROR] Falló {name}: {e}", file=sys.stderr)

    viz_info = None
    if status == "ok" and generate_viz:
        viz_out_dir = out_dir / "extra_visualizations_70"
        try:
            # Import lazy: permite correr --help o entrenar sin matplotlib instalado
            from extra_visualizations import generate_visualizations  # type: ignore

            viz_info = generate_visualizations(
                preds_dir=out_dir / "preds_npy",
                test_preds_dir=out_dir / "test_preds_npy",
                dataset_root=dataset_root,
                out_dir=viz_out_dir,
                num_total=VISUALIZATION_TOTAL,
            )
            print(f"[INFO] Visualizaciones extras ({viz_info.get('generated', 0)}) -> {viz_out_dir}")
        except Exception as exc:
            viz_info = {"generated": 0, "error": str(exc)}
            print(f"[WARN] No se pudieron generar visualizaciones para {name}: {exc}", file=sys.stderr)
    # Limpieza de caché CUDA si torch está disponible
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

    return {
        "model": name,
        "status": status,
        "out_dir": str(out_dir.relative_to(ROOT)) if out_dir.is_absolute() else str(out_dir),
        "visualizations": viz_info,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ejecuta nnUnet_original y guarda salidas en outputs_global/")
    p.add_argument("--max-epochs", type=int, default=None, help="Override de max_epochs (si no se da, usa el YAML).")
    p.add_argument("--num-workers", type=int, default=None, help="Override de num_workers (si no se da, usa el YAML).")
    p.add_argument("--batch-size", type=int, default=None, help="Override de batch_size (si no se da, usa el YAML).")
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Override del dataset_root. Default: <repo>/Dataset_Preprocesed.",
    )
    p.add_argument("--skip-viz", action="store_true", help="No generar visualizaciones extras al terminar (más rápido).")
    p.add_argument(
        "--show-samples",
        type=int,
        default=3,
        help="Cuántos ejemplos de archivos img/mask mostrar al inicio (default: 3; 0 desactiva).",
    )
    p.add_argument(
        "--start-from",
        type=str,
        default=None,
        help="Ejecuta desde este modelo (por nombre; case-insensitive) en adelante. Ej: --start-from \"Swin UNet\"",
    )
    p.add_argument(
        "--list-models",
        action="store_true",
        help="Lista el orden de modelos y sale (no entrena).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if bool(args.list_models):
        print("[INFO] Orden de modelos:")
        for i, m in enumerate(MODELS):
            print(f"  {i:02d}: {m['name']}")
        return

    dataset_root = args.dataset_root if args.dataset_root is not None else DEFAULT_DATASET_ROOT
    _print_dataset_samples(dataset_root, n=int(args.show_samples))
    results = []

    models_to_run = MODELS
    if args.start_from:
        needle = str(args.start_from).strip().lower()
        start_idx = None
        for i, m in enumerate(MODELS):
            if str(m["name"]).strip().lower() == needle:
                start_idx = i
                break
        if start_idx is None:
            raise ValueError(
                f"--start-from '{args.start_from}' no coincide con ningún modelo. "
                f"Usa --list-models para ver los nombres exactos."
            )
        models_to_run = MODELS[start_idx:]

    for entry in models_to_run:
        res = run_model(
            entry,
            max_epochs=args.max_epochs,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            dataset_root=dataset_root,
            generate_viz=not bool(args.skip_viz),
        )
        results.append(res)
    summary_path = _next_available_file(GLOBAL_OUT / "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Resumen guardado en {summary_path}")


if __name__ == "__main__":
    main()

