#!/usr/bin/env python3
"""
Script para ejecutar entrenamientos cortos de prueba (3 épocas) con todos los modelos y datasets.
Total: 3 modelos × 4 datasets = 12 entrenamientos.

Uso:
    python run_test_trainings.py
    python run_test_trainings.py --epochs 5  # cambiar épocas
    python run_test_trainings.py --models nnUnet_original  # solo un modelo
    python run_test_trainings.py --datasets Dataset_Both_RealWorld Dataset_Both  # solo algunos datasets
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

ROOT = Path(__file__).resolve().parent

# Configuración de modelos
MODELS = [
    {
        "name": "nnUnet_original",
        "path": ROOT / "Models" / "nnUnet_original",
        "script": "train_unet_dbt.py",
        "config": "configs/config.yaml",
    },
    {
        "name": "Attention_UNet",
        "path": ROOT / "Models" / "Attention U_Net",
        "script": "train_unet_dbt.py",
        "config": "configs/config.yaml",
    },
    {
        "name": "UNet_BCE",
        "path": ROOT / "Models" / "U_Net BCE",
        "script": "train_unet_bce.py",
        "config": "configs/config.yaml",
    },
]

# Datasets preprocesados disponibles
DATASETS = [
    "Dataset_Both_RealWorld",
    "Dataset_Both",
    "Dataset_small_tumor",
    "Dataset_large_tumor",
]

DEFAULT_EPOCHS = 3


def run_training(
    model: Dict[str, Any],
    dataset_name: str,
    dataset_root: Path,
    output_root: Path,
    epochs: int,
) -> Dict[str, Any]:
    """Ejecuta un entrenamiento y devuelve el resultado."""
    import os
    
    model_name = model["name"]
    model_path = model["path"]
    script = model_path / model["script"]
    config = model_path / model["config"]
    
    # Directorio de salida para este experimento (siempre absoluto)
    out_dir = output_root.resolve() / f"{model_name}_{dataset_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Construir comando
    cmd = [
        sys.executable,
        str(script),
        "--config", str(config),
        "--out-dir", str(out_dir),
        "--dataset-root", str(dataset_root / dataset_name),
        "--max-epochs", str(epochs),
    ]
    
    print(f"\n{'='*70}")
    print(f"🚀 ENTRENAMIENTO: {model_name} + {dataset_name}")
    print(f"{'='*70}")
    print(f"   Script: {script}")
    print(f"   Config: {config}")
    print(f"   Dataset: {dataset_root / dataset_name}")
    print(f"   Output: {out_dir}")
    print(f"   Epochs: {epochs}")
    print(f"   Comando: {' '.join(cmd)}")
    print("-" * 70)
    
    start_time = time.time()
    
    # Configurar PYTHONPATH para incluir el directorio del modelo Y su carpeta src/
    env = os.environ.copy()
    src_path = model_path / "src"
    pythonpath = f"{model_path}:{src_path}"
    if "PYTHONPATH" in env:
        pythonpath = f"{pythonpath}:{env['PYTHONPATH']}"
    env["PYTHONPATH"] = pythonpath
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(model_path),
            env=env,
            check=True,
            capture_output=False,  # Mostrar output en tiempo real
        )
        elapsed = time.time() - start_time
        status = "✅ SUCCESS"
        error = None
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        status = "❌ FAILED"
        error = str(e)
    except Exception as e:
        elapsed = time.time() - start_time
        status = "❌ ERROR"
        error = str(e)
    
    print(f"\n{status} | Tiempo: {elapsed:.1f}s | {model_name} + {dataset_name}")
    
    # Extraer métricas del entrenamiento si existen
    metrics = {}
    val_metrics_path = out_dir / "logs" / "val_metrics.json"
    metrics_csv_path = out_dir / "logs" / "metrics.csv"
    
    # Intentar leer val_metrics.json primero
    if val_metrics_path.exists():
        try:
            import json as json_mod
            with open(val_metrics_path, "r") as f:
                val_data = json_mod.load(f)
                metrics["val_dice"] = val_data.get("dice")
                metrics["val_iou"] = val_data.get("iou")
                metrics["val_precision"] = val_data.get("precision")
                metrics["val_recall"] = val_data.get("recall")
                metrics["val_f1"] = val_data.get("f1")
        except Exception:
            pass
    
    # Si no hay val_metrics.json, intentar leer la última fila de metrics.csv
    if not metrics and metrics_csv_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(metrics_csv_path)
            if len(df) > 0:
                last_row = df.iloc[-1]
                metrics["train_loss"] = float(last_row.get("train_loss", 0))
                metrics["val_dice"] = float(last_row.get("val_dice", 0))
                metrics["val_iou"] = float(last_row.get("val_iou", 0))
                metrics["val_precision"] = float(last_row.get("val_precision", 0))
                metrics["val_recall"] = float(last_row.get("val_recall", 0))
                metrics["val_f1"] = float(last_row.get("val_f1", 0))
        except Exception:
            pass
    
    return {
        "model": model_name,
        "dataset": dataset_name,
        "status": status,
        "elapsed_seconds": elapsed,
        "error": error,
        "output_dir": str(out_dir),
        "metrics": metrics if metrics else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Entrenamientos cortos de prueba")
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Número de épocas por entrenamiento (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[m["name"] for m in MODELS],
        help="Modelos a entrenar (default: todos)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        help="Datasets a usar (default: todos)",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "Dataset_Preprocessed",
        help="Carpeta con datasets preprocesados",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs_test",
        help="Carpeta base para salidas de prueba",
    )
    args = parser.parse_args()
    
    # Filtrar modelos y datasets según argumentos
    models_to_run = MODELS
    if args.models:
        models_to_run = [m for m in MODELS if m["name"] in args.models]
    
    datasets_to_run = DATASETS
    if args.datasets:
        datasets_to_run = [d for d in DATASETS if d in args.datasets]
    
    # Verificar que existan los datasets (resolver a absolutas)
    dataset_root = args.dataset_root.resolve()
    args.output_root = args.output_root.resolve()
    if not dataset_root.exists():
        print(f"❌ ERROR: No existe la carpeta de datasets: {dataset_root}")
        sys.exit(1)
    
    available_datasets = [d for d in datasets_to_run if (dataset_root / d).exists()]
    missing_datasets = set(datasets_to_run) - set(available_datasets)
    
    if missing_datasets:
        print(f"⚠️  WARN: Datasets no encontrados (se omiten): {missing_datasets}")
    
    if not available_datasets:
        print("❌ ERROR: No hay datasets disponibles para entrenar")
        sys.exit(1)
    
    # Resumen de lo que se va a ejecutar
    total_runs = len(models_to_run) * len(available_datasets)
    
    print("\n" + "=" * 70)
    print("📋 RESUMEN DE ENTRENAMIENTOS DE PRUEBA")
    print("=" * 70)
    print(f"   Modelos ({len(models_to_run)}): {[m['name'] for m in models_to_run]}")
    print(f"   Datasets ({len(available_datasets)}): {available_datasets}")
    print(f"   Épocas por entrenamiento: {args.epochs}")
    print(f"   Total de entrenamientos: {total_runs}")
    print(f"   Salidas en: {args.output_root}")
    print("=" * 70)
    
    # Crear carpeta de salida
    args.output_root.mkdir(parents=True, exist_ok=True)
    
    # Ejecutar entrenamientos
    results: List[Dict[str, Any]] = []
    start_total = time.time()
    
    for i, model in enumerate(models_to_run, 1):
        for j, dataset_name in enumerate(available_datasets, 1):
            run_idx = (i - 1) * len(available_datasets) + j
            print(f"\n\n🔄 Progreso: {run_idx}/{total_runs}")
            
            result = run_training(
                model=model,
                dataset_name=dataset_name,
                dataset_root=dataset_root,
                output_root=args.output_root,
                epochs=args.epochs,
            )
            results.append(result)
    
    total_elapsed = time.time() - start_total
    
    # Resumen final
    print("\n\n" + "=" * 70)
    print("📊 RESUMEN FINAL")
    print("=" * 70)
    
    success_count = sum(1 for r in results if "SUCCESS" in r["status"])
    failed_count = total_runs - success_count
    
    print(f"\n   Total: {total_runs} entrenamientos")
    print(f"   ✅ Exitosos: {success_count}")
    print(f"   ❌ Fallidos: {failed_count}")
    print(f"   ⏱️  Tiempo total: {total_elapsed/60:.1f} minutos")
    
    print("\n   Detalle por entrenamiento:")
    print("   " + "-" * 80)
    for r in results:
        status_emoji = "✅" if "SUCCESS" in r["status"] else "❌"
        metrics_str = ""
        if r.get("metrics") and r["metrics"].get("val_dice") is not None:
            dice = r["metrics"]["val_dice"]
            metrics_str = f" | dice={dice:.4f}"
        print(f"   {status_emoji} {r['model']:20s} + {r['dataset']:25s} | {r['elapsed_seconds']:6.1f}s{metrics_str}")
        if r["error"]:
            print(f"      └─ Error: {r['error'][:60]}...")
    
    print("\n   " + "-" * 66)
    print(f"   📁 Resultados guardados en: {args.output_root}")
    print("=" * 70)
    
    # Guardar resumen en JSON
    import json
    summary_path = args.output_root / "training_summary.json"
    summary = {
        "timestamp": datetime.now().isoformat(),
        "epochs": args.epochs,
        "total_runs": total_runs,
        "success_count": success_count,
        "failed_count": failed_count,
        "total_elapsed_seconds": total_elapsed,
        "results": results,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n   📝 Resumen guardado en: {summary_path}")


if __name__ == "__main__":
    main()

