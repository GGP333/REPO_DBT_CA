#!/usr/bin/env python3
"""
Script de entrenamiento/evaluación para Attention U-Net 3D (versión notebook -> script).
- Split reproducible en train/val/test
- Métricas incluyen Hausdorff Distance
- Gráficas estandarizadas estilo paper con bandas ±1 desviación (fill_between)
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import random
import warnings
from pathlib import Path
from typing import List, Tuple

_shared_dir = str(Path(__file__).resolve().parent.parent / "shared")
if _shared_dir not in sys.path:
    sys.path.insert(0, _shared_dir)

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config_loader import load_config
from dataset_dbt import DBTVolumeDataset, collate_pad, set_global_seeds
from io_utils import discover_studies, get_logger, StudyRecord
from train_eval import evaluate_from_checkpoint, fit, get_device, validate_one_epoch
from attention_unet3d import AttentionUNet3D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenamiento Attention U-Net 3D en DBT (.npy) con métricas extendidas.")
    parser.add_argument("--config", default="configs/config.yaml", help="Ruta al YAML de configuración.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Fracción para entrenamiento.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fracción para validación.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Fracción para test.")
    parser.add_argument("--batch-size", type=int, help="Override de batch_size.")
    parser.add_argument("--num-workers", type=int, help="Override de num_workers.")
    parser.add_argument("--max-epochs", type=int, help="Override de max_epochs.")
    parser.add_argument("--lr", type=float, help="Override de learning rate.")
    parser.add_argument("--weight-decay", type=float, help="Override de weight decay.")
    parser.add_argument("--threshold", type=float, help="Override de umbral de binarización.")
    parser.add_argument("--base-ch", type=int, default=32, help="Canales base del modelo.")
    parser.add_argument("--levels", type=int, default=4, help="Niveles de la U.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout de Attention U-Net.")
    parser.add_argument("--amp", action="store_true", help="Forzar AMP activado.")
    parser.add_argument("--no-amp", action="store_true", help="Forzar AMP desactivado.")
    parser.add_argument("--out-dir", default="outputs", help="Carpeta base de salidas.")
    parser.add_argument("--dataset-root", type=str, default=None, help="Override de dataset_root.")
    parser.add_argument("--checkpoint", action="store_true", default=True, 
                        help="Usar gradient checkpointing para reducir memoria (activado por defecto).")
    parser.add_argument("--no-checkpoint", action="store_true", 
                        help="Desactivar gradient checkpointing.")
    return parser.parse_args()


def _normalize_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[float, float, float]:
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("Las proporciones deben ser positivas.")
    if total > 1.0:
        warnings.warn(f"Ratios suman {total:.2f}>1; se renormalizan para que sumen 1.", stacklevel=2)
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total
    return train_ratio, val_ratio, test_ratio


def split_records(records: List[StudyRecord],
                  train_ratio: float,
                  val_ratio: float,
                  test_ratio: float,
                  seed: int) -> Tuple[List[StudyRecord], List[StudyRecord], List[StudyRecord]]:
    """
    Split de records en train/val/test.
    
    CASO ESPECIAL (Dataset_Both_RealWorld):
    - Test: 50% de real_dbt_* (solo datos reales)
    - Train/Val: dbt_* + 50% restante de real_dbt_* (mezclados)
    """
    rng = random.Random(int(seed))
    by_id = {r.study_id: r for r in records}
    
    # Separar casos "real_dbt_*" de "dbt_*"
    real_ids = sorted([r.study_id for r in records if r.study_id.startswith("real_dbt_")])
    synth_ids = sorted([r.study_id for r in records if not r.study_id.startswith("real_dbt_")])
    
    # Si hay casos real_dbt_*, aplicar lógica especial
    if real_ids:
        rng.shuffle(real_ids)
        n_real = len(real_ids)
        
        # Test = 50% de real_dbt_* (al menos 1)
        n_test = max(1, n_real // 2)
        test_ids = set(real_ids[:n_test])
        real_for_trainval = real_ids[n_test:]  # real_dbt_* restantes para train/val
        
        # Combinar dbt_* + real_dbt_* restantes para train/val
        trainval_ids = synth_ids + real_for_trainval
        rng.shuffle(trainval_ids)
        n_trainval = len(trainval_ids)
        
        tv_total = train_ratio + val_ratio
        adj_train = train_ratio / tv_total if tv_total > 0 else 0.8
        
        n_train = max(1, int(round(n_trainval * adj_train)))
        n_val = max(1, n_trainval - n_train)
        
        train_ids = set(trainval_ids[:n_train])
        val_ids = set(trainval_ids[n_train:n_train + n_val])
        
        real_in_train = len([i for i in train_ids if i.startswith("real_dbt_")])
        real_in_val = len([i for i in val_ids if i.startswith("real_dbt_")])
        
        print(f"[INFO] Dataset con real_dbt_*:")
        print(f"       test={len(test_ids)} (solo real_dbt_*)")
        print(f"       train={len(train_ids)} ({real_in_train} real + {len(train_ids)-real_in_train} synth)")
        print(f"       val={len(val_ids)} ({real_in_val} real + {len(val_ids)-real_in_val} synth)")
    else:
        # Comportamiento original: split normal
        train_ratio, val_ratio, test_ratio = _normalize_ratios(train_ratio, val_ratio, test_ratio)
        
        ids = sorted([r.study_id for r in records])
        rng.shuffle(ids)
        n = len(ids)

        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        n_test = max(0, int(round(n * test_ratio)))

        # Ajuste si la suma excede n
        while n_train + n_val + n_test > n:
            if n_train >= n_val and n_train > 1:
                n_train -= 1
            elif n_val >= n_test and n_val > 1:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1
            else:
                break

        train_ids = set(ids[:n_train])
        val_ids = set(ids[n_train:n_train + n_val])
        test_ids = set(ids[n_train + n_val:n_train + n_val + n_test])
    
    all_ids = list(train_ids) + list(val_ids) + list(test_ids)
    return (
        [by_id[i] for i in all_ids if i in train_ids],
        [by_id[i] for i in all_ids if i in val_ids],
        [by_id[i] for i in all_ids if i in test_ids],
    )


def make_loader(records: List[StudyRecord], cfg, shuffle: bool) -> DataLoader:
    ds = DBTVolumeDataset(records, logger=get_logger("loader"))
    return DataLoader(
        ds,
        batch_size=int(cfg.batch_size),
        shuffle=shuffle,
        num_workers=int(cfg.num_workers),
        pin_memory=True,
        collate_fn=collate_pad,
    )


def _plot_with_band(ax, epochs, series, label, color):
    rolling = series.rolling(window=5, min_periods=1)
    mean = rolling.mean()
    std = rolling.std(ddof=0).fillna(0.0)
    ax.plot(epochs, series, color=color, lw=2, label=label)
    ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.18, label="±1 std")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(label)
    ax.grid(True, ls="--", alpha=0.55)
    ax.legend(fontsize=9)


def plot_paper_curves(log_csv: Path, out_dir: Path) -> None:
    if not log_csv.exists():
        return
    df = pd.read_csv(log_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = df["epoch"]
    
    # Todas las métricas a graficar
    all_metrics = [
        ("train_loss", "Train Loss", "#1f77b4"),
        ("val_dice", "Val Dice (↑)", "#d62728"),
        ("val_iou", "Val IoU (↑)", "#9467bd"),
        ("val_precision", "Val Precision (↑)", "#2ca02c"),
        ("val_recall", "Val Recall (↑)", "#ff7f0e"),
        ("val_f1", "Val F1 (↑)", "#17becf"),
        ("val_accuracy", "Val Accuracy (↑)", "#7f7f7f"),
        ("val_mAP", "Val mAP (↑)", "#e377c2"),
        ("val_hausdorff", "Val Hausdorff (↓)", "#8c564b"),
        ("val_hausdorff_95", "Val HD95 (↓)", "#bcbd22"),
        ("val_fp_per_volume", "Val FP/Volume (↓)", "#ff9896"),
        ("val_fp_per_image", "Val FP/Image (↓)", "#c5b0d5"),
        ("val_ap", "Val AP (↑)", "#98df8a"),
    ]
    
    # === 1. GRÁFICOS INDIVIDUALES ===
    indiv_dir = out_dir / "individual_plots"
    indiv_dir.mkdir(parents=True, exist_ok=True)
    
    for col, label, color in all_metrics:
        if col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        _plot_with_band(ax, epochs, df[col], label, color)
        ax.set_title(label, fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(indiv_dir / f"{col}.png", dpi=200)
        plt.close(fig)
    
    # LR individual
    if "lr" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(epochs, df["lr"], color="#1f77b4", lw=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule", fontsize=13, fontweight="bold")
        ax.grid(True, ls="--", alpha=0.55)
        fig.tight_layout()
        fig.savefig(indiv_dir / "lr.png", dpi=200)
        plt.close(fig)
    
    # === 2. PANEL PRINCIPAL: métricas clásicas (2x4) ===
    metrics_main = [
        ("train_loss", "Train Loss", "#1f77b4"),
        ("val_dice", "Val Dice (↑)", "#d62728"),
        ("val_iou", "Val IoU (↑)", "#9467bd"),
        ("val_precision", "Val Precision (↑)", "#2ca02c"),
        ("val_recall", "Val Recall (↑)", "#ff7f0e"),
        ("val_f1", "Val F1 (↑)", "#17becf"),
        ("val_accuracy", "Val Accuracy (↑)", "#7f7f7f"),
        ("val_mAP", "Val mAP (↑)", "#e377c2"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=True)
    axes = axes.ravel()
    for (col, label, color), ax in zip(metrics_main, axes):
        if col not in df.columns:
            ax.axis("off")
            continue
        _plot_with_band(ax, epochs, df[col], label, color)
    fig.suptitle("Curvas de entrenamiento/validación (Attention U-Net)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "paper_curves.png", dpi=200)
    plt.close(fig)
    
    # === 3. PANEL SECUNDARIO: métricas adicionales (Hausdorff, FP, AP) ===
    metrics_extra = [
        ("val_hausdorff", "Val Hausdorff (↓)", "#8c564b"),
        ("val_hausdorff_95", "Val HD95 (↓)", "#bcbd22"),
        ("val_fp_per_volume", "Val FP/Volume (↓)", "#ff9896"),
        ("val_fp_per_image", "Val FP/Image (↓)", "#c5b0d5"),
        ("val_ap", "Val AP (↑)", "#98df8a"),
    ]
    
    fig2, axes2 = plt.subplots(1, 5, figsize=(17, 3.5))
    for (col, label, color), ax in zip(metrics_extra, axes2):
        if col not in df.columns:
            ax.axis("off")
            continue
        _plot_with_band(ax, epochs, df[col], label, color)
    fig2.suptitle("Métricas adicionales (Hausdorff, FP & AP)", fontsize=14, fontweight="bold")
    fig2.tight_layout(rect=[0, 0, 1, 0.92])
    fig2.savefig(out_dir / "paper_curves_extra.png", dpi=200)
    plt.close(fig2)

    # === 4. LR aparte (legacy) ===
    if "lr" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, df["lr"], color="#1f77b4", lw=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning rate")
        ax.grid(True, ls="--", alpha=0.55)
        fig.tight_layout()
        fig.savefig(out_dir / "lr_curve.png", dpi=200)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # Overrides
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.num_workers:
        cfg.num_workers = args.num_workers
    if args.max_epochs:
        cfg.max_epochs = args.max_epochs
    if args.lr:
        cfg.lr = args.lr
    if args.weight_decay:
        cfg.weight_decay = args.weight_decay
    if args.threshold:
        cfg.threshold = args.threshold
    if args.amp:
        cfg.amp = True
    if args.no_amp:
        cfg.amp = False
    if args.dataset_root:
        cfg.dataset_root = args.dataset_root

    out_root = Path(args.out_dir)
    ckpt_dir = out_root / "checkpoints"
    logs_dir = out_root / "logs"
    preds_dir = out_root / "preds_npy"
    test_preds_dir = out_root / "test_preds_npy"
    for d in [ckpt_dir, logs_dir, preds_dir, test_preds_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger = get_logger("train_attention_unet_script")
    set_global_seeds(int(cfg.seed))

    # Descubrir estudios y split
    records = discover_studies(cfg.dataset_root, logger=logger)
    train_records, val_records, test_records = split_records(
        records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=int(cfg.seed),
    )
    logger.info(f"Split: train={len(train_records)} | val={len(val_records)} | test={len(test_records)}")

    train_loader = make_loader(train_records, cfg, shuffle=True)
    val_loader = make_loader(val_records, cfg, shuffle=False)
    test_loader = make_loader(test_records, cfg, shuffle=False) if test_records else None

    # Configurar gradient checkpointing
    use_checkpointing = args.checkpoint and not args.no_checkpoint
    logger.info(f"Gradient checkpointing: {'activado' if use_checkpointing else 'desactivado'}")

    # Modelo con gradient checkpointing para reducir memoria
    model = AttentionUNet3D(
        in_ch=1,
        base_ch=int(args.base_ch),
        levels=int(args.levels),
        dropout_rate=float(args.dropout),
        use_checkpointing=use_checkpointing,
    )
    logger.info(f"Parámetros del modelo: {sum(p.numel() for p in model.parameters())/1e6:.3f}M")

    # Traceability: guardar config de ejecución
    try:
        from traceability import save_run_config
        save_run_config(out_root, cfg, model,
                        train_records=train_records, val_records=val_records, test_records=test_records)
        logger.info(f"run_config.json guardado en {out_root}")
    except Exception as e:
        logger.warning(f"No se pudo guardar run_config.json: {e}")

    import time as _time
    _train_start = _time.time()

    # Entrenamiento + validación
    history = fit(
        model,
        train_loader,
        val_loader,
        cfg,
        out_ckpt_dir=str(ckpt_dir),
        out_logs_dir=str(logs_dir),
        preds_out_dir=str(preds_dir),
    )
    best_ckpt = history["best_ckpt"]
    last_ckpt = history.get("last_ckpt")
    
    # Verificar qué checkpoint usar (best si existe, sino last)
    if Path(best_ckpt).exists():
        eval_ckpt = best_ckpt
        logger.info(f"Mejor checkpoint: {best_ckpt}")
    elif last_ckpt and Path(last_ckpt).exists():
        eval_ckpt = last_ckpt
        logger.warning(f"best.pt no existe, usando last.pt: {last_ckpt}")
    else:
        eval_ckpt = None
        logger.warning("No hay checkpoint disponible para evaluación")

    # Evaluación validación agregada (con manejo robusto de errores)
    def ctor():
        return AttentionUNet3D(
            in_ch=1,
            base_ch=int(args.base_ch),
            levels=int(args.levels),
            dropout_rate=float(args.dropout),
            use_checkpointing=use_checkpointing,
        )

    val_metrics = None
    test_metrics = None

    # Solo evaluar si hay checkpoint disponible
    if eval_ckpt:
        try:
            val_metrics = evaluate_from_checkpoint(
                ctor,
                eval_ckpt,
                val_loader,
                threshold=float(cfg.threshold),
                preds_out_dir=str(preds_dir),
            )
            with open(logs_dir / "val_metrics.json", "w") as f:
                json.dump(val_metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Error durante evaluación de validación: {e}")
            import traceback
            traceback.print_exc()

        # Evaluación test (si existe)
        if test_loader is not None:
            try:
                device = get_device()
                test_model = ctor()
                data = torch.load(eval_ckpt, map_location=device)
                test_model.load_state_dict(data["model_state"])
                test_model.to(device).eval()
                test_metrics = validate_one_epoch(
                    test_model,
                    test_loader,
                    device,
                    threshold=float(cfg.threshold),
                    preds_out_dir=str(test_preds_dir),
                )
                with open(logs_dir / "test_metrics.json", "w") as f:
                    json.dump(test_metrics, f, indent=2)
                logger.info(f"Métricas test: {test_metrics}")
            except Exception as e:
                logger.error(f"Error durante evaluación de test: {e}")
                import traceback
                traceback.print_exc()
    else:
        logger.warning("Saltando evaluación: no hay checkpoint disponible")

    # Gráficas estilo paper (con manejo robusto de errores)
    try:
        plot_paper_curves(logs_dir / "metrics.csv", logs_dir)
    except Exception as e:
        logger.error(f"Error al generar gráficas: {e}")
        import traceback
        traceback.print_exc()

    # Samples PNG de validación
    n_samples = int(getattr(cfg, "save_samples", 5))
    if n_samples > 0 and eval_ckpt:
        try:
            from visualization import generate_validation_samples
            device = get_device()
            model_for_samples = ctor()
            ckpt_data = torch.load(eval_ckpt, map_location="cpu")
            model_for_samples.load_state_dict(ckpt_data["model_state"])
            model_for_samples.to(device)
            model_for_samples.eval()
            generate_validation_samples(
                model_for_samples, val_loader, device, out_root,
                n_samples=n_samples, threshold=float(cfg.threshold)
            )
            del model_for_samples
        except Exception as e:
            logger.error(f"Error al generar samples PNG: {e}")
            import traceback
            traceback.print_exc()

    # Resumen final
    logger.info("=== Resumen ===")
    if val_metrics:
        logger.info(f"Val metrics: {json.dumps(val_metrics, indent=2)}")
    else:
        logger.warning("No se pudieron calcular métricas de validación.")
    if test_metrics:
        logger.info(f"Test metrics: {json.dumps(test_metrics, indent=2)}")
    logger.info(f"Curvas guardadas en: {logs_dir}")
    logger.info(f"Predicciones val en: {preds_dir}")
    if test_loader is not None:
        logger.info(f"Predicciones test en: {test_preds_dir}")

    # Traceability: guardar resumen final
    try:
        from traceability import save_run_summary
        _train_elapsed = _time.time() - _train_start
        save_run_summary(
            out_root,
            history=history.get("history", []),
            best_ckpt=eval_ckpt,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            total_time_s=_train_elapsed,
        )
        logger.info(f"run_summary.json guardado en {out_root}")
    except Exception as e:
        logger.warning(f"No se pudo guardar run_summary.json: {e}")


if __name__ == "__main__":
    main()

