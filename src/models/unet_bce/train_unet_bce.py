#!/usr/bin/env python3
"""
Entrenamiento UNet 3D (U_Net BCE) con pérdida BCE.
- Gráficas estandarizadas estilo paper con bandas ±1 desviación (fill_between)
"""
from __future__ import annotations
import argparse
import json
import sys
import torch
from pathlib import Path

_shared_dir = str(Path(__file__).resolve().parent.parent / "shared")
if _shared_dir not in sys.path:
    sys.path.insert(0, _shared_dir)

import matplotlib.pyplot as plt
import pandas as pd

from config_loader import load_config
from dataset_dbt import make_dataloaders, set_global_seeds
from io_utils import get_logger
from train_eval import fit, evaluate_from_checkpoint
from unet3d import UNet3D


def _plot_with_band(ax, epochs, series, label, color):
    """Grafica una serie con banda de ±1 desviación estándar (rolling window=5)."""
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
    """Genera gráficas estilo paper con bandas de desviación."""
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
    fig.suptitle("Curvas de entrenamiento/validación (U-Net BCE)", fontsize=14, fontweight="bold")
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


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--dataset-root", type=str, default=None, help="Override de dataset_root")
    ap.add_argument("--test-ids-file", type=str, default=None,
                    help="Path a un .txt con un study_id por línea para forzar el test set (excluye esos IDs de train/val).")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.max_epochs: cfg.max_epochs = args.max_epochs
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.num_workers: cfg.num_workers = args.num_workers
    if args.lr: cfg.lr = args.lr
    if args.weight_decay: cfg.weight_decay = args.weight_decay
    if args.threshold: cfg.threshold = args.threshold
    if args.dataset_root: cfg.dataset_root = args.dataset_root
    cfg.loss = "bce"

    out_root = Path(args.out_dir)
    ckpt_dir = out_root / "checkpoints"
    logs_dir = out_root / "logs"
    preds_dir = out_root / "preds_npy"
    for d in [ckpt_dir, logs_dir, preds_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger = get_logger("train_unet_bce")
    set_global_seeds(int(cfg.seed))

    force_test_ids = None
    if args.test_ids_file:
        with open(args.test_ids_file) as f:
            force_test_ids = {ln.strip() for ln in f if ln.strip() and not ln.startswith("#")}
        logger.info(f"Test IDs forzados ({len(force_test_ids)}) desde {args.test_ids_file}")

    train_loader, val_loader, test_loader, train_recs, val_recs, test_recs = make_dataloaders(cfg, force_test_ids=force_test_ids)
    logger.info(f"Split: train={len(train_recs)} | val={len(val_recs)} | test={len(test_recs)}")
    assert not (set(r.study_id for r in test_recs) & set(r.study_id for r in train_recs)), "LEAK: test ∩ train != ∅"
    assert not (set(r.study_id for r in test_recs) & set(r.study_id for r in val_recs)), "LEAK: test ∩ val != ∅"

    model = UNet3D(in_ch=1, base_ch=32, levels=4)
    logger.info(f"Parámetros del modelo: {sum(p.numel() for p in model.parameters())/1e6:.3f}M")

    # Traceability: guardar config de ejecución
    try:
        from traceability import save_run_config
        save_run_config(out_root, cfg, model,
                        train_records=train_recs, val_records=val_recs, test_records=test_recs)
        logger.info(f"run_config.json guardado en {out_root}")
    except Exception as e:
        logger.warning(f"No se pudo guardar run_config.json: {e}")

    import time as _time
    _train_start = _time.time()

    history = fit(model, train_loader, val_loader, cfg, out_ckpt_dir=str(ckpt_dir), out_logs_dir=str(logs_dir), preds_out_dir=str(preds_dir))
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

    def ctor():
        return UNet3D(in_ch=1, base_ch=32, levels=4)

    # Solo evaluar si hay checkpoint disponible
    if eval_ckpt:
        try:
            val_metrics = evaluate_from_checkpoint(ctor, eval_ckpt, val_loader, threshold=float(cfg.threshold))
            with open(logs_dir / "val_metrics.json", "w") as f:
                json.dump(val_metrics, f, indent=2)
            logger.info(f"Val metrics: {val_metrics}")
        except Exception as e:
            logger.error(f"Error durante evaluación val: {e}")
            import traceback
            traceback.print_exc()
        
        # Evaluar en test si existe (real_dbt_* en Dataset_Hybrid)
        if test_loader is not None:
            try:
                test_preds_dir = out_root / "test_preds_npy"
                test_preds_dir.mkdir(parents=True, exist_ok=True)
                test_metrics = evaluate_from_checkpoint(
                    ctor, eval_ckpt, test_loader,
                    threshold=float(cfg.threshold),
                    preds_out_dir=str(test_preds_dir)
                )
                with open(logs_dir / "test_metrics.json", "w") as f:
                    json.dump(test_metrics, f, indent=2)
                logger.info(f"Test metrics (real_dbt_*): {test_metrics}")
            except Exception as e:
                logger.error(f"Error durante evaluación test: {e}")
                import traceback
                traceback.print_exc()
    else:
        logger.warning("Saltando evaluación: no hay checkpoint disponible")

    # Gráficas estilo paper con fill_between
    try:
        plot_paper_curves(logs_dir / "metrics.csv", logs_dir)
        logger.info(f"Curvas guardadas en: {logs_dir}")
    except Exception as e:
        logger.error(f"Error al generar gráficas: {e}")
        import traceback
        traceback.print_exc()

    # Samples PNG de validación
    n_samples = int(getattr(cfg, "save_samples", 5))
    if n_samples > 0 and eval_ckpt:
        try:
            from visualization import generate_validation_samples
            from train_eval import get_device
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

    # Traceability: guardar resumen final
    try:
        from traceability import save_run_summary
        _train_elapsed = _time.time() - _train_start
        save_run_summary(
            out_root,
            history=history.get("history", []),
            best_ckpt=eval_ckpt,
            val_metrics=val_metrics if 'val_metrics' in dir() else None,
            test_metrics=test_metrics if 'test_metrics' in dir() else None,
            total_time_s=_train_elapsed,
        )
        logger.info(f"run_summary.json guardado en {out_root}")
    except Exception as e:
        logger.warning(f"No se pudo guardar run_summary.json: {e}")


if __name__ == "__main__":
    main()

