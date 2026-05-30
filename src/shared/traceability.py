"""
Traceability logging para entrenamientos.
Genera run_config.json al inicio y run_summary.json al final.
"""
from __future__ import annotations
import json
import os
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def _get_system_info() -> Dict[str, Any]:
    """Recopila información del sistema."""
    info = {
        "hostname": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        info["cuda_version"] = torch.version.cuda or "N/A"
    try:
        import subprocess
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            info["git_commit"] = result.stdout.strip()
    except Exception:
        pass
    return info


def _get_dataset_stats(records, label: str = "train") -> Dict[str, Any]:
    """Estadísticas del dataset: IDs, foreground ratio, etc."""
    stats: Dict[str, Any] = {
        "n_studies": len(records),
        "study_ids": [r.study_id for r in records],
    }
    fg_ratios = []
    for rec in records:
        try:
            m = np.load(rec.mask_path)
            fg = float((m > 0).sum()) / max(1, m.size) * 100
            fg_ratios.append(fg)
        except Exception:
            fg_ratios.append(0.0)
    if fg_ratios:
        stats["fg_ratio_mean_pct"] = float(np.mean(fg_ratios))
        stats["fg_ratio_median_pct"] = float(np.median(fg_ratios))
        stats["n_with_foreground"] = int(sum(1 for r in fg_ratios if r > 0))
        stats["n_without_foreground"] = len(fg_ratios) - stats["n_with_foreground"]
    return stats


def save_run_config(
    out_dir: str | Path,
    cfg,
    model: torch.nn.Module,
    train_records=None,
    val_records=None,
    test_records=None,
) -> Path:
    """
    Guarda run_config.json al inicio del entrenamiento.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_dict: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "system": _get_system_info(),
    }

    # Config completa
    if hasattr(cfg, "__dict__"):
        config_dict["hyperparameters"] = {k: str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v
                                           for k, v in vars(cfg).items()}
    elif isinstance(cfg, dict):
        config_dict["hyperparameters"] = cfg
    else:
        config_dict["hyperparameters"] = str(cfg)

    # Model info
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config_dict["model"] = {
        "class": type(model).__name__,
        "total_params": n_params,
        "trainable_params": n_trainable,
        "total_params_M": round(n_params / 1e6, 3),
    }

    # Dataset stats
    config_dict["dataset"] = {}
    if train_records:
        config_dict["dataset"]["train"] = _get_dataset_stats(train_records, "train")
    if val_records:
        config_dict["dataset"]["val"] = _get_dataset_stats(val_records, "val")
    if test_records:
        config_dict["dataset"]["test"] = _get_dataset_stats(test_records, "test")

    path = out_dir / "run_config.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, default=str)
    return path


def save_run_summary(
    out_dir: str | Path,
    history: List[Dict[str, Any]],
    best_ckpt: Optional[str] = None,
    val_metrics: Optional[Dict[str, float]] = None,
    test_metrics: Optional[Dict[str, float]] = None,
    total_time_s: Optional[float] = None,
    sample_paths: Optional[List[str]] = None,
) -> Path:
    """
    Guarda run_summary.json al final del entrenamiento.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "total_epochs": len(history),
        "total_time_seconds": total_time_s,
        "total_time_human": f"{total_time_s/3600:.1f}h" if total_time_s else None,
    }

    # Mejor época
    if history:
        best_epoch_data = max(history, key=lambda h: h.get("val_dice", 0))
        summary["best_epoch"] = {
            "epoch": best_epoch_data.get("epoch"),
            "val_dice": best_epoch_data.get("val_dice"),
            "train_loss": best_epoch_data.get("train_loss"),
        }
        last_epoch_data = history[-1]
        summary["last_epoch"] = {
            "epoch": last_epoch_data.get("epoch"),
            "val_dice": last_epoch_data.get("val_dice"),
            "train_loss": last_epoch_data.get("train_loss"),
        }

    summary["best_checkpoint"] = best_ckpt
    summary["val_metrics"] = val_metrics
    summary["test_metrics"] = test_metrics
    summary["sample_pngs"] = [str(p) for p in sample_paths] if sample_paths else []

    path = out_dir / "run_summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    return path

