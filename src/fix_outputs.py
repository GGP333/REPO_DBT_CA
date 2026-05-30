#!/usr/bin/env python3
"""
Post-processing script to fix outputs_improved/ results WITHOUT retraining.

Fixes:
  1. Trim overlapping experiment data in nnUnet_original_Dataset_Both_RealWorld
  2. Re-evaluate ALL experiments with complete metrics (12 metrics)
  3. Run test evaluation for UNet_BCE (4 experiments missing test_metrics)
  4. Generate sample PNGs for experiments missing them
  5. Regenerate plots from (corrected) metrics.csv
  6. Update run_summary.json for all experiments
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs_improved"
MODELS_DIR = ROOT / "Models"
DATASET_PREPROCESSED = ROOT / "Dataset_Preprocessed"

# All 12 experiment names
EXPERIMENTS = [
    "nnUnet_original_Dataset_Both_RealWorld",
    "nnUnet_original_Dataset_Both",
    "nnUnet_original_Dataset_small_tumor",
    "nnUnet_original_Dataset_large_tumor",
    "Attention_UNet_Dataset_Both_RealWorld",
    "Attention_UNet_Dataset_Both",
    "Attention_UNet_Dataset_small_tumor",
    "Attention_UNet_Dataset_large_tumor",
    "UNet_BCE_Dataset_Both_RealWorld",
    "UNet_BCE_Dataset_Both",
    "UNet_BCE_Dataset_small_tumor",
    "UNet_BCE_Dataset_large_tumor",
]

# Model name -> model directory
MODEL_DIRS = {
    "nnUnet_original": MODELS_DIR / "nnUnet_original",
    "Attention_UNet": MODELS_DIR / "Attention U_Net",
    "UNet_BCE": MODELS_DIR / "U_Net BCE",
}

DATASETS = [
    "Dataset_Both_RealWorld",
    "Dataset_Both",
    "Dataset_small_tumor",
    "Dataset_large_tumor",
]


def _get_model_name(exp_name: str) -> str:
    for m in ["nnUnet_original", "Attention_UNet", "UNet_BCE"]:
        if exp_name.startswith(m):
            return m
    raise ValueError(f"Unknown model in experiment: {exp_name}")


def _get_dataset_name(exp_name: str) -> str:
    model = _get_model_name(exp_name)
    return exp_name[len(model) + 1:]


# ---------------------------------------------------------------------------
# Step 0: setup sys.path for each model
# ---------------------------------------------------------------------------
def _setup_model_path(model_name: str):
    """Add model src/ and shared/ to sys.path."""
    model_dir = MODEL_DIRS[model_name]
    src_dir = str(model_dir / "src")
    shared_dir = str(MODELS_DIR / "shared")
    for p in [src_dir, shared_dir, str(model_dir)]:
        if p not in sys.path:
            sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Step 1: Trim overlapping data
# ---------------------------------------------------------------------------
def trim_overlapping_csv(exp_dir: Path, expected_epochs: int = 500):
    """Keep only the LAST `expected_epochs` rows from metrics.csv and metrics.jsonl."""
    csv_path = exp_dir / "logs" / "metrics.csv"
    jsonl_path = exp_dir / "logs" / "metrics.jsonl"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if len(df) > expected_epochs:
            print(f"  [TRIM] {csv_path.name}: {len(df)} rows -> keeping last {expected_epochs}")
            df_trimmed = df.tail(expected_epochs).copy()
            df_trimmed["epoch"] = range(1, expected_epochs + 1)
            # Backup original
            backup = csv_path.with_suffix(".csv.bak")
            if not backup.exists():
                shutil.copy2(csv_path, backup)
            df_trimmed.to_csv(csv_path, index=False)
        else:
            print(f"  [OK] {csv_path.name}: {len(df)} rows (no trimming needed)")

    if jsonl_path.exists():
        lines = jsonl_path.read_text().strip().split("\n")
        if len(lines) > expected_epochs:
            print(f"  [TRIM] {jsonl_path.name}: {len(lines)} lines -> keeping last {expected_epochs}")
            trimmed = lines[-expected_epochs:]
            # Re-number epochs
            new_lines = []
            for i, line in enumerate(trimmed, 1):
                data = json.loads(line)
                data["epoch"] = i
                new_lines.append(json.dumps(data))
            backup = jsonl_path.with_suffix(".jsonl.bak")
            if not backup.exists():
                shutil.copy2(jsonl_path, backup)
            jsonl_path.write_text("\n".join(new_lines) + "\n")


# ---------------------------------------------------------------------------
# Step 2 & 3: Re-evaluate with full metrics
# ---------------------------------------------------------------------------
def _build_records_from_config(run_config: dict, split: str) -> list:
    """Build StudyRecord-like objects from run_config.json dataset section."""
    from io_utils import StudyRecord, discover_studies, get_logger

    dataset_root = run_config["hyperparameters"]["dataset_root"]
    split_data = run_config.get("dataset", {}).get(split)
    if not split_data or not split_data.get("study_ids"):
        return []

    # Discover all studies to build the mapping
    logger = get_logger("fix_outputs")
    all_records = discover_studies(dataset_root, logger=logger)
    by_id = {r.study_id: r for r in all_records}

    records = []
    for sid in split_data["study_ids"]:
        if sid in by_id:
            records.append(by_id[sid])
        else:
            print(f"    [WARN] study_id '{sid}' not found in dataset_root")
    return records


def _build_loader(records, batch_size: int = 1, num_workers: int = 0):
    """Build a DataLoader from records."""
    from dataset_dbt import DBTVolumeDataset, collate_pad
    from io_utils import get_logger

    ds = DBTVolumeDataset(records, logger=get_logger("loader"))
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_pad,
    )


def _create_model(model_name: str):
    """Create a model instance by name."""
    if model_name == "nnUnet_original":
        from nnunet import nnUNet3D
        return nnUNet3D(in_ch=1, base_ch=32, levels=4, deep_supervision=False, use_checkpointing=True)
    elif model_name == "Attention_UNet":
        from attention_unet3d import AttentionUNet3D
        return AttentionUNet3D(in_ch=1, base_ch=32, levels=4, dropout_rate=0.1, use_checkpointing=True)
    elif model_name == "UNet_BCE":
        from unet3d import UNet3D
        return UNet3D(in_ch=1, base_ch=32, levels=4)
    else:
        raise ValueError(f"Unknown model: {model_name}")


@torch.no_grad()
def evaluate_full_metrics(model, loader, device, threshold: float = 0.5,
                          preds_out_dir: Optional[str] = None) -> Dict[str, float]:
    """Evaluate with ALL metrics from compute_metrics (12 metrics)."""
    from metrics import compute_metrics

    model.eval()
    agg: Dict[str, float] = {}
    n = 0

    for batch in loader:
        imgs = batch["image"].to(device)
        masks = batch["mask"].to(device)
        probs = model(imgs)

        m = compute_metrics(probs, masks, threshold=threshold)
        if not agg:
            agg = {k: 0.0 for k in m}
        for k in agg:
            agg[k] += m.get(k, 0.0) * imgs.size(0)
        n += imgs.size(0)

        # Save predictions if output dir specified
        if preds_out_dir:
            study_ids = batch["study_ids"]
            shapes = batch["shapes"]
            probs_cpu = probs.detach().cpu()
            for i, sid in enumerate(study_ids):
                Z, H, W = shapes[i]
                p = probs_cpu[i, 0, :Z, :H, :W].numpy()
                binp = (p >= threshold).astype("float32")
                os.makedirs(preds_out_dir, exist_ok=True)
                np.save(os.path.join(preds_out_dir, f"{sid}_probs.npy"), p)
                np.save(os.path.join(preds_out_dir, f"{sid}_bin.npy"), binp)

    if n == 0:
        return {k: 0.0 for k in agg} if agg else {}
    return {k: v / n for k, v in agg.items()}


def run_evaluation(exp_name: str, exp_dir: Path) -> Tuple[Optional[dict], Optional[dict]]:
    """Re-evaluate val and test sets with full metrics for one experiment."""
    model_name = _get_model_name(exp_name)
    _setup_model_path(model_name)

    # Read run_config
    config_path = exp_dir / "run_config.json"
    if not config_path.exists():
        print(f"  [SKIP] No run_config.json found")
        return None, None
    with open(config_path) as f:
        run_config = json.load(f)

    # Find checkpoint
    best_ckpt = exp_dir / "checkpoints" / "best.pt"
    last_ckpt = exp_dir / "checkpoints" / "last.pt"
    if best_ckpt.exists():
        ckpt_path = str(best_ckpt)
    elif last_ckpt.exists():
        ckpt_path = str(last_ckpt)
        print(f"  [WARN] Using last.pt (best.pt not found)")
    else:
        print(f"  [SKIP] No checkpoint found")
        return None, None

    threshold = float(run_config["hyperparameters"].get("threshold", 0.5))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = _create_model(model_name)
    data = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(data["model_state"])
    model.to(device)
    model.eval()

    # --- Validation ---
    val_records = _build_records_from_config(run_config, "val")
    val_metrics = None
    if val_records:
        print(f"  [VAL] Evaluating {len(val_records)} val studies...")
        val_loader = _build_loader(val_records)
        val_metrics = evaluate_full_metrics(
            model, val_loader, device, threshold,
            preds_out_dir=str(exp_dir / "preds_npy"),
        )
        # Save
        with open(exp_dir / "logs" / "val_metrics.json", "w") as f:
            json.dump(val_metrics, f, indent=2)
        print(f"    dice={val_metrics['dice']:.4f} | iou={val_metrics['iou']:.4f} | "
              f"hd95={val_metrics.get('hausdorff_95', 'N/A')}")
    else:
        print(f"  [SKIP] No val records found")

    # --- Test ---
    test_records = _build_records_from_config(run_config, "test")
    test_metrics = None

    # For UNet_BCE experiments without test split, borrow test IDs from nnUnet
    if not test_records and model_name == "UNet_BCE":
        dataset_name = _get_dataset_name(exp_name)
        ref_exp = f"nnUnet_original_{dataset_name}"
        ref_config_path = OUTPUTS / ref_exp / "run_config.json"
        if ref_config_path.exists():
            with open(ref_config_path) as f:
                ref_config = json.load(f)
            # Use nnUnet's test study_ids but with UNet_BCE's dataset_root
            ref_test = ref_config.get("dataset", {}).get("test", {})
            if ref_test and ref_test.get("study_ids"):
                # Build records from UNet_BCE's dataset
                _setup_model_path(model_name)
                from io_utils import discover_studies, get_logger
                dataset_root = run_config["hyperparameters"]["dataset_root"]
                all_records = discover_studies(dataset_root, logger=get_logger("fix"))
                by_id = {r.study_id: r for r in all_records}
                test_records = [by_id[sid] for sid in ref_test["study_ids"] if sid in by_id]
                if test_records:
                    print(f"  [TEST] Using {len(test_records)} test IDs from {ref_exp}")

    if test_records:
        print(f"  [TEST] Evaluating {len(test_records)} test studies...")
        test_preds_dir = exp_dir / "test_preds_npy"
        test_preds_dir.mkdir(parents=True, exist_ok=True)
        test_loader = _build_loader(test_records)
        test_metrics = evaluate_full_metrics(
            model, test_loader, device, threshold,
            preds_out_dir=str(test_preds_dir),
        )
        # Save
        with open(exp_dir / "logs" / "test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)
        print(f"    dice={test_metrics['dice']:.4f} | iou={test_metrics['iou']:.4f} | "
              f"hd95={test_metrics.get('hausdorff_95', 'N/A')}")
    else:
        print(f"  [INFO] No test records (skip test evaluation)")

    # Cleanup GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return val_metrics, test_metrics


# ---------------------------------------------------------------------------
# Step 4: Generate sample PNGs
# ---------------------------------------------------------------------------
def generate_samples(exp_name: str, exp_dir: Path, n_samples: int = 5):
    """Generate sample PNGs for visual inspection."""
    model_name = _get_model_name(exp_name)
    _setup_model_path(model_name)

    samples_dir = exp_dir / "samples"
    if samples_dir.exists() and any(samples_dir.glob("*.png")):
        print(f"  [OK] Samples already exist ({len(list(samples_dir.glob('*.png')))} PNGs)")
        return

    config_path = exp_dir / "run_config.json"
    if not config_path.exists():
        print(f"  [SKIP] No run_config.json")
        return

    with open(config_path) as f:
        run_config = json.load(f)

    best_ckpt = exp_dir / "checkpoints" / "best.pt"
    last_ckpt = exp_dir / "checkpoints" / "last.pt"
    ckpt_path = str(best_ckpt) if best_ckpt.exists() else str(last_ckpt) if last_ckpt.exists() else None
    if not ckpt_path:
        print(f"  [SKIP] No checkpoint for samples")
        return

    threshold = float(run_config["hyperparameters"].get("threshold", 0.5))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _create_model(model_name)
    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(data["model_state"])
    model.to(device)
    model.eval()

    val_records = _build_records_from_config(run_config, "val")
    if not val_records:
        print(f"  [SKIP] No val records for samples")
        del model
        return

    val_loader = _build_loader(val_records)

    try:
        from visualization import generate_validation_samples
        paths = generate_validation_samples(
            model, val_loader, device, exp_dir,
            n_samples=n_samples, threshold=threshold,
        )
        print(f"  [SAMPLES] Generated {len(paths)} sample PNGs")
    except Exception as e:
        print(f"  [ERROR] Failed to generate samples: {e}")
        import traceback
        traceback.print_exc()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Step 5: Regenerate plots
# ---------------------------------------------------------------------------
def _plot_with_band(ax, epochs, series, label, color):
    rolling = series.rolling(window=5, min_periods=1)
    mean = rolling.mean()
    std = rolling.std(ddof=0).fillna(0.0)
    ax.plot(epochs, series, color=color, lw=2, label=label)
    ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.18, label="+-1 std")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(label)
    ax.grid(True, ls="--", alpha=0.55)
    ax.legend(fontsize=9)


def regenerate_plots(exp_name: str, exp_dir: Path):
    """Regenerate all plots from metrics.csv."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csv_path = exp_dir / "logs" / "metrics.csv"
    if not csv_path.exists():
        print(f"  [SKIP] No metrics.csv for plots")
        return

    df = pd.read_csv(csv_path)
    out_dir = exp_dir / "logs"
    epochs = df["epoch"]

    model_name = _get_model_name(exp_name)
    model_label = {"nnUnet_original": "nnUNet", "Attention_UNet": "Attention U-Net", "UNet_BCE": "U-Net BCE"}[model_name]

    # All metrics to plot
    all_metrics = [
        ("train_loss", "Train Loss", "#1f77b4"),
        ("val_dice", "Val Dice", "#d62728"),
        ("val_iou", "Val IoU", "#9467bd"),
        ("val_precision", "Val Precision", "#2ca02c"),
        ("val_recall", "Val Recall", "#ff7f0e"),
        ("val_f1", "Val F1", "#17becf"),
        ("val_accuracy", "Val Accuracy", "#7f7f7f"),
        ("val_mAP", "Val mAP", "#e377c2"),
        ("val_hausdorff", "Val Hausdorff", "#8c564b"),
        ("val_hausdorff_95", "Val HD95", "#bcbd22"),
        ("val_fp_per_volume", "Val FP/Volume", "#ff9896"),
        ("val_fp_per_image", "Val FP/Image", "#c5b0d5"),
        ("val_ap", "Val AP", "#98df8a"),
    ]

    # === 1. Individual plots ===
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

    # === 2. Paper curves (2x4) ===
    metrics_main = [
        ("train_loss", "Train Loss", "#1f77b4"),
        ("val_dice", "Val Dice", "#d62728"),
        ("val_iou", "Val IoU", "#9467bd"),
        ("val_precision", "Val Precision", "#2ca02c"),
        ("val_recall", "Val Recall", "#ff7f0e"),
        ("val_f1", "Val F1", "#17becf"),
        ("val_accuracy", "Val Accuracy", "#7f7f7f"),
        ("val_mAP", "Val mAP", "#e377c2"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=True)
    axes_flat = axes.ravel()
    for (col, label, color), ax in zip(metrics_main, axes_flat):
        if col not in df.columns:
            ax.axis("off")
            continue
        _plot_with_band(ax, epochs, df[col], label, color)
    fig.suptitle(f"Training/Validation Curves ({model_label})", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "paper_curves.png", dpi=200)
    plt.close(fig)

    # === 3. Extra panel (Hausdorff, FP, AP) ===
    metrics_extra = [
        ("val_hausdorff", "Val Hausdorff", "#8c564b"),
        ("val_hausdorff_95", "Val HD95", "#bcbd22"),
        ("val_fp_per_volume", "Val FP/Volume", "#ff9896"),
        ("val_fp_per_image", "Val FP/Image", "#c5b0d5"),
        ("val_ap", "Val AP", "#98df8a"),
    ]

    fig2, axes2 = plt.subplots(1, 5, figsize=(17, 3.5))
    for (col, label, color), ax in zip(metrics_extra, axes2):
        if col not in df.columns:
            ax.axis("off")
            continue
        _plot_with_band(ax, epochs, df[col], label, color)
    fig2.suptitle("Additional Metrics (Hausdorff, FP & AP)", fontsize=14, fontweight="bold")
    fig2.tight_layout(rect=[0, 0, 1, 0.92])
    fig2.savefig(out_dir / "paper_curves_extra.png", dpi=200)
    plt.close(fig2)

    # === 4. LR curve ===
    if "lr" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, df["lr"], color="#1f77b4", lw=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning rate")
        ax.grid(True, ls="--", alpha=0.55)
        fig.tight_layout()
        fig.savefig(out_dir / "lr_curve.png", dpi=200)
        plt.close(fig)

    n_plots = sum(1 for col, _, _ in all_metrics if col in df.columns)
    print(f"  [PLOTS] Regenerated {n_plots} individual + 2 panels + LR")


# ---------------------------------------------------------------------------
# Step 6: Update run_summary.json
# ---------------------------------------------------------------------------
def update_run_summary(exp_dir: Path, val_metrics: Optional[dict], test_metrics: Optional[dict]):
    """Update run_summary.json with new metrics."""
    summary_path = exp_dir / "run_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        summary = {}

    if val_metrics is not None:
        summary["val_metrics"] = val_metrics
    if test_metrics is not None:
        summary["test_metrics"] = test_metrics

    # Check for sample PNGs
    samples_dir = exp_dir / "samples"
    if samples_dir.exists():
        pngs = sorted([str(p) for p in samples_dir.glob("*.png")])
        summary["sample_pngs"] = pngs

    summary["fixed_timestamp"] = datetime.now().isoformat()

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [SUMMARY] Updated run_summary.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fix outputs_improved results")
    parser.add_argument("--experiments", nargs="*", default=None,
                        help="Specific experiments to fix (default: all)")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip re-evaluation (only trim, plots, summary)")
    parser.add_argument("--skip-samples", action="store_true",
                        help="Skip sample PNG generation")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Skip plot regeneration")
    parser.add_argument("--n-samples", type=int, default=5,
                        help="Number of sample PNGs to generate")
    args = parser.parse_args()

    experiments = args.experiments or EXPERIMENTS

    print("=" * 70)
    print("FIX OUTPUTS - Post-processing Script")
    print("=" * 70)
    print(f"Experiments to fix: {len(experiments)}")
    print(f"Output root: {OUTPUTS}")
    print()

    # ===== STEP 1: Trim overlapping data =====
    print("=" * 70)
    print("STEP 1: Trim overlapping experiment data")
    print("=" * 70)
    overlap_exp = "nnUnet_original_Dataset_Both_RealWorld"
    if overlap_exp in experiments:
        exp_dir = OUTPUTS / overlap_exp
        if exp_dir.exists():
            trim_overlapping_csv(exp_dir, expected_epochs=500)
    print()

    # ===== STEP 2 & 3: Re-evaluate with full metrics =====
    if not args.skip_eval:
        print("=" * 70)
        print("STEP 2 & 3: Re-evaluate with complete metrics (12 metrics)")
        print("=" * 70)
        for exp_name in experiments:
            exp_dir = OUTPUTS / exp_name
            if not exp_dir.exists():
                print(f"\n[SKIP] {exp_name} (directory not found)")
                continue

            print(f"\n--- {exp_name} ---")

            # Clear module caches to avoid cross-model import conflicts
            for mod_name in list(sys.modules.keys()):
                if mod_name in ("metrics", "train_eval", "dataset_dbt", "io_utils",
                                "config_loader", "unet3d", "nnunet", "attention_unet3d",
                                "nnunet_train", "viz", "augmentations"):
                    del sys.modules[mod_name]

            # Remove old model src paths from sys.path
            sys.path = [p for p in sys.path if "Models" not in p or "shared" in p]
            # Keep shared in path
            shared_dir = str(MODELS_DIR / "shared")
            if shared_dir not in sys.path:
                sys.path.insert(0, shared_dir)

            try:
                val_metrics, test_metrics = run_evaluation(exp_name, exp_dir)
                update_run_summary(exp_dir, val_metrics, test_metrics)
            except Exception as e:
                print(f"  [ERROR] {e}")
                import traceback
                traceback.print_exc()
        print()

    # ===== STEP 4: Generate sample PNGs =====
    if not args.skip_samples:
        print("=" * 70)
        print("STEP 4: Generate sample PNGs")
        print("=" * 70)
        for exp_name in experiments:
            exp_dir = OUTPUTS / exp_name
            if not exp_dir.exists():
                continue
            print(f"\n--- {exp_name} ---")

            # Clear module caches
            for mod_name in list(sys.modules.keys()):
                if mod_name in ("metrics", "train_eval", "dataset_dbt", "io_utils",
                                "config_loader", "unet3d", "nnunet", "attention_unet3d",
                                "nnunet_train", "viz", "augmentations"):
                    del sys.modules[mod_name]

            sys.path = [p for p in sys.path if "Models" not in p or "shared" in p]
            shared_dir = str(MODELS_DIR / "shared")
            if shared_dir not in sys.path:
                sys.path.insert(0, shared_dir)

            try:
                generate_samples(exp_name, exp_dir, n_samples=args.n_samples)
            except Exception as e:
                print(f"  [ERROR] {e}")
                import traceback
                traceback.print_exc()
        print()

    # ===== STEP 5: Regenerate plots =====
    if not args.skip_plots:
        print("=" * 70)
        print("STEP 5: Regenerate plots")
        print("=" * 70)
        for exp_name in experiments:
            exp_dir = OUTPUTS / exp_name
            if not exp_dir.exists():
                continue
            print(f"\n--- {exp_name} ---")
            try:
                regenerate_plots(exp_name, exp_dir)
            except Exception as e:
                print(f"  [ERROR] {e}")
                import traceback
                traceback.print_exc()
        print()

    # ===== Final: update run_summary with sample_pngs =====
    print("=" * 70)
    print("STEP 6: Final run_summary.json update (sample PNGs)")
    print("=" * 70)
    for exp_name in experiments:
        exp_dir = OUTPUTS / exp_name
        if not exp_dir.exists():
            continue
        update_run_summary(exp_dir, val_metrics=None, test_metrics=None)
    print()

    # ===== Done =====
    print("=" * 70)
    print("ALL DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()

