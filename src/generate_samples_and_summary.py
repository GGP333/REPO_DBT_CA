#!/usr/bin/env python3
"""
Script to:
  1. Regenerate 20 sample PNGs per experiment (mixed val + test, clearly labeled)
  2. Update training_summary.json with complete metrics
  3. (README is generated separately)

Uses existing prediction .npy files -- NO model inference needed.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs_improved"

# Add shared visualization
_shared = str(ROOT / "Models" / "shared")
if _shared not in sys.path:
    sys.path.insert(0, _shared)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Robust min-max normalisation (same as in dataset_dbt.py)
# ---------------------------------------------------------------------------
def robust_minmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    flat = x.ravel()
    nonzero = flat[flat != 0]
    ref = nonzero if nonzero.size > 0 else flat
    lo = float(np.percentile(ref, 1.0))
    hi = float(np.percentile(ref, 99.0))
    if hi <= lo:
        hi = float(ref.max()) if ref.size else 1.0
        lo = float(ref.min()) if ref.size else 0.0
    x = (x - lo) / (hi - lo + 1e-6)
    return np.clip(x, 0.0, 1.0)


# ---------------------------------------------------------------------------
# PNG generation (enhanced version with split label)
# ---------------------------------------------------------------------------
def _pick_representative_slices(mask_3d: np.ndarray, n: int = 3) -> List[int]:
    fg_slices = [z for z in range(mask_3d.shape[0]) if mask_3d[z].any()]
    if not fg_slices:
        total = mask_3d.shape[0]
        return [0, total // 2, total - 1][:n]
    first_fg = fg_slices[0]
    last_fg = fg_slices[-1]
    mid_fg = fg_slices[len(fg_slices) // 2]
    return sorted(set([first_fg, mid_fg, last_fg]))[:n]


def save_labeled_sample_png(
    img: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    study_id: str,
    split_label: str,  # "VAL" or "TEST"
    out_dir: Path,
) -> Path:
    """Save a PNG grid [Input | GT | Prediction | Overlay] with split label."""
    out_dir.mkdir(parents=True, exist_ok=True)

    slices = _pick_representative_slices(gt_mask, n=3)
    n_slices = len(slices)

    fig, axes = plt.subplots(n_slices, 4, figsize=(16, 4 * n_slices))
    if n_slices == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Input", "Ground Truth", "Prediction", "Overlay"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold")

    for row, z in enumerate(slices):
        axes[row, 0].imshow(img[z], cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_ylabel(f"Z={z}", fontsize=10)
        axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])

        axes[row, 1].imshow(gt_mask[z], cmap="Reds", vmin=0, vmax=1)
        axes[row, 1].set_xticks([]); axes[row, 1].set_yticks([])

        axes[row, 2].imshow(pred_mask[z], cmap="Blues", vmin=0, vmax=1)
        axes[row, 2].set_xticks([]); axes[row, 2].set_yticks([])

        overlay = np.stack([img[z]] * 3, axis=-1)
        overlay = np.clip(overlay, 0, 1)
        overlay[..., 0] = np.clip(overlay[..., 0] + gt_mask[z] * 0.5, 0, 1)
        overlay[..., 2] = np.clip(overlay[..., 2] + pred_mask[z] * 0.5, 0, 1)
        axes[row, 3].imshow(overlay)
        axes[row, 3].set_xticks([]); axes[row, 3].set_yticks([])

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="red", alpha=0.5, label="GT"),
        Patch(facecolor="blue", alpha=0.5, label="Pred"),
    ]
    axes[-1, 3].legend(handles=legend_elements, loc="lower right", fontsize=8)

    # Title with split label colour
    colour = "#2e7d32" if split_label == "TEST" else "#1565c0"
    fig.suptitle(
        f"[{split_label}] {study_id}",
        fontsize=15, fontweight="bold", color=colour,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    prefix = "test" if split_label == "TEST" else "val"
    out_path = out_dir / f"{prefix}_sample_{study_id}.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Discover study paths inside a dataset_root
# ---------------------------------------------------------------------------
_ID_RE = re.compile(r"dbt[_\-]?(\d+)", re.IGNORECASE)

def _find_study_paths(dataset_root: str, study_id: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (img_path, mask_path) for a study inside dataset_root."""
    sdir = os.path.join(dataset_root, study_id)
    if not os.path.isdir(sdir):
        return None, None
    # Scan all .npy
    npys = []
    for r, _, files in os.walk(sdir):
        for f in files:
            if f.endswith(".npy"):
                npys.append(os.path.join(r, f))
    img_path = None
    mask_path = None
    for p in npys:
        bl = os.path.basename(p).lower()
        if "mask" in bl:
            mask_path = p
        elif "img" in bl:
            img_path = p
    # fallback: if only 2 files, guess by name length
    if not img_path and not mask_path and len(npys) == 2:
        npys.sort(key=lambda x: len(x))
        img_path, mask_path = npys
    return img_path, mask_path


# ---------------------------------------------------------------------------
# TASK 1: Generate extended samples
# ---------------------------------------------------------------------------
def generate_extended_samples(target_total: int = 20):
    """Generate val + test sample PNGs for all 12 experiments."""
    experiments = sorted([
        d for d in os.listdir(OUTPUTS)
        if os.path.isdir(OUTPUTS / d) and not d.endswith(".json")
    ])

    for exp_name in experiments:
        exp_dir = OUTPUTS / exp_name
        config_path = exp_dir / "run_config.json"
        if not config_path.exists():
            print(f"[SKIP] {exp_name}: no run_config.json")
            continue

        with open(config_path) as f:
            run_config = json.load(f)
        dataset_root = run_config["hyperparameters"]["dataset_root"]
        dataset_info = run_config.get("dataset", {})

        # Gather study_ids for val and test
        val_ids = dataset_info.get("val", {}).get("study_ids", [])
        test_ids = dataset_info.get("test", {}).get("study_ids", [])

        # Directories with saved predictions
        val_preds_dir = exp_dir / "preds_npy"
        test_preds_dir = exp_dir / "test_preds_npy"

        # Filter to studies that actually have saved predictions
        val_available = [sid for sid in val_ids
                         if (val_preds_dir / f"{sid}_bin.npy").exists()]
        test_available = [sid for sid in test_ids
                          if (test_preds_dir / f"{sid}_bin.npy").exists()]

        # For UNet_BCE experiments that borrowed test IDs from nnUnet
        if not test_available and not test_ids:
            # Try to find test IDs from corresponding nnUnet experiment
            model_name = exp_name.split("_Dataset")[0]
            dataset_name = exp_name[len(model_name) + 1:]
            ref_exp = f"nnUnet_original_{dataset_name}"
            ref_config = OUTPUTS / ref_exp / "run_config.json"
            if ref_config.exists():
                with open(ref_config) as f:
                    rc = json.load(f)
                test_ids = rc.get("dataset", {}).get("test", {}).get("study_ids", [])
                test_available = [sid for sid in test_ids
                                  if (test_preds_dir / f"{sid}_bin.npy").exists()]

        total_available = len(val_available) + len(test_available)
        if total_available == 0:
            print(f"[SKIP] {exp_name}: no predictions found")
            continue

        # Distribute: aim for ~half from each, but adapt to availability
        n_test = min(len(test_available), target_total // 2)
        n_val = min(len(val_available), target_total - n_test)
        # If one set was small, give more to the other
        n_test = min(len(test_available), target_total - n_val)

        # Delete old samples directory
        samples_dir = exp_dir / "samples"
        if samples_dir.exists():
            shutil.rmtree(samples_dir)
        samples_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- {exp_name} ---")
        print(f"    val available: {len(val_available)}, test available: {len(test_available)}")
        print(f"    generating: {n_val} val + {n_test} test = {n_val + n_test}")

        paths = []

        # Generate VAL samples
        for sid in val_available[:n_val]:
            img_path, mask_path = _find_study_paths(dataset_root, sid)
            pred_bin_path = val_preds_dir / f"{sid}_bin.npy"
            if not img_path or not mask_path:
                print(f"    [WARN] {sid}: missing img/mask in dataset")
                continue

            img = robust_minmax(np.load(img_path))
            gt = (np.load(mask_path) > 0).astype(np.float32)
            pred = np.load(str(pred_bin_path))

            # Align shapes
            z = min(img.shape[0], gt.shape[0], pred.shape[0])
            h = min(img.shape[1], gt.shape[1], pred.shape[1])
            w = min(img.shape[2], gt.shape[2], pred.shape[2])
            img, gt, pred = img[:z, :h, :w], gt[:z, :h, :w], pred[:z, :h, :w]

            p = save_labeled_sample_png(img, gt, pred, sid, "VAL", samples_dir)
            paths.append(str(p))

        # Generate TEST samples
        for sid in test_available[:n_test]:
            img_path, mask_path = _find_study_paths(dataset_root, sid)
            pred_bin_path = test_preds_dir / f"{sid}_bin.npy"
            if not img_path or not mask_path:
                print(f"    [WARN] {sid}: missing img/mask in dataset")
                continue

            img = robust_minmax(np.load(img_path))
            gt = (np.load(mask_path) > 0).astype(np.float32)
            pred = np.load(str(pred_bin_path))

            z = min(img.shape[0], gt.shape[0], pred.shape[0])
            h = min(img.shape[1], gt.shape[1], pred.shape[1])
            w = min(img.shape[2], gt.shape[2], pred.shape[2])
            img, gt, pred = img[:z, :h, :w], gt[:z, :h, :w], pred[:z, :h, :w]

            p = save_labeled_sample_png(img, gt, pred, sid, "TEST", samples_dir)
            paths.append(str(p))

        # Update run_summary.json
        summary_path = exp_dir / "run_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
        else:
            summary = {}
        summary["sample_pngs"] = sorted(paths)
        summary["samples_timestamp"] = datetime.now().isoformat()
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"    OK: {len(paths)} PNGs saved to {samples_dir}")


# ---------------------------------------------------------------------------
# TASK 2: Update training_summary.json
# ---------------------------------------------------------------------------
def update_training_summary():
    """Rebuild training_summary.json from individual run_summary.json files."""
    summary_path = OUTPUTS / "training_summary.json"

    # Read existing to preserve timestamps / elapsed seconds
    existing = {}
    if summary_path.exists():
        with open(summary_path) as f:
            existing = json.load(f)

    # Build a lookup of existing results by (model, dataset)
    existing_lookup: Dict[Tuple[str, str], dict] = {}
    for r in existing.get("results", []):
        key = (r["model"], r["dataset"])
        existing_lookup[key] = r

    experiments = sorted([
        d for d in os.listdir(OUTPUTS)
        if os.path.isdir(OUTPUTS / d) and not d.endswith(".json")
    ])

    results = []
    for exp_name in experiments:
        exp_dir = OUTPUTS / exp_name

        # Parse model and dataset from experiment name
        for model_prefix in ["nnUnet_original", "Attention_UNet", "UNet_BCE"]:
            if exp_name.startswith(model_prefix):
                model_name = model_prefix
                dataset_name = exp_name[len(model_prefix) + 1:]
                break
        else:
            continue

        # Read run_summary
        rs_path = exp_dir / "run_summary.json"
        if not rs_path.exists():
            continue
        with open(rs_path) as f:
            rs = json.load(f)

        # Retrieve preserved fields from existing summary
        old = existing_lookup.get((model_name, dataset_name), {})

        entry = {
            "model": model_name,
            "dataset": dataset_name,
            "status": old.get("status", "\u2705 SUCCESS"),
            "elapsed_seconds": old.get("elapsed_seconds", rs.get("total_time_seconds")),
            "error": old.get("error", None),
            "output_dir": str(exp_dir),
            "total_epochs": rs.get("total_epochs", 500),
            "best_epoch": rs.get("best_epoch"),
            "last_epoch": rs.get("last_epoch"),
            "total_time_human": rs.get("total_time_human"),
            "val_metrics": rs.get("val_metrics"),
            "test_metrics": rs.get("test_metrics"),
        }
        results.append(entry)

    # Sort: nnUnet first, then Attention, then BCE; within each by dataset order
    ds_order = ["Dataset_Both_RealWorld", "Dataset_Both", "Dataset_small_tumor", "Dataset_large_tumor"]
    model_order = ["nnUnet_original", "Attention_UNet", "UNet_BCE"]

    def sort_key(r):
        mi = model_order.index(r["model"]) if r["model"] in model_order else 99
        di = ds_order.index(r["dataset"]) if r["dataset"] in ds_order else 99
        return (mi, di)

    results.sort(key=sort_key)

    total_elapsed = sum(r.get("elapsed_seconds", 0) or 0 for r in results)

    new_summary = {
        "timestamp": existing.get("timestamp", datetime.now().isoformat()),
        "updated_timestamp": datetime.now().isoformat(),
        "epochs": 500,
        "total_runs": len(results),
        "success_count": sum(1 for r in results if "SUCCESS" in (r.get("status") or "")),
        "failed_count": sum(1 for r in results if "SUCCESS" not in (r.get("status") or "")),
        "total_elapsed_seconds": total_elapsed,
        "total_elapsed_human": f"{total_elapsed / 3600:.1f}h",
        "results": results,
    }

    with open(summary_path, "w") as f:
        json.dump(new_summary, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] training_summary.json updated with {len(results)} experiments")
    print(f"     All have val_metrics: {all(r.get('val_metrics') for r in results)}")
    print(f"     All have test_metrics: {all(r.get('test_metrics') for r in results)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("TASK 1: Generate extended sample PNGs (val + test)")
    print("=" * 70)
    generate_extended_samples(target_total=20)

    print()
    print("=" * 70)
    print("TASK 2: Update training_summary.json")
    print("=" * 70)
    update_training_summary()

    print()
    print("=" * 70)
    print("ALL DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()

