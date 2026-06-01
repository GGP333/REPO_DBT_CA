#!/usr/bin/env python3
"""
Regenerate the per-dataset training-loss figure for the manuscript's three models.

Curve sources (the runs reported in the paper):
  - nnU-Net          -> official nnU-Netv2 training_log_*.txt (train_loss per epoch)
  - 3D U-Net (base.) -> U-Net BCE  logs/metrics.csv (train_loss)
  - Attention U-Net  -> Attention U-Net logs/metrics.csv (train_loss)

Style matches src/fix_outputs.py::_plot_with_band (rolling window=5 mean +-1 std,
colour #1f77b4). Note: nnU-Netv2's loss is its compound DC+CE loss (can be negative)
and runs for 1000 epochs, whereas the in-house models log a positive loss over 500.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
COLOR = "#1f77b4"

# dataset key -> (nnU-Net results dir, pretty dataset name)
NNUNET_DS = {
    "large_tumor": "Dataset001_DBTLarge",
    "small_tumor": "Dataset002_DBTSmall",
    "Mixed_Size": "Dataset003_DBTMixedSize",
    "Hybrid": "Dataset004_DBTHybrid",
}
PRETTY = {
    "large_tumor": "Large tumor",
    "small_tumor": "Small tumor",
    "Mixed_Size": "Mixed-size tumor",
    "Hybrid": "Hybrid synthetic-clinical",
}


def unet_bce_dir(ds: str) -> Path:
    # Hybrid was already leakage-free (outputs_improved); the other three
    # use the leakage-free re-trains in outputs_clean.
    base = "outputs_improved" if ds == "Hybrid" else "outputs_clean"
    return REPO / "results" / base / f"UNet_BCE_Dataset_{ds}"


def attention_dir(ds: str) -> Path:
    return REPO / "results" / "outputs_improved" / f"Attention_UNet_Dataset_{ds}"


def loss_from_metrics_csv(exp_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(exp_dir / "logs" / "metrics.csv")
    return pd.DataFrame({"epoch": df["epoch"], "train_loss": df["train_loss"]})


def loss_from_nnunet_log(ds: str) -> pd.DataFrame:
    fold = (REPO / "experiments" / "nnunet_official" / "nnunet_root" / "results"
            / NNUNET_DS[ds] / "nnUNetTrainer_Seeded42__nnUNetPlans__3d_fullres" / "fold_0")
    logs = sorted(fold.glob("training_log_*.txt"))
    if not logs:
        raise FileNotFoundError(f"no training_log_*.txt under {fold}")
    # Pair each train_loss with the epoch number from the preceding "Epoch N" line,
    # so resumed runs (whose log starts past epoch 0) keep the true x-axis.
    epochs, losses = [], []
    cur_epoch = None
    ep_pat = re.compile(r"Epoch\s+(\d+)\s*$")
    loss_pat = re.compile(r"train_loss\s+(-?\d+\.?\d*)")
    for line in logs[-1].read_text().splitlines():
        em = ep_pat.search(line)
        if em:
            cur_epoch = int(em.group(1))
            continue
        lm = loss_pat.search(line)
        if lm and cur_epoch is not None:
            epochs.append(cur_epoch)
            losses.append(float(lm.group(1)))
    return pd.DataFrame({"epoch": epochs, "train_loss": losses})


def plot_band(ax, df: pd.DataFrame, title: str):
    epochs, series = df["epoch"], df["train_loss"]
    roll = series.rolling(window=5, min_periods=1)
    mean = roll.mean()
    std = roll.std(ddof=0).fillna(0.0)
    ax.plot(epochs, series, color=COLOR, lw=2, label="Train Loss")
    ax.fill_between(epochs, mean - std, mean + std, color=COLOR, alpha=0.18, label="±1 std")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Loss")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.55)
    ax.legend(fontsize=9)


def make_figure(ds: str, out_path: Path):
    panels = [
        ("(A) 3D U-Net", loss_from_metrics_csv(unet_bce_dir(ds))),
        ("(B) nnU-Net", loss_from_nnunet_log(ds)),
        ("(C) Attention U-Net", loss_from_metrics_csv(attention_dir(ds))),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (title, df) in zip(axes, panels):
        plot_band(ax, df, title)
    fig.suptitle(f"Training curves dynamics — {PRETTY[ds]} dataset",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[OK] {out_path}")
    for title, df in panels:
        print(f"     {title:24s} epochs={len(df):4d}  "
              f"loss {df['train_loss'].iloc[0]:+.3f} -> {df['train_loss'].iloc[-1]:+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["Hybrid"],
                    choices=list(NNUNET_DS))
    ap.add_argument("--out_dir", default=str(REPO / "results" / "figures"))
    args = ap.parse_args()
    for ds in args.datasets:
        make_figure(ds, Path(args.out_dir) / f"training_curves_{ds}.png")


if __name__ == "__main__":
    main()
