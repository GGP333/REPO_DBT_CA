"""
Figura pareada de la cohorte clinica de test (sustituye a la Figura 3).

Con n = 10 y cuatro metodos evaluados sobre los MISMOS volumenes, un grafico de
barras media +/- DE oculta lo esencial: que las comparaciones son pareadas y que
la ventaja del ensemble no es uniforme caso a caso. Aqui cada paciente es una
linea que recorre los cuatro metodos, sobre una caja que resume la distribucion.

Uso:  python ensemble/make_statistics_figures.py
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Paleta categorica validada para deficiencia de vision cromatica
COLORS = {"nnU-Net": "#2a78d6", "Attention": "#eb6834",
          "U-Net BCE": "#1baf7a", "Ensemble": "#4a3aa7"}
METHODS = ["nnU-Net", "Attention", "U-Net BCE", "Ensemble"]
METRICS = ["Dice", "IoU", "Precision", "Recall"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
LINE = "#c9c8c3"


def style():
    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.labelcolor": INK, "text.color": INK,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def panel(ax, wide, metric, wilc, rng):
    x = np.arange(len(METHODS))

    # Lineas pareadas: un paciente = una linea
    for _, row in wide.iterrows():
        ax.plot(x, [row[m] for m in METHODS], color=LINE, lw=0.8, zorder=1)

    bp = ax.boxplot([wide[m].values for m in METHODS], positions=x,
                    widths=0.52, showfliers=False, patch_artist=True,
                    medianprops=dict(color=INK, lw=1.6),
                    whiskerprops=dict(color=INK_2, lw=1.0),
                    capprops=dict(color=INK_2, lw=1.0),
                    boxprops=dict(lw=1.0))
    for patch, m in zip(bp["boxes"], METHODS):
        patch.set_facecolor(COLORS[m])
        patch.set_alpha(0.20)
        patch.set_edgecolor(COLORS[m])

    # Puntos por caso, con dispersion horizontal reproducible
    for i, m in enumerate(METHODS):
        v = wide[m].values
        jitter = rng.uniform(-0.11, 0.11, size=len(v))
        ax.plot(i + jitter, v, "o", ms=4.5, color=COLORS[m],
                mec="white", mew=0.7, zorder=4)
        ax.plot([i - 0.26, i + 0.26], [v.mean()] * 2, color=COLORS[m],
                lw=2.0, ls=(0, (2.5, 1.5)), zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, rotation=18, ha="right")
    ax.set_ylabel(metric)
    ax.set_ylim(-0.04, 1.12)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # p de Holm frente al ensemble, sobre cada metodo comparado
    sub = wilc[wilc.metric == metric].set_index("comparison")
    for i, m in enumerate(METHODS[:-1]):
        r = sub.loc[f"Ensemble vs {m}"]
        p = r["p_holm"]
        txt = ("$p$ < 0.001" if p < 0.001 else f"$p$ = {p:.3f}")
        star = " *" if p < 0.05 else ""
        ax.text(i, 1.05, txt + star, ha="center", fontsize=7.2,
                color=INK if p < 0.05 else INK_2,
                fontweight="bold" if p < 0.05 else "normal")
    ax.text(len(METHODS) - 1, 1.05, "reference", ha="center", fontsize=7.2,
            color=INK_2, style="italic")
    ax.set_title(metric + (" (primary endpoint)" if metric == "Dice" else ""),
                 loc="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores",
                    default="results/paper_assets/scores/scores_per_case_test.csv")
    ap.add_argument("--stats", default="results/statistics")
    ap.add_argument("--out", default="results/statistics")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    style()

    df = pd.read_csv(args.scores)
    wilc = pd.read_csv(f"{args.stats}/wilcoxon_pairwise.csv")

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.3))
    for ax, metric in zip(axes, METRICS):
        wide = df.pivot(index="case", columns="method",
                        values=metric).sort_index()
        panel(ax, wide, metric, wilc, np.random.default_rng(args.seed))

    fig.suptitle(
        "Independent clinical test cohort (n = 10). Each grey line is one "
        "patient across all four methods; boxes show median and IQR, dashed "
        "rules the mean. $p$-values are two-sided Wilcoxon signed-rank versus "
        "the ensemble, Holm-adjusted within each metric.",
        y=1.005, fontsize=8.5, color=INK_2)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    os.makedirs(args.out, exist_ok=True)
    for ext in ("png", "pdf"):
        p = f"{args.out}/fig_paired_test_metrics.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"-> {p}")


if __name__ == "__main__":
    main()
