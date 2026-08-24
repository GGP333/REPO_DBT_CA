"""
Figuras redibujadas para proyeccion (presentacion de la revision).

La figura del paper (results/statistics/fig_paired_test_metrics.pdf) pone los
cuatro endpoints en una fila: relacion de aspecto 3.5:1. A ancho de diapositiva
16:9 eso deja las etiquetas en ~3.4 pt, ilegibles en pantalla.

Aqui se redibuja SOLO el endpoint primario (Dice) con proporcion ~2:1 y cuerpos
de letra escalados para que, ya reducida a ancho de texto, la figura se lea a
~10 pt. Los datos y la paleta son los mismos que los de la figura del paper: se
importan de ensemble/make_statistics_figures.py, que sigue siendo la unica
fuente de verdad.

Uso:  python docs/make_slide_figures.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ensemble"))
from make_statistics_figures import COLORS, METHODS, INK, INK_2, GRID, LINE

SCORES = "results/paper_assets/scores/scores_per_case_test.csv"
STATS = "results/statistics/wilcoxon_pairwise.csv"
OUT = "docs/figs"


def style():
    plt.rcParams.update({
        "font.size": 15, "axes.labelsize": 16, "axes.titlesize": 17,
        "xtick.labelsize": 15, "ytick.labelsize": 14,
        "axes.edgecolor": GRID, "axes.linewidth": 1.2,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.labelcolor": INK, "text.color": INK,
        "grid.color": GRID, "grid.linewidth": 1.0,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def panel(ax, wide, metric, wilc, rng):
    x = np.arange(len(METHODS))

    for _, row in wide.iterrows():
        ax.plot(x, [row[m] for m in METHODS], color=LINE, lw=1.4, zorder=1)

    bp = ax.boxplot([wide[m].values for m in METHODS], positions=x,
                    widths=0.52, showfliers=False, patch_artist=True,
                    medianprops=dict(color=INK, lw=2.4),
                    whiskerprops=dict(color=INK_2, lw=1.5),
                    capprops=dict(color=INK_2, lw=1.5),
                    boxprops=dict(lw=1.5))
    for patch, m in zip(bp["boxes"], METHODS):
        patch.set_facecolor(COLORS[m])
        patch.set_alpha(0.20)
        patch.set_edgecolor(COLORS[m])

    for i, m in enumerate(METHODS):
        v = wide[m].values
        jitter = rng.uniform(-0.11, 0.11, size=len(v))
        ax.plot(i + jitter, v, "o", ms=8.5, color=COLORS[m],
                mec="white", mew=1.2, zorder=4)
        ax.plot([i - 0.26, i + 0.26], [v.mean()] * 2, color=COLORS[m],
                lw=3.2, ls=(0, (2.5, 1.5)), zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(["nnU-Net\n(FT)", "Attention\nU-Net (FT)",
                        "3D U-Net\n(FT)", "Ensemble"])
    ax.set_ylabel(metric)
    ax.set_ylim(-0.04, 1.16)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # p de Holm frente al ensemble
    sub = wilc[wilc.metric == metric].set_index("comparison")
    label = {"nnU-Net": "nnU-Net", "Attention": "Attention",
             "U-Net BCE": "U-Net BCE"}
    for i, m in enumerate(METHODS[:-1]):
        p = sub.loc[f"Ensemble vs {label[m]}"]["p_holm"]
        txt = ("$p$ < 0.001" if p < 0.001 else f"$p$ = {p:.3f}")
        star = " *" if p < 0.05 else ""
        ax.text(i, 1.08, txt + star, ha="center", fontsize=14,
                color=INK if p < 0.05 else INK_2,
                fontweight="bold" if p < 0.05 else "normal")
    ax.text(len(METHODS) - 1, 1.08, "referencia", ha="center", fontsize=14,
            color=INK_2, style="italic")


def main():
    style()
    df = pd.read_csv(SCORES)
    wilc = pd.read_csv(STATS)

    fig, ax = plt.subplots(figsize=(9.4, 4.5))
    wide = df.pivot(index="case", columns="method",
                    values="Dice").sort_index()
    panel(ax, wide, "Dice", wilc, np.random.default_rng(42))
    fig.tight_layout()

    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        p = f"{OUT}/fig_paired_dice_slide.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"-> {p}")


if __name__ == "__main__":
    main()
