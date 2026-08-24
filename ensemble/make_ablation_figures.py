"""
Figura de la ablacion del ensemble (Figure 6 del manuscrito).

Tres paneles, TODOS con metricas del conjunto de desarrollo (out-of-fold):

  (a) Dice en desarrollo frente al umbral de decision, para los siete esquemas
      de tres miembros de la rejilla obligatoria. Muestra que el optimo es una
      meseta y no un pico aislado.
  (b) Mapa de calor Dice(esquema, umbral) sobre los 13 esquemas obligatorios y
      los 19 umbrales del barrido fino, con la configuracion seleccionada
      marcada.
  (c) Ablacion de miembros: modelos individuales, pares y el ensemble de tres,
      cada uno en su mejor umbral de desarrollo, con IC 95 % bootstrap.

Lee unicamente los CSV de results/ablation/tables/ (que a su vez provienen solo de
predicciones de desarrollo). No accede a las probabilidades de test.

Uso:  python ensemble/make_ablation_figures.py
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

# Paleta categorica validada para vision con deficiencia de color
# (adjacent-pair CVD dE >= 8, normal-vision floor 19.6).
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
     "yellow": "#eda100", "magenta": "#e87ba4", "green": "#008300",
     "violet": "#4a3aa7"}
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d9d8d4"

SELECTED = ("2:2:3", 0.3)
MANDATORY_THRESHOLDS = [0.3, 0.5, 0.6]

# Etiquetas legibles de los 13 esquemas obligatorios
LABELS = {
    "1:0:0": "nnU-Net", "0:1:0": "Attention U-Net", "0:0:1": "3D U-Net",
    "1:1:0": "nnU-Net + Attention", "1:0:1": "nnU-Net + 3D U-Net",
    "0:1:1": "Attention + 3D U-Net", "1:1:1": "Equal 1:1:1",
    "2:2:3": "Selected 2:2:3", "2:1:1": "2:1:1", "1:2:1": "1:2:1",
    "1:1:2": "1:1:2", "3:2:2": "3:2:2", "2:3:2": "2:3:2",
}
SINGLES = ["1:0:0", "0:1:0", "0:0:1"]
PAIRS = ["1:1:0", "1:0:1", "0:1:1"]
TRIPLES = ["1:1:1", "2:2:3", "2:1:1", "1:2:1", "1:1:2", "3:2:2", "2:3:2"]


def style():
    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.labelcolor": INK, "text.color": INK,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def panel_a(ax, grid):
    """Dice(dev) vs umbral para los siete esquemas de tres miembros."""
    colors = [C["blue"], C["orange"], C["aqua"], C["yellow"],
              C["magenta"], C["green"], C["violet"]]
    # El seleccionado primero, en azul y con trazo grueso
    order = ["2:2:3"] + [w for w in TRIPLES if w != "2:2:3"]
    for w, col in zip(order, colors):
        d = grid[grid.weights == w].sort_values("threshold")
        sel = w == SELECTED[0]
        ax.plot(d.threshold, d.dice_mean, color=col,
                lw=2.4 if sel else 1.3, zorder=5 if sel else 3,
                label=LABELS[w], solid_capstyle="round")
    best = grid[(grid.weights == SELECTED[0])
                & (np.isclose(grid.threshold, SELECTED[1]))].iloc[0]
    ax.plot([SELECTED[1]], [best.dice_mean], "o", ms=8, color=C["blue"],
            mec="white", mew=2, zorder=6)
    ax.annotate(f"selected · {best.dice_mean:.3f}",
                (SELECTED[1], best.dice_mean), textcoords="offset points",
                xytext=(0, 13), fontsize=7.5, color=INK, ha="center",
                fontweight="bold")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Development Dice (mean, n = 10)")
    ax.set_title("(a) Threshold sweep, three-member schemes", loc="left")
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0, 0.72)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower left", frameon=False, ncol=2, handlelength=1.4)


def panel_b(ax, grid, fig):
    """Mapa de calor Dice(esquema, umbral) sobre la rejilla obligatoria."""
    rows = SINGLES + PAIRS + TRIPLES
    thrs = sorted(grid.threshold.unique())
    M = np.array([[grid[(grid.weights == w)
                        & (np.isclose(grid.threshold, t))].dice_mean.iloc[0]
                   for t in thrs] for w in rows])

    im = ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=M.max(),
                   interpolation="nearest")
    ax.set_xticks(range(len(thrs)))
    ax.set_xticklabels([f"{t:g}" for t in thrs], rotation=90)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABELS[w] for w in rows])
    ax.set_xlabel("Decision threshold")
    ax.set_title("(b) Development Dice across the mandatory grid", loc="left")

    # Separadores entre single / pairwise / triple
    for y in (len(SINGLES) - 0.5, len(SINGLES) + len(PAIRS) - 0.5):
        ax.axhline(y, color="white", lw=2)

    # Marcar la configuracion seleccionada
    r, c = rows.index(SELECTED[0]), thrs.index(SELECTED[1])
    ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                           edgecolor=C["orange"], lw=2.2, zorder=5))
    ax.annotate("selected", (c, r), textcoords="offset points",
                xytext=(14, 0), fontsize=7.5, color=C["orange"],
                va="center", fontweight="bold")
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.03)
    cb.set_label("Dice", fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7)


def panel_c(ax, grid):
    """Ablacion de miembros. El umbral de cada esquema es el mejor de los tres
    umbrales obligatorios (0.3 / 0.5 / 0.6), de modo que el panel refleja
    exactamente la rejilla que pidio la revision."""
    mand = grid[grid.threshold.isin(MANDATORY_THRESHOLDS)]
    groups = [("Single model", SINGLES, C["yellow"]),
              ("Pairwise", PAIRS, C["aqua"]),
              ("Three-model", ["2:2:3"], C["blue"])]
    labels, vals, los, his, cols = [], [], [], [], []
    for _, members, col in groups:
        for w in members:
            d = mand[mand.weights == w]
            b = d.loc[d.dice_mean.idxmax()]
            labels.append(f"{LABELS[w]}\n(thr {b.threshold:g})")
            vals.append(b.dice_mean)
            los.append(b.dice_mean - b.dice_ci_lo)
            his.append(b.dice_ci_hi - b.dice_mean)
            cols.append(col)

    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, height=0.68, color=cols, zorder=3)
    ax.errorbar(vals, y, xerr=[los, his], fmt="none", ecolor=INK_2,
                elinewidth=1.1, capsize=3, zorder=4)
    # Etiqueta de valor despues del extremo del intervalo, para no solaparlo
    for yi, v, h in zip(y, vals, his):
        ax.text(v + h + 0.022, yi, f"{v:.3f}", va="center", fontsize=7.5,
                color=INK, zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Development Dice (mean, 95 % bootstrap CI)")
    ax.set_title("(c) Member ablation, mandatory thresholds", loc="left")
    ax.set_xlim(0, 0.98)
    ax.set_ylim(-0.7, len(vals) - 0.2)
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in groups]
    ax.legend(handles, [g[0] for g in groups], loc="upper right",
              frameon=False, handlelength=1.2, borderaxespad=0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", default="results/ablation")
    args = ap.parse_args()
    style()

    figures = f"{args.ablation}/figures"
    os.makedirs(figures, exist_ok=True)

    grid = pd.read_csv(f"{args.ablation}/tables/ablation_grid_dev.csv")
    grid = grid[grid.weights.isin(LABELS)]

    fig = plt.figure(figsize=(16.5, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.3, 1.05], wspace=0.62,
                          left=0.05, right=0.995, top=0.855, bottom=0.145)
    panel_a(fig.add_subplot(gs[0]), grid)
    panel_b(fig.add_subplot(gs[1]), grid, fig)
    panel_c(fig.add_subplot(gs[2]), grid)

    fig.suptitle(
        "Ensemble ablation — all values from out-of-fold predictions on the "
        "clinical development subset (n = 10); the independent test cohort was "
        "not used for configuration comparison",
        y=0.975, fontsize=8.5, color=INK_2)

    for ext in ("png", "pdf"):
        p = f"{figures}/fig_ablation_dev.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"-> {p}")


if __name__ == "__main__":
    main()
