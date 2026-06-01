"""
Genera TODOS los artefactos para el paper de la contribución ensemble sim->real:

  A) SCORES   : Dice / IoU / Precision / Recall por caso y agregados, para cada
                miembro (nnU-Net, Attention, U-Net BCE) y el ENSEMBLE ganador.
                -> CSV (per-case + summary) y figuras de comparación.
  B) EJEMPLOS : para los 10 casos de test, overlay imagen + GT + predicción del
                ensemble en la slice más representativa, anotado con su Dice.
                -> 1 PNG por caso + un montaje (marcando excelente/bueno/regular).
  C) CURVAS   : - fine-tuning de nnU-Net (miembro principal): train_loss, val_loss
                  y Pseudo-Dice por época, media±banda de los 5 folds.
                - entrenamiento base sintético (Attention / U-Net BCE): train_loss
                  y val Dice/IoU/Precision/Recall por época.
                -> PNG + CSV.

Todo se calcula desde data_probs/ + data/real/ (sin GPU, sin reentrenar) y desde
los logs reales en results/finetuning_logs/ y results/outputs_*/.

Uso:  <python_con_numpy_matplotlib> ensemble/make_paper_assets.py
Salidas en: results/paper_assets/{scores,examples,curves}/
"""
import os, re, glob, csv, json, statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ----------------------------------------------------------------------------
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DP   = f"{REPO}/data_probs"
GT   = os.environ.get("DBT_DATA", f"{REPO}/data/real")
OUT  = f"{REPO}/results/paper_assets"
for sub in ("scores", "examples", "curves"):
    os.makedirs(f"{OUT}/{sub}", exist_ok=True)

TEST = ["005", "006", "007", "010", "013", "014", "015", "016", "019", "020"]
DEV  = ["001", "002", "003", "004", "008", "009", "011", "012", "017", "018"]

# nombre de la carpeta de probabilidades por split
SPLIT_DIR = {"test": "test", "dev": "OOF_dev"}

# Configuración ganadora del ensemble (elegida en OOF dev, sin tocar test)
W   = {"nn": 2, "att": 2, "bce": 3}
THR_ENS = 0.3      # umbral del ensemble (seleccionado en dev)
THR_SINGLE = 0.5   # umbral estándar para modelos individuales

# ----------------------------------------------------------------------------
# Carga de probabilidades y máscaras
def nn_prob(c, split):
    return np.load(f"{DP}/nnunet_{SPLIT_DIR[split]}/real_dbt_{c}.npz")["probabilities"][1].astype(np.float32)
def cust_prob(net, c, split):
    return np.load(f"{DP}/{net}_{SPLIT_DIR[split]}/real_dbt_{c}.npz")["prob"].astype(np.float32)
def img(c):
    return np.load(f"{GT}/real_dbt_{c}/real_dbt_{c}_img.npy").astype(np.float32)
def mask(c):
    return (np.load(f"{GT}/real_dbt_{c}/real_dbt_{c}_mask.npy") > 0)

def probs_for(c, split):
    """Devuelve dict de probabilidades por método + ensemble para un caso."""
    nn  = nn_prob(c, split)
    att = cust_prob("attention", c, split)
    bce = cust_prob("unet_bce", c, split)
    ens = (W["nn"]*nn + W["att"]*att + W["bce"]*bce) / sum(W.values())
    return {"nnU-Net": nn, "Attention": att, "U-Net BCE": bce, "Ensemble": ens}

# ----------------------------------------------------------------------------
# Métricas
def metrics(pred, g):
    pred = pred.astype(bool); g = g.astype(bool)
    tp = int((pred & g).sum()); fp = int((pred & ~g).sum()); fn = int((~pred & g).sum())
    union = tp + fp + fn
    dice = 1.0 if (pred.sum() + g.sum()) == 0 else 2*tp / (pred.sum() + g.sum())
    iou  = 1.0 if union == 0 else tp / union
    prec = 1.0 if (tp + fp) == 0 else tp / (tp + fp)
    rec  = 1.0 if (tp + fn) == 0 else tp / (tp + fn)
    return dict(dice=dice, iou=iou, precision=prec, recall=rec)

METHODS = ["nnU-Net", "Attention", "U-Net BCE", "Ensemble"]
METRIC_KEYS = ["dice", "iou", "precision", "recall"]
METRIC_LABEL = {"dice": "Dice", "iou": "IoU", "precision": "Precision", "recall": "Recall"}


# ============================================================================
# A) SCORES
# ============================================================================
def compute_scores(split, ids):
    """per_case[method][case] = {dice,iou,...}"""
    per = {m: {} for m in METHODS}
    for c in ids:
        pr = probs_for(c, split)
        g = mask(c)
        for m in METHODS:
            thr = THR_ENS if m == "Ensemble" else THR_SINGLE
            per[m][c] = metrics(pr[m] >= thr, g)
    return per

def write_scores_csv(per, ids, split):
    # per-case largo
    path_pc = f"{OUT}/scores/scores_per_case_{split}.csv"
    with open(path_pc, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["case", "method", "Dice", "IoU", "Precision", "Recall"])
        for c in ids:
            for m in METHODS:
                d = per[m][c]
                w.writerow([f"real_dbt_{c}", m,
                            f"{d['dice']:.4f}", f"{d['iou']:.4f}",
                            f"{d['precision']:.4f}", f"{d['recall']:.4f}"])
    # resumen por método (media, std, mediana)
    path_sm = f"{OUT}/scores/scores_summary_{split}.csv"
    with open(path_sm, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "metric", "mean", "std", "median", "min", "max"])
        for m in METHODS:
            for k in METRIC_KEYS:
                xs = [per[m][c][k] for c in ids]
                w.writerow([m, METRIC_LABEL[k], f"{np.mean(xs):.4f}", f"{np.std(xs):.4f}",
                            f"{np.median(xs):.4f}", f"{np.min(xs):.4f}", f"{np.max(xs):.4f}"])
    return path_pc, path_sm

def fig_scores_comparison(per, ids, split):
    """Barras agrupadas: métrica en X, una barra por método (media ± std)."""
    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(METRIC_KEYS)); width = 0.2
    colors = {"nnU-Net": "#4C72B0", "Attention": "#55A868", "U-Net BCE": "#C44E52", "Ensemble": "#8172B3"}
    for i, m in enumerate(METHODS):
        means = [np.mean([per[m][c][k] for c in ids]) for k in METRIC_KEYS]
        stds  = [np.std([per[m][c][k] for c in ids]) for k in METRIC_KEYS]
        bars = ax.bar(x + (i - 1.5)*width, means, width, yerr=stds, capsize=3,
                      label=m, color=colors[m], edgecolor="black", linewidth=0.5,
                      error_kw=dict(lw=0.8))
        for b, mv in zip(bars, means):
            ax.text(b.get_x() + b.get_width()/2, mv + 0.012, f"{mv:.2f}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([METRIC_LABEL[k] for k in METRIC_KEYS])
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.0)
    ax.set_title(f"Segmentation metrics on {('test' if split=='test' else 'dev (OOF)')} "
                 f"({len(ids)} real cases) — members vs. ensemble")
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.grid(axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    p = f"{OUT}/scores/fig_scores_comparison_{split}.png"
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    return p

def fig_per_case_dice(per, ids, split):
    """Dice por caso, ensemble vs miembros (barras agrupadas por caso)."""
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(ids)); width = 0.2
    colors = {"nnU-Net": "#4C72B0", "Attention": "#55A868", "U-Net BCE": "#C44E52", "Ensemble": "#8172B3"}
    for i, m in enumerate(METHODS):
        vals = [per[m][c]["dice"] for c in ids]
        ax.bar(x + (i - 1.5)*width, vals, width, label=m, color=colors[m],
               edgecolor="black", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels([f"dbt_{c}" for c in ids], rotation=45, ha="right")
    ax.set_ylabel("Dice"); ax.set_ylim(0, 1.0)
    ens_mean = np.mean([per["Ensemble"][c]["dice"] for c in ids])
    ax.axhline(ens_mean, color="#8172B3", ls="--", lw=1, alpha=0.8,
               label=f"Ensemble mean = {ens_mean:.3f}")
    ax.set_title(f"Per-case Dice on {('test' if split=='test' else 'dev (OOF)')}")
    ax.legend(ncol=5, fontsize=8); ax.grid(axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    p = f"{OUT}/scores/fig_per_case_dice_{split}.png"
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    return p


# ============================================================================
# B) EJEMPLOS (test)
# ============================================================================
def best_slice(g):
    """slice con mayor área de GT."""
    areas = g.reshape(g.shape[0], -1).sum(1)
    return int(np.argmax(areas))

def contour(binary):
    """borde de una máscara binaria 2D (para dibujar contornos)."""
    from scipy.ndimage import binary_erosion
    return binary & ~binary_erosion(binary)

def category(dice):
    if dice >= 0.65: return "Excelente", "#2ca02c"
    if dice >= 0.45: return "Bueno", "#ff7f0e"
    return "Regular", "#d62728"

def tri_overlay(sl_gt, sl_pr):
    """Overlay RGBA con convención TP (amarillo) / FP (rojo) / FN (verde)."""
    tp = sl_gt & sl_pr; fp = sl_pr & ~sl_gt; fn = sl_gt & ~sl_pr
    ov = np.zeros((*sl_gt.shape, 4))
    ov[fn] = [0, 1, 0, 0.55]   # solo GT  -> falso negativo (verde)
    ov[fp] = [1, 0, 0, 0.55]   # solo pred -> falso positivo (rojo)
    ov[tp] = [1, 1, 0, 0.65]   # overlap   -> verdadero positivo (amarillo)
    return ov

TRI_LEGEND = [Patch(facecolor=(1, 1, 0, 0.65), label="Overlap (TP)"),
              Patch(facecolor=(1, 0, 0, 0.55), label="Pred only (FP)"),
              Patch(facecolor=(0, 1, 0, 0.55), label="GT only (FN)")]

RED_M, BLUE_M, PUR_M = (0.80, 0.10, 0.12), (0.12, 0.25, 0.70), (0.55, 0.10, 0.55)
BG_GT_M, BG_PR_M = (0.99, 0.93, 0.93), (0.91, 0.94, 0.99)

def _mask_only(shape, m, color, bg):
    rgb = np.ones((*shape, 3)) * np.array(bg); rgb[m] = color; return rgb

def fig_example(c, dice_ens):
    """4 paneles: Input | Ground Truth (máscara) | Prediction (máscara) | Overlay."""
    g = mask(c); v = img(c); z = best_slice(g)
    pr = probs_for(c, "test")["Ensemble"] >= THR_ENS
    sl_img = v[z]; sl_gt = g[z]; sl_pr = pr[z]
    cat, col = category(dice_ens)

    fig, axs = plt.subplots(1, 4, figsize=(16, 3.1))
    vmin, vmax = np.percentile(sl_img, [1, 99])
    for ax in axs: ax.axis("off")
    axs[0].imshow(sl_img, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
    axs[0].set_title("Input", fontsize=11, fontweight="bold")
    axs[1].imshow(_mask_only(sl_gt.shape, sl_gt, RED_M, BG_GT_M), aspect="equal")
    axs[1].set_title("Ground Truth", fontsize=11, fontweight="bold")
    axs[2].imshow(_mask_only(sl_pr.shape, sl_pr, BLUE_M, BG_PR_M), aspect="equal")
    axs[2].set_title("Prediction", fontsize=11, fontweight="bold")
    axs[3].imshow(sl_img, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
    ov = np.zeros((*sl_gt.shape, 4))
    ov[sl_gt & ~sl_pr] = [*RED_M, 0.55]; ov[sl_pr & ~sl_gt] = [*BLUE_M, 0.55]
    ov[sl_gt & sl_pr] = [*PUR_M, 0.75]; axs[3].imshow(ov, aspect="equal")
    axs[3].set_title("Overlay", fontsize=11, fontweight="bold")

    fig.suptitle(f"real_dbt_{c}   |   Ensemble Dice = {dice_ens:.3f}   |   {cat}",
                 fontsize=12, color=col, y=1.03, fontweight="bold")
    leg = [Patch(facecolor=RED_M, label="Ground truth"),
           Patch(facecolor=BLUE_M, label="Prediction"),
           Patch(facecolor=PUR_M, label="Overlap (GT$\\cap$Pred)")]
    fig.legend(handles=leg, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    p = f"{OUT}/examples/example_real_dbt_{c}.png"
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    return p

def fig_examples_grid(dice_by_case):
    """Montaje 2x5: cada caso, slice repr., GT (verde) + pred (rojo) overlay, etiqueta de categoría."""
    fig, axs = plt.subplots(2, 5, figsize=(20, 5.4))
    for ax, c in zip(axs.ravel(), TEST):
        g = mask(c); v = img(c); z = best_slice(g)
        pr = probs_for(c, "test")["Ensemble"] >= THR_ENS
        sl_img, sl_gt, sl_pr = v[z], g[z], pr[z]
        vmin, vmax = np.percentile(sl_img, [1, 99])
        ax.imshow(sl_img, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
        ax.imshow(tri_overlay(sl_gt, sl_pr), aspect="equal")
        d = dice_by_case[c]; cat, col = category(d)
        ax.set_title(f"real_dbt_{c}\nDice {d:.3f} · {cat}", fontsize=10, color=col)
        ax.axis("off")
    fig.legend(handles=TRI_LEGEND, loc="lower center", ncol=3, fontsize=11, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Qualitative results — ensemble predictions on the 10 held-out real test cases",
                 fontsize=15, y=1.0, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.99])
    p = f"{OUT}/examples/examples_grid_all_test.png"
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    return p

def fig_examples_grid_4col(dice_by_case):
    """Grilla 10x4: una fila por caso, columnas Input | Ground Truth | Prediction | Overlay."""
    n = len(TEST)
    fig, axs = plt.subplots(n, 4, figsize=(11, 1.55*n + 0.6))
    heads = ["Input", "Ground Truth", "Prediction", "Overlay"]
    for j, t in enumerate(heads):
        axs[0, j].set_title(t, fontsize=12, fontweight="bold")
    for i, c in enumerate(TEST):
        g = mask(c); v = img(c); z = best_slice(g)
        pr = probs_for(c, "test")["Ensemble"] >= THR_ENS
        sl_img, sl_gt, sl_pr = v[z], g[z], pr[z]
        vmin, vmax = np.percentile(sl_img, [1, 99])
        for j in range(4): axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
        axs[i, 0].imshow(sl_img, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
        axs[i, 1].imshow(_mask_only(sl_gt.shape, sl_gt, RED_M, BG_GT_M), aspect="equal")
        axs[i, 2].imshow(_mask_only(sl_pr.shape, sl_pr, BLUE_M, BG_PR_M), aspect="equal")
        axs[i, 3].imshow(sl_img, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
        ov = np.zeros((*sl_gt.shape, 4))
        ov[sl_gt & ~sl_pr] = [*RED_M, 0.55]; ov[sl_pr & ~sl_gt] = [*BLUE_M, 0.55]
        ov[sl_gt & sl_pr] = [*PUR_M, 0.75]; axs[i, 3].imshow(ov, aspect="equal")
        d = dice_by_case[c]; cat, col = category(d)
        axs[i, 0].set_ylabel(f"real_dbt_{c}\nDice {d:.3f}\n{cat}", fontsize=9, color=col,
                             rotation=90, labelpad=10, va="center")
    leg = [Patch(facecolor=RED_M, label="Ground truth"),
           Patch(facecolor=BLUE_M, label="Prediction"),
           Patch(facecolor=PUR_M, label="Overlap (GT$\\cap$Pred)")]
    fig.legend(handles=leg, loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Qualitative ensemble results on the 10 held-out clinical test cases",
                 fontsize=14, y=0.997, fontweight="bold")
    fig.tight_layout(rect=[0, 0.012, 1, 0.99])
    p = f"{OUT}/examples/examples_grid_4col_all_test.png"
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    return p


# ============================================================================
# C) CURVAS
# ============================================================================
def parse_nnunet_log(path):
    """Devuelve listas por época: train_loss, val_loss, pseudo_dice."""
    ep, tr, vl, ps = -1, {}, {}, {}
    for line in open(path, encoding="utf-8", errors="ignore"):
        m = re.search(r"Epoch (\d+)\s*$", line.strip())
        if m: ep = int(m.group(1)); continue
        m = re.search(r"train_loss\s+(-?[\d.]+)", line)
        if m: tr[ep] = float(m.group(1)); continue
        m = re.search(r"val_loss\s+(-?[\d.]+)", line)
        if m: vl[ep] = float(m.group(1)); continue
        m = re.search(r"Pseudo dice.*?(\d+\.\d+)", line)
        if m: ps[ep] = float(m.group(1)); continue
    n = max(max(tr or [0]), max(vl or [0]), max(ps or [0])) + 1
    f = lambda d: np.array([d.get(i, np.nan) for i in range(n)])
    return f(tr), f(vl), f(ps)

def nnunet_ft_curves():
    folds = sorted(glob.glob(f"{REPO}/results/finetuning_logs/nnunet_FT_Tversky/fold_*/training_log.txt"))
    if not folds: return None
    TR, VL, PS = [], [], []
    for fp in folds:
        tr, vl, ps = parse_nnunet_log(fp); TR.append(tr); VL.append(vl); PS.append(ps)
    n = min(len(a) for a in TR)
    TR = np.array([a[:n] for a in TR]); VL = np.array([a[:n] for a in VL]); PS = np.array([a[:n] for a in PS])
    ep = np.arange(n)
    # CSV per-epoch (media de folds)
    with open(f"{OUT}/curves/nnunet_ft_per_epoch.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss_mean", "val_loss_mean", "pseudo_dice_mean",
                    "pseudo_dice_std", "n_folds"])
        for i in ep:
            w.writerow([i, f"{np.nanmean(TR[:,i]):.4f}", f"{np.nanmean(VL[:,i]):.4f}",
                        f"{np.nanmean(PS[:,i]):.4f}", f"{np.nanstd(PS[:,i]):.4f}", TR.shape[0]])
    # figura: loss (train/val) + pseudo dice
    fig, ax1 = plt.subplots(figsize=(9, 5.2))
    ax1.plot(ep, np.nanmean(TR, 0), color="#4C72B0", label="train loss (mean 5-fold)")
    ax1.fill_between(ep, np.nanmean(TR,0)-np.nanstd(TR,0), np.nanmean(TR,0)+np.nanstd(TR,0),
                     color="#4C72B0", alpha=0.15)
    ax1.plot(ep, np.nanmean(VL, 0), color="#C44E52", label="val loss (mean 5-fold)")
    ax1.fill_between(ep, np.nanmean(VL,0)-np.nanstd(VL,0), np.nanmean(VL,0)+np.nanstd(VL,0),
                     color="#C44E52", alpha=0.15)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss (Tversky+CE)")
    ax2 = ax1.twinx()
    ax2.plot(ep, np.nanmean(PS, 0), color="#55A868", lw=2, label="val pseudo-Dice (mean)")
    ax2.fill_between(ep, np.nanmean(PS,0)-np.nanstd(PS,0), np.nanmean(PS,0)+np.nanstd(PS,0),
                     color="#55A868", alpha=0.18)
    ax2.set_ylabel("Validation pseudo-Dice"); ax2.set_ylim(0, 1)
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc="upper center", bbox_to_anchor=(0.5, -0.12),
               ncol=3, fontsize=8, frameon=False)
    ax1.set_title("nnU-Net fine-tuning on real DBT (sim→real, Tversky+CE) — 5-fold mean ± std")
    ax1.grid(ls=":", alpha=0.5); fig.tight_layout()
    p = f"{OUT}/curves/nnunet_ft_curves.png"
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    return p

def nnunet_train_val_loss():
    """Curva train vs val loss de nnU-Net (DC+CE) por dataset, desde sus training_log."""
    import glob as _glob
    DS = [("Large tumor", "Dataset001_DBTLarge"),
          ("Small tumor", "Dataset002_DBTSmall"),
          ("Mixed-size", "Dataset003_DBTMixedSize"),
          ("Hybrid synthetic-clinical", "Dataset004_DBTHybrid")]
    base = f"{REPO}/experiments/nnunet_official/nnunet_root/results"
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.2)); axs = axs.ravel()
    wrote_csv = open(f"{OUT}/curves/nnunet_train_val_loss_per_epoch.csv", "w", newline="")
    cw = csv.writer(wrote_csv); cw.writerow(["dataset", "epoch", "train_loss", "val_loss"])
    made = False
    for ax, (name, ds) in zip(axs, DS):
        hits = _glob.glob(f"{base}/{ds}/*Seeded42*/fold_0/training_log*.txt")
        if not hits:
            ax.set_visible(False); continue
        tr, vl, _ = parse_nnunet_log(hits[0]); ep = np.arange(len(tr))
        ax.plot(ep, tr, color="#4C72B0", lw=1.3, label="train loss")
        ax.plot(ep, vl, color="#C44E52", lw=1.3, label="val loss")
        # marca del mejor checkpoint (mínima val loss)
        if np.isfinite(vl).any():
            b = int(np.nanargmin(vl))
            ax.axvline(b, color="gray", ls="--", lw=0.8, alpha=0.7)
            ax.annotate(f"best val\nep {b}", (b, np.nanmin(vl)), fontsize=7,
                        color="gray", xytext=(6, 6), textcoords="offset points")
        ax.set_title(name, fontsize=11); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (DC+CE)")
        ax.grid(ls=":", alpha=0.5); ax.legend(fontsize=8, loc="upper right")
        for i in ep: cw.writerow([ds, i, f"{tr[i]:.4f}", f"{vl[i]:.4f}"])
        made = True
    wrote_csv.close()
    if not made:
        plt.close(fig); return None
    fig.suptitle("nnU-Net training vs. validation loss (DC+CE) across dataset configurations",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = f"{OUT}/curves/nnunet_train_val_loss.png"
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    return p

def all_arch_train_val_curves():
    """Grid 3x4 (arquitecturas x datasets): train loss + Dice de validación.
    val_loss solo lo registró nnU-Net; la señal de validación común a las 3 redes
    es el Dice de validación (val_dice para custom; pseudo-Dice para nnU-Net)."""
    import glob as _glob
    DSETS = [("Large tumor", "large_tumor", "Dataset001_DBTLarge"),
             ("Small tumor", "small_tumor", "Dataset002_DBTSmall"),
             ("Mixed-size",  "Mixed_Size",        "Dataset003_DBTMixedSize"),
             ("Hybrid",      "Hybrid", "Dataset004_DBTHybrid")]
    # rutas de metrics.csv de las redes custom en el repo
    def custom_csv(net, key):
        cands = [f"{REPO}/results/outputs_clean/{net}_Dataset_{key}/logs/metrics.csv",
                 f"{REPO}/results/outputs_improved/{net}_Dataset_{key}/logs/metrics.csv"]
        for c in cands:
            if os.path.exists(c): return c
        return None
    def read_csv_curve(path):
        rows = list(csv.DictReader(open(path)))
        ep = np.array([int(r["epoch"]) for r in rows])
        tr = np.array([float(r["train_loss"]) for r in rows])
        vd = np.array([float(r["val_dice"]) for r in rows])
        return ep, tr, vd
    def nnunet_curve(ds):
        hits = _glob.glob(f"{REPO}/experiments/nnunet_official/nnunet_root/results/{ds}/*Seeded42*/fold_0/training_log*.txt")
        if not hits: return None
        tr, _, ps = parse_nnunet_log(hits[0]); ep = np.arange(len(tr))
        return ep, tr, ps

    ROWS = [("3D U-Net", "UNet_BCE"), ("nnU-Net", None), ("Attention U-Net", "Attention_UNet")]
    fig, axs = plt.subplots(3, 4, figsize=(16, 9.5))
    csvf = open(f"{OUT}/curves/all_arch_train_val_per_epoch.csv", "w", newline="")
    cw = csv.writer(csvf); cw.writerow(["architecture", "dataset", "epoch", "train_loss", "val_dice"])
    for r, (arch, net) in enumerate(ROWS):
        for c, (label, key, ds) in enumerate(DSETS):
            ax = axs[r, c]
            data = nnunet_curve(ds) if net is None else (
                read_csv_curve(custom_csv(net, key)) if custom_csv(net, key) else None)
            if data is None:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center"); ax.set_axis_off(); continue
            ep, tr, vd = data
            ax.plot(ep, tr, color="#4C72B0", lw=1.0, label="train loss")
            ax.set_ylabel("Train loss", color="#4C72B0", fontsize=8)
            ax.tick_params(axis="y", labelcolor="#4C72B0", labelsize=7)
            ax.tick_params(axis="x", labelsize=7)
            ax2 = ax.twinx()
            ax2.plot(ep, vd, color="#2ca02c", lw=1.2, label="val Dice")
            ax2.set_ylim(0, 1); ax2.set_ylabel("Val Dice", color="#2ca02c", fontsize=8)
            ax2.tick_params(axis="y", labelcolor="#2ca02c", labelsize=7)
            if np.isfinite(vd).any():
                b = int(np.nanargmax(vd))
                ax2.axvline(b, color="gray", ls="--", lw=0.7, alpha=0.7)
            if r == 0: ax.set_title(label, fontsize=11, fontweight="bold")
            if c == 0: ax.text(-0.32, 0.5, arch, transform=ax.transAxes, rotation=90,
                               va="center", ha="center", fontsize=11, fontweight="bold")
            ax.grid(ls=":", alpha=0.4)
            for k in range(len(ep)):
                cw.writerow([arch, ds, int(ep[k]), f"{tr[k]:.4f}", f"{vd[k]:.4f}"])
    csvf.close()
    fig.suptitle("Training loss and validation Dice across architectures and dataset configurations",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0.02, 0, 1, 0.97])
    p = f"{OUT}/curves/all_arch_train_val_curves.png"
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    return p

def base_curves():
    """Curvas train/val del entrenamiento base sintético (Mixed_Size) de Attention y U-Net BCE."""
    srcs = {
        "Attention U-Net": f"{REPO}/results/outputs_improved/Attention_UNet_Dataset_Mixed_Size/logs/metrics.csv",
        "U-Net BCE":       f"{REPO}/results/outputs_clean/UNet_BCE_Dataset_Mixed_Size/logs/metrics.csv",
    }
    made = []
    for name, path in srcs.items():
        if not os.path.exists(path): continue
        rows = list(csv.DictReader(open(path)))
        ep = np.array([int(r["epoch"]) for r in rows])
        def col(k): return np.array([float(r[k]) if r.get(k) not in (None, "") else np.nan for r in rows])
        tr = col("train_loss"); vd = col("val_dice"); vi = col("val_iou")
        vp = col("val_precision"); vr = col("val_recall")
        fig, ax1 = plt.subplots(figsize=(9, 5.2))
        ax1.plot(ep, tr, color="#4C72B0", label="train loss")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train loss")
        ax2 = ax1.twinx()
        ax2.plot(ep, vd, color="#55A868", label="val Dice")
        ax2.plot(ep, vi, color="#8172B3", label="val IoU", alpha=0.8)
        ax2.plot(ep, vp, color="#ff7f0e", label="val Precision", alpha=0.7, ls="--")
        ax2.plot(ep, vr, color="#C44E52", label="val Recall", alpha=0.7, ls="--")
        ax2.set_ylabel("Validation metric"); ax2.set_ylim(0, 1)
        h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc="center right", fontsize=8)
        ax1.set_title(f"{name} — base training on synthetic 'Mixed_Size' (warm-start source)")
        ax1.grid(ls=":", alpha=0.5); fig.tight_layout()
        tag = "attention" if "Attention" in name else "unet_bce"
        p = f"{OUT}/curves/base_{tag}_curves.png"
        fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig); made.append(p)
    return made


# ============================================================================
def main():
    print("== A) SCORES ==")
    summaries = {}
    for split, ids in [("test", TEST), ("dev", DEV)]:
        per = compute_scores(split, ids)
        pc, sm = write_scores_csv(per, ids, split)
        f1 = fig_scores_comparison(per, ids, split)
        f2 = fig_per_case_dice(per, ids, split)
        ens = {c: per["Ensemble"][c]["dice"] for c in ids}
        summaries[split] = (per, ens)
        em = np.mean(list(ens.values()))
        print(f"  [{split}] ensemble Dice mean={em:.4f} | CSVs+figs -> {pc}, {sm}, {f1}, {f2}")

    print("== B) EJEMPLOS (test) ==")
    per_test, ens_test = summaries["test"]
    for c in TEST:
        fig_example(c, ens_test[c])
    grid = fig_examples_grid(ens_test)
    grid4 = fig_examples_grid_4col(ens_test)
    ranked = sorted(TEST, key=lambda c: ens_test[c], reverse=True)
    print(f"  10 PNG + montaje -> {grid}")
    print(f"  montaje 4-col   -> {grid4}")
    print(f"  excelente={ranked[0]}({ens_test[ranked[0]]:.3f}) "
          f"bueno={ranked[len(ranked)//2]}({ens_test[ranked[len(ranked)//2]]:.3f}) "
          f"regular={ranked[-1]}({ens_test[ranked[-1]]:.3f})")

    print("== C) CURVAS ==")
    p = nnunet_ft_curves(); print(f"  nnU-Net FT -> {p}")
    p = nnunet_train_val_loss(); print(f"  nnU-Net train/val loss -> {p}")
    p = all_arch_train_val_curves(); print(f"  all-arch train/val Dice -> {p}")
    for p in base_curves(): print(f"  base -> {p}")
    print("\nDONE. Todo en results/paper_assets/")

if __name__ == "__main__":
    main()
