"""
Figura cualitativa POR RED (estilo Figure 3): para casos clínicos de test,
filas = 3D U-Net (BCE) / nnU-Net / Attention U-Net; columnas = Input / Ground
Truth / Prediction / Overlay. Predicciones de modelos individuales con umbral 0.5.

Genera un panel por caso en results/paper_assets/examples/.
Uso:  <python_con_numpy_matplotlib_scipy> ensemble/make_per_network_figs.py
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DP   = f"{REPO}/data_probs"
GT   = os.environ.get("DBT_DATA", f"{REPO}/data/real")
OUT  = f"{REPO}/results/paper_assets/examples"; os.makedirs(OUT, exist_ok=True)
THR  = 0.5   # umbral estándar para modelos individuales

# filas (modelo -> cargador de probabilidad)
def nn(c):  return np.load(f"{DP}/nnunet_test/real_dbt_{c}.npz")["probabilities"][1].astype(np.float32)
def att(c): return np.load(f"{DP}/attention_test/real_dbt_{c}.npz")["prob"].astype(np.float32)
def bce(c): return np.load(f"{DP}/unet_bce_test/real_dbt_{c}.npz")["prob"].astype(np.float32)
ROWS = [("3D U-Net", bce), ("nnU-Net", nn), ("Attention U-Net", att)]

def img(c):  return np.load(f"{GT}/real_dbt_{c}/real_dbt_{c}_img.npy").astype(np.float32)
def mask(c): return (np.load(f"{GT}/real_dbt_{c}/real_dbt_{c}_mask.npy") > 0)
def best_slice(g): return int(np.argmax(g.reshape(g.shape[0], -1).sum(1)))
def dice(p, g):
    s = p.sum() + g.sum()
    return 1.0 if s == 0 else float(2*(p & g).sum()/s)

RED  = (0.80, 0.10, 0.12)   # GT
BLUE = (0.12, 0.25, 0.70)   # prediction
PUR  = (0.55, 0.10, 0.55)   # overlap

BG_GT   = (0.99, 0.93, 0.93)   # fondo claro para el panel de ground truth
BG_PRED = (0.91, 0.94, 0.99)   # fondo claro para el panel de predicción

def mask_only(shape, m, color, bg):
    """Imagen RGB con la máscara `m` (color) sobre fondo plano `bg`."""
    rgb = np.ones((*shape, 3)) * np.array(bg)
    rgb[m] = color
    return rgb

def panel(c, tag):
    g = mask(c); v = img(c); z = best_slice(c if False else g)
    sl_img, sl_gt = v[z], g[z]
    vmin, vmax = np.percentile(sl_img, [1, 99])
    fig, axs = plt.subplots(len(ROWS), 4, figsize=(16, 6.6))
    cols = ["Input", "Ground Truth", "Prediction", "Overlay"]
    for j, t in enumerate(cols):
        axs[0, j].set_title(t, fontsize=12, fontweight="bold")
    for i, (name, loader) in enumerate(ROWS):
        sl_pr = (loader(c) >= THR)[z]
        d = dice(loader(c) >= THR, g)
        for j in range(4):
            axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
        # Input: imagen en escala de grises
        axs[i, 0].imshow(sl_img, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
        # Ground Truth: SOLO la máscara sobre fondo plano
        axs[i, 1].imshow(mask_only(sl_gt.shape, sl_gt, RED, BG_GT), aspect="equal")
        # Prediction: SOLO la máscara sobre fondo plano
        axs[i, 2].imshow(mask_only(sl_pr.shape, sl_pr, BLUE, BG_PRED), aspect="equal")
        # Overlay: imagen + GT + pred + solape
        axs[i, 3].imshow(sl_img, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
        ov = np.zeros((*sl_gt.shape, 4))
        ov[sl_gt & ~sl_pr] = [*RED, 0.55]; ov[sl_pr & ~sl_gt] = [*BLUE, 0.55]
        ov[sl_gt & sl_pr] = [*PUR, 0.75]; axs[i, 3].imshow(ov, aspect="equal")
        axs[i, 0].set_ylabel(f"{name}\n(Dice {d:.3f})", fontsize=11, rotation=90,
                             labelpad=12, va="center")
    leg = [Patch(facecolor=RED, label="Ground truth"),
           Patch(facecolor=BLUE, label="Prediction"),
           Patch(facecolor=PUR, label="Overlap (GT$\\cap$Pred)")]
    fig.legend(handles=leg, loc="lower center", ncol=3, fontsize=11, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"({tag}) Per-architecture segmentation — clinical case real_dbt_{c} (slice {z})",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    p = f"{OUT}/per_network_real_dbt_{c}.png"
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  {p}")
    return p

if __name__ == "__main__":
    # un caso "fácil/grande" y uno "difícil/pequeño" entre los clínicos de test
    print("Per-network qualitative panels:")
    panel("005", "A")   # tumor visible, buen desempeño
    panel("016", "B")   # caso intermedio
    panel("019", "C")   # tumor de muy bajo contraste (difícil)
    print("DONE")
