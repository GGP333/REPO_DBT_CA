# src/viz.py
from __future__ import annotations
import os
import random
from typing import List, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt

def _make_mosaic(img: np.ndarray, gt: np.ndarray, prob: np.ndarray, binp: np.ndarray, title: str, out_path: str) -> None:
    """
    Genera un mosaico de 4 columnas: original | GT | prob | bin
    """
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Original")
    axes[1].imshow(gt, cmap="gray")
    axes[1].set_title("GT")
    im = axes[2].imshow(prob, cmap="viridis")
    axes[2].set_title("Prob")
    axes[3].imshow(binp, cmap="gray")
    axes[3].set_title("Bin (0.5)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

@torch.no_grad()
def save_random_val_slices_grid(model: torch.nn.Module,
                                val_loader,
                                device: torch.device,
                                out_dir: str = "outputs/visualizations",
                                threshold: float = 0.5,
                                num_slices_total: int = 50,
                                seed: int = 42) -> List[str]:
    """
    Muestrea num_slices_total cortes aleatorios de la validación y guarda mosaicos.

    Retorna lista de rutas PNG generadas.
    """
    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)
    saved = []

    # Recolectar todas las predicciones y GT para elegir cortes aleatorios globales
    vols = []  # lista de (study_id, img(Z,H,W), gt(Z,H,W), prob(Z,H,W))
    for batch in val_loader:
        imgs: torch.Tensor = batch["image"].to(device)  # (B,1,Zp,Hp,Wp)
        probs = model(imgs).cpu().numpy()               # (B,1,Zp,Hp,Wp)
        shapes = batch["shapes"]
        study_ids = batch["study_ids"]
        # Recortar a shapes originales
        for i, sid in enumerate(study_ids):
            Z,H,W = shapes[i]
            img = batch["image"][i,0,:Z,:H,:W].cpu().numpy()
            gt  = batch["mask"][i,0,:Z,:H,:W].cpu().numpy()
            pr  = probs[i,0,:Z,:H,:W]
            vols.append((sid, img, gt, pr))

    # Si hay menos de num_slices_total, limitar
    total_slices = sum(v[1].shape[0] for v in vols)
    k = min(num_slices_total, total_slices)

    # Muestreo global: (vol_idx, z_idx)
    pool = []
    for vi, (_, img, _, _) in enumerate(vols):
        Z = img.shape[0]
        for z in range(Z):
            pool.append((vi, z))
    random.shuffle(pool)
    picks = pool[:k]

    for j, (vi, z) in enumerate(picks, 1):
        sid, img, gt, pr = vols[vi]
        binp = (pr[z] >= threshold).astype(np.float32)
        title = f"{sid} | slice {z} | img[min={img[z].min():.3f}, max={img[z].max():.3f}] prob[min={pr[z].min():.3f}, max={pr[z].max():.3f}] shape={img.shape}"
        out_path = os.path.join(out_dir, f"{sid}_slice{z:03d}_{j:02d}.png")
        _make_mosaic(img[z], gt[z], pr[z], binp, title, out_path)
        saved.append(out_path)

    return saved
