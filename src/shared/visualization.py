"""
Generador de samples PNG para inspección visual de predicciones.
Produce grids [input | GT | prediction | overlay] para slices representativos.
"""
from __future__ import annotations
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _pick_representative_slices(mask_3d: np.ndarray, n: int = 3) -> List[int]:
    """Elige slices Z representativos: primera con fg, media, última con fg."""
    fg_slices = [z for z in range(mask_3d.shape[0]) if mask_3d[z].any()]
    if not fg_slices:
        # Sin foreground: devolver slices equiespaciados
        total = mask_3d.shape[0]
        return [0, total // 2, total - 1][:n]
    first_fg = fg_slices[0]
    last_fg = fg_slices[-1]
    mid_fg = fg_slices[len(fg_slices) // 2]
    candidates = sorted(set([first_fg, mid_fg, last_fg]))
    return candidates[:n]


def save_sample_png(
    img: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    study_id: str,
    out_dir: str | Path,
    pred_probs: Optional[np.ndarray] = None,
) -> Path:
    """
    Guarda un PNG con grid [input | GT | pred | overlay] para 3 slices representativos.

    Args:
        img: (Z,H,W) float normalizado [0,1]
        gt_mask: (Z,H,W) binario {0,1}
        pred_mask: (Z,H,W) binario {0,1}
        study_id: ID del estudio
        out_dir: directorio de salida
        pred_probs: (Z,H,W) probabilidades opcionales
    
    Returns:
        Path del archivo PNG generado
    """
    if not HAS_MPL:
        return Path(out_dir) / f"sample_{study_id}.png"

    out_dir = Path(out_dir)
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
        # Input
        axes[row, 0].imshow(img[z], cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_ylabel(f"Z={z}", fontsize=10)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        # GT
        axes[row, 1].imshow(gt_mask[z], cmap="Reds", vmin=0, vmax=1)
        axes[row, 1].set_xticks([])
        axes[row, 1].set_yticks([])

        # Prediction
        axes[row, 2].imshow(pred_mask[z], cmap="Blues", vmin=0, vmax=1)
        axes[row, 2].set_xticks([])
        axes[row, 2].set_yticks([])

        # Overlay: input + GT en rojo + pred en azul
        overlay = np.stack([img[z]] * 3, axis=-1)  # (H,W,3)
        overlay = np.clip(overlay, 0, 1)
        # GT en rojo
        overlay[..., 0] = np.clip(overlay[..., 0] + gt_mask[z] * 0.5, 0, 1)
        # Pred en azul
        overlay[..., 2] = np.clip(overlay[..., 2] + pred_mask[z] * 0.5, 0, 1)
        axes[row, 3].imshow(overlay)
        axes[row, 3].set_xticks([])
        axes[row, 3].set_yticks([])

    # Agregar leyenda al overlay
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="red", alpha=0.5, label="GT"),
        Patch(facecolor="blue", alpha=0.5, label="Pred"),
    ]
    axes[-1, 3].legend(handles=legend_elements, loc="lower right", fontsize=8)

    fig.suptitle(f"Sample: {study_id}", fontsize=14, fontweight="bold")
    fig.tight_layout()

    out_path = out_dir / f"sample_{study_id}.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_validation_samples(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    out_dir: str | Path,
    n_samples: int = 5,
    threshold: float = 0.5,
) -> List[Path]:
    """
    Genera PNG de samples aleatorios del validation set.

    Args:
        model: modelo entrenado (en eval mode)
        val_loader: DataLoader de validación
        device: dispositivo
        out_dir: directorio de salida
        n_samples: número de muestras a generar
        threshold: umbral de binarización
    
    Returns:
        Lista de paths de PNGs generados
    """
    if not HAS_MPL:
        print("[WARN] matplotlib no disponible, saltando generación de samples PNG")
        return []

    model.eval()
    out_dir = Path(out_dir) / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_batches = list(val_loader)
    # Seleccionar batches aleatorios
    rng = random.Random(42)
    selected = rng.sample(range(len(all_batches)), min(n_samples, len(all_batches)))

    paths = []
    with torch.no_grad():
        for batch_idx in selected:
            batch = all_batches[batch_idx]
            imgs = batch["image"].to(device)  # (B,1,Z,H,W)
            masks = batch["mask"]  # CPU
            study_ids = batch["study_ids"]
            shapes = batch["shapes"]

            probs = model(imgs)  # (B,1,Z,H,W)

            for i in range(imgs.size(0)):
                sid = study_ids[i]
                orig_shape = shapes[i]  # (Z_orig, H_orig, W_orig)

                # Extraer y recortar al tamaño original (quitar padding)
                z, h, w = orig_shape
                img_np = imgs[i, 0, :z, :h, :w].cpu().numpy()
                gt_np = masks[i, 0, :z, :h, :w].numpy()
                pred_probs_np = probs[i, 0, :z, :h, :w].cpu().numpy()
                pred_bin_np = (pred_probs_np >= threshold).astype(np.float32)

                # Normalizar imagen para display
                vmin, vmax = img_np.min(), img_np.max()
                if vmax > vmin:
                    img_display = (img_np - vmin) / (vmax - vmin)
                else:
                    img_display = img_np

                p = save_sample_png(
                    img=img_display,
                    gt_mask=gt_np,
                    pred_mask=pred_bin_np,
                    study_id=sid,
                    out_dir=out_dir,
                    pred_probs=pred_probs_np,
                )
                paths.append(p)

                if len(paths) >= n_samples:
                    break
            if len(paths) >= n_samples:
                break

    print(f"[INFO] Generados {len(paths)} samples PNG en {out_dir}")
    return paths

