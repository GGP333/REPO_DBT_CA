# src/patch_utils.py
"""
Utilidades de parches para nnU-Net auto-configurado:
  - Extracción de patches aleatorios con oversampling de foreground
  - Sliding window inference con ponderación Gaussiana
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


# ===================== Extracción de patches ===================== #

def extract_patch(
    volume: np.ndarray,
    mask: np.ndarray,
    patch_size: Tuple[int, int, int],
    force_fg: bool = False,
    fg_coords: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extrae un patch aleatorio de (volume, mask).

    Parameters
    ----------
    volume : ndarray (Z, H, W)
    mask : ndarray (Z, H, W)
    patch_size : (Pz, Py, Px)
    force_fg : bool
        Si True, centra el patch en un voxel de foreground.
    fg_coords : ndarray (N, 3), optional
        Coordenadas de voxeles foreground pre-calculadas.

    Returns
    -------
    patch_img, patch_mask : ambos de shape patch_size
    """
    vol_shape = np.array(volume.shape)
    ps = np.array(patch_size)

    # Si el volumen es más pequeño que el patch en algún eje, pad
    pad_needed = np.maximum(ps - vol_shape, 0)
    if pad_needed.any():
        pad_before = pad_needed // 2
        pad_after = pad_needed - pad_before
        padding = list(zip(pad_before, pad_after))
        volume = np.pad(volume, padding, mode="constant", constant_values=0)
        mask = np.pad(mask, padding, mode="constant", constant_values=0)
        vol_shape = np.array(volume.shape)

    # Rango válido para la esquina superior-izquierda del patch
    max_start = vol_shape - ps  # (3,)

    if force_fg and fg_coords is not None and len(fg_coords) > 0:
        # Elegir un voxel de foreground aleatorio y centrar el patch ahí
        idx = np.random.randint(len(fg_coords))
        center = fg_coords[idx]

        # Ajustar si el volumen fue paddeado
        if pad_needed.any():
            center = center + pad_needed // 2

        start = np.clip(center - ps // 2, 0, max_start)
    else:
        # Patch completamente aleatorio
        start = np.array([np.random.randint(0, m + 1) for m in max_start])

    start = start.astype(int)
    end = start + ps

    patch_img = volume[start[0]:end[0], start[1]:end[1], start[2]:end[2]].copy()
    patch_mask = mask[start[0]:end[0], start[1]:end[1], start[2]:end[2]].copy()

    return patch_img, patch_mask


def precompute_fg_coords(mask_path: str) -> np.ndarray:
    """
    Carga la máscara y devuelve las coordenadas (N, 3) de los voxeles foreground.
    Si no hay foreground, devuelve array vacío (0, 3).
    """
    mask = np.load(mask_path)
    if mask.ndim == 4:
        mask = mask.squeeze()
    coords = np.argwhere(mask > 0)
    return coords.astype(np.int32)


# ===================== Sliding Window Inference ===================== #

def _get_gaussian_importance_map(
    patch_size: Tuple[int, int, int],
    sigma_scale: float = 0.125,
) -> torch.Tensor:
    """
    Genera un mapa de importancia Gaussiano 3D.
    Los voxeles centrales pesan más que los bordes.
    """
    tmp = np.zeros(patch_size, dtype=np.float32)
    center = [s // 2 for s in patch_size]
    sigmas = [s * sigma_scale for s in patch_size]

    # Crear grid de distancias al centro
    for z in range(patch_size[0]):
        for y in range(patch_size[1]):
            for x in range(patch_size[2]):
                dist_sq = (
                    ((z - center[0]) / max(sigmas[0], 1)) ** 2
                    + ((y - center[1]) / max(sigmas[1], 1)) ** 2
                    + ((x - center[2]) / max(sigmas[2], 1)) ** 2
                )
                tmp[z, y, x] = math.exp(-0.5 * dist_sq)

    # Normalizar para que el máximo sea 1
    tmp = tmp / (tmp.max() + 1e-8)
    tmp = np.clip(tmp, 1e-8, 1.0)  # evitar ceros

    return torch.from_numpy(tmp).float()


def _compute_steps(image_size: int, patch_size: int, overlap: float) -> List[int]:
    """Calcula las posiciones de inicio para tiles a lo largo de un eje."""
    if image_size <= patch_size:
        return [0]

    step = max(1, int(patch_size * (1.0 - overlap)))
    steps = list(range(0, image_size - patch_size + 1, step))

    # Asegurar que el último tile cubra el borde
    if steps[-1] + patch_size < image_size:
        steps.append(image_size - patch_size)

    return steps


@torch.no_grad()
def sliding_window_inference(
    model: torch.nn.Module,
    volume: torch.Tensor,
    patch_size: Tuple[int, int, int],
    overlap: float = 0.5,
    device: Optional[torch.device] = None,
    use_gaussian: bool = True,
    amp: bool = True,
) -> torch.Tensor:
    """
    Inferencia por sliding window con ponderación Gaussiana.

    Parameters
    ----------
    model : nn.Module (en modo eval)
    volume : Tensor (1, 1, Z, H, W) o (Z, H, W).
        Si es 3D se expande a (1, 1, Z, H, W).
    patch_size : (Pz, Py, Px)
    overlap : float [0, 1)
        Fracción de overlap entre patches.
    device : torch.device
    use_gaussian : bool
        Ponderar con mapa Gaussiano.
    amp : bool
        Usar mixed precision.

    Returns
    -------
    probs : Tensor (1, 1, Z, H, W) con probabilidades [0,1]
    """
    if device is None:
        device = next(model.parameters()).device

    # Normalizar input
    if volume.ndim == 3:
        volume = volume.unsqueeze(0).unsqueeze(0)  # (1,1,Z,H,W)
    elif volume.ndim == 4:
        volume = volume.unsqueeze(0)  # (1,1,Z,H,W)

    _, _, vz, vy, vx = volume.shape
    pz, py, px = patch_size

    # Pad si el volumen es menor que el patch en algún eje
    pad_z = max(0, pz - vz)
    pad_y = max(0, py - vy)
    pad_x = max(0, px - vx)
    if pad_z > 0 or pad_y > 0 or pad_x > 0:
        volume = F.pad(volume, (0, pad_x, 0, pad_y, 0, pad_z))
        _, _, vz, vy, vx = volume.shape

    # Mapa de importancia Gaussiano
    if use_gaussian:
        importance = _get_gaussian_importance_map(patch_size).to(device)
    else:
        importance = torch.ones(patch_size, device=device)

    # Acumuladores
    aggregated = torch.zeros(1, 1, vz, vy, vx, device=device)
    weight_map = torch.zeros(1, 1, vz, vy, vx, device=device)

    # Posiciones de inicio por eje
    steps_z = _compute_steps(vz, pz, overlap)
    steps_y = _compute_steps(vy, py, overlap)
    steps_x = _compute_steps(vx, px, overlap)

    volume = volume.to(device)

    for sz in steps_z:
        for sy in steps_y:
            for sx in steps_x:
                patch = volume[:, :, sz:sz+pz, sy:sy+py, sx:sx+px]

                with torch.cuda.amp.autocast(enabled=amp):
                    pred = model(patch)

                # Si el modelo devuelve tuple (deep supervision), tomar solo la principal
                if isinstance(pred, tuple):
                    pred = pred[0]

                aggregated[:, :, sz:sz+pz, sy:sy+py, sx:sx+px] += pred * importance
                weight_map[:, :, sz:sz+pz, sy:sy+py, sx:sx+px] += importance

    # Normalizar
    probs = aggregated / (weight_map + 1e-8)

    # Quitar padding
    orig_z = vz - pad_z
    orig_y = vy - pad_y
    orig_x = vx - pad_x
    probs = probs[:, :, :orig_z, :orig_y, :orig_x]

    return probs

