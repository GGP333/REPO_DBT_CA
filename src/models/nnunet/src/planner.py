# src/planner.py
"""
Architecture Planner: toma el fingerprint del dataset y produce un plan
de arquitectura auto-configurado para nnU-Net 3D.

Decide automáticamente:
  - patch_size
  - strides por nivel (anisótropos)
  - canales por nivel
  - batch_size estimado
"""
from __future__ import annotations

import json
import math
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

from io_utils import get_logger


# ---------- Constantes por defecto (inspiradas en nnU-Net v2) ---------- #
MAX_CHANNELS = 320          # Cap de canales por nivel
DEFAULT_BASE_CH = 32        # Canales iniciales
MIN_FEATURE_SIZE = 4        # Mínimo de voxeles por eje en el bottleneck
OVERHEAD_FACTOR = 8.0       # Factor multiplicativo para estimar VRAM
MAX_LEVELS = 6              # Máximo de niveles en la U
MIN_LEVELS = 3              # Mínimo de niveles


def _max_pools_for_axis(dim: int, min_feat: int = MIN_FEATURE_SIZE) -> int:
    """Cuántos poolings de stride-2 se pueden hacer antes de caer bajo min_feat."""
    if dim <= min_feat:
        return 0
    return int(math.floor(math.log2(dim / min_feat)))


def _round_down_to_multiple(val: int, multiple: int) -> int:
    """Redondea hacia abajo al múltiplo más cercano."""
    return max(multiple, (val // multiple) * multiple)


def _estimate_vram_gb(
    patch_size: Tuple[int, int, int],
    channels_per_level: List[int],
    batch_size: int,
) -> float:
    """
    Estimación heurística del uso de VRAM en GB.
    
    Incluye: activaciones de encoder + decoder, parámetros del modelo, gradientes.
    Es una aproximación conservadora.
    """
    pz, py, px = patch_size
    total_voxels = pz * py * px
    max_ch = max(channels_per_level)

    # Activaciones: cada nivel tiene voxels/(2^level) * channels * sizeof(float32)
    # Sumamos encoder + decoder + gradientes ≈ factor 3
    act_bytes = 0
    for i, ch in enumerate(channels_per_level):
        scale = 2 ** i
        level_voxels = total_voxels / (scale ** 3) if i > 0 else total_voxels
        act_bytes += level_voxels * ch * 4  # float32

    # Multiplicar por 3 (fwd + bwd + optimizer states) y batch
    act_bytes *= 3.0 * batch_size

    # Parámetros del modelo (convs 3x3x3, ~27 * in_ch * out_ch por conv, 2 convs por nivel)
    param_bytes = 0
    for i in range(len(channels_per_level)):
        in_ch = 1 if i == 0 else channels_per_level[i - 1]
        out_ch = channels_per_level[i]
        param_bytes += 27 * in_ch * out_ch * 4 * 2  # 2 convs por bloque

    total_bytes = act_bytes + param_bytes * 3  # params + grads + optimizer
    return total_bytes / (1024 ** 3)


def generate_plan(
    fingerprint: Dict[str, Any],
    gpu_vram_gb: float = 16.0,
    base_ch: int = DEFAULT_BASE_CH,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Genera un plan de arquitectura a partir del fingerprint del dataset.

    Parameters
    ----------
    fingerprint : dict
        Salida de compute_fingerprint().
    gpu_vram_gb : float
        VRAM disponible de la GPU en GB.
    base_ch : int
        Canales base iniciales (se puede reducir si no cabe en VRAM).
    logger : Logger, optional

    Returns
    -------
    dict con claves:
        - patch_size        : [Pz, Py, Px]
        - strides           : [[sz,sy,sx], ...] por nivel de downsampling
        - channels          : [ch0, ch1, ch2, ...] por nivel
        - num_levels        : int (incluye bottleneck)
        - batch_size        : int
        - base_channels     : int
        - in_channels       : 1
        - num_classes       : 1
        - kernel_sizes      : [[kz,ky,kx], ...] por nivel
        - oversample_foreground : float
        - median_shape      : [Z, H, W]
    """
    logger = logger or get_logger("planner")

    median_shape = tuple(fingerprint["median_shape"])  # (Z, H, W)
    spacing = tuple(fingerprint.get("spacing", [1.0, 1.0, 1.0]))

    logger.info(f"Planificando para median_shape={median_shape}, VRAM={gpu_vram_gb}GB")

    # ---- 1. Calcular poolings máximos por eje ---- #
    max_pools_per_axis = [_max_pools_for_axis(d) for d in median_shape]
    logger.info(f"Max pools por eje (Z,Y,X): {max_pools_per_axis}")

    # Número total de niveles de downsampling = min de lo que permiten
    # los ejes más largos, pero considerando el eje corto
    # Usamos el máximo de los ejes Y,X y ajustamos Z independientemente
    n_downsamplings = min(MAX_LEVELS - 1, max(max_pools_per_axis))
    n_downsamplings = max(MIN_LEVELS - 1, n_downsamplings)

    # ---- 2. Strides anisótropos por nivel de downsampling ---- #
    # Para cada nivel, decidir qué ejes hacer pooling
    strides: List[List[int]] = []
    remaining_pools = list(max_pools_per_axis)  # pools restantes por eje

    for level in range(n_downsamplings):
        stride = [1, 1, 1]
        for axis in range(3):
            if remaining_pools[axis] > 0:
                stride[axis] = 2
                remaining_pools[axis] -= 1
        strides.append(stride)

    # Reordenar strides: empezar pooling por los ejes largos primero.
    # Si Z no puede hacer tantos pools como Y/X, postponer su pooling.
    # Estrategia: ordenar niveles para que los strides con Z=1 vayan primero
    # (pool solo Y,X), luego los que también hacen pool en Z.
    strides_z1 = [s for s in strides if s[0] == 1]  # pool solo Y,X
    strides_z2 = [s for s in strides if s[0] == 2]  # pool en Z también
    strides = strides_z1 + strides_z2

    num_levels = len(strides) + 1  # +1 para el nivel inicial (sin stride)

    logger.info(f"Strides por nivel de downsampling: {strides}")
    logger.info(f"Niveles totales (incl. inicial): {num_levels}")

    # ---- 3. Kernel sizes ---- #
    # Kernel 3x3x3 por defecto. Si el spacing es muy anisótropo,
    # se podría usar kernel 1x3x3 en los primeros niveles donde Z no se reduce.
    # Para datos ya resampleados a spacing uniforme, usamos 3x3x3 siempre.
    kernel_sizes = [[3, 3, 3]] * num_levels

    # ---- 4. Canales por nivel ---- #
    channels = []
    ch = base_ch
    for i in range(num_levels):
        channels.append(min(ch, MAX_CHANNELS))
        ch *= 2

    # ---- 5. Patch size ---- #
    # Empezar con el median shape y reducir si necesario
    patch_size = list(median_shape)

    # Ajustar cada dimensión al múltiplo de 2^(pools en ese eje)
    for axis in range(3):
        n_pools = sum(s[axis] == 2 for s in strides)
        divisor = 2 ** n_pools
        patch_size[axis] = _round_down_to_multiple(patch_size[axis], divisor)

    # ---- 6. Batch size y ajuste de VRAM ---- #
    batch_size = 1
    est_vram = _estimate_vram_gb(tuple(patch_size), channels, batch_size)

    # Si no cabe, reducir patch o base_ch
    max_vram = gpu_vram_gb * 0.70  # Usar máximo 70% de VRAM
    while est_vram > max_vram and base_ch >= 16:
        # Primero intentar reducir patch size
        reduced = False
        for axis in [0, 2, 1]:  # priorizar reducir Z, luego X, luego Y
            n_pools = sum(s[axis] == 2 for s in strides)
            divisor = 2 ** n_pools
            new_dim = patch_size[axis] - divisor
            if new_dim >= divisor * 2:  # no reducir demasiado
                patch_size[axis] = new_dim
                reduced = True
                break
        if not reduced:
            # Reducir canales base
            base_ch //= 2
            channels = []
            ch = base_ch
            for i in range(num_levels):
                channels.append(min(ch, MAX_CHANNELS))
                ch *= 2
        est_vram = _estimate_vram_gb(tuple(patch_size), channels, batch_size)

    # Intentar aumentar batch size si sobra VRAM
    while True:
        test_bs = batch_size + 1
        est = _estimate_vram_gb(tuple(patch_size), channels, test_bs)
        if est < max_vram:
            batch_size = test_bs
        else:
            break

    logger.info(f"Patch size final: {patch_size}")
    logger.info(f"Canales: {channels}")
    logger.info(f"Batch size estimado: {batch_size}")
    logger.info(f"VRAM estimada: {est_vram:.2f} GB / {gpu_vram_gb} GB")

    plan: Dict[str, Any] = {
        "patch_size": patch_size,
        "strides": strides,
        "channels": channels,
        "num_levels": num_levels,
        "batch_size": batch_size,
        "base_channels": base_ch,
        "in_channels": 1,
        "num_classes": 1,
        "kernel_sizes": kernel_sizes,
        "median_shape": list(median_shape),
        "spacing": list(spacing),
        "estimated_vram_gb": round(est_vram, 3),
        "gpu_vram_gb": gpu_vram_gb,
    }

    return plan


def save_plan(plan: Dict[str, Any], out_dir: str) -> str:
    """Guarda el plan como JSON. Devuelve la ruta al archivo."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    plan_path = out_path / "architecture_plan.json"
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)
    return str(plan_path)

