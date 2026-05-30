"""
Augmentaciones 3D on-the-fly para volúmenes DBT.
Aplica transformaciones espaciales idénticamente a imagen y máscara,
y transformaciones de intensidad solo a la imagen.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple


def random_flip(img: np.ndarray, mask: np.ndarray, axes: Tuple[int, ...] = (1, 2), prob: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Flip aleatorio en los ejes indicados (0=Z, 1=Y, 2=X)."""
    for ax in axes:
        if np.random.rand() < prob:
            img = np.flip(img, axis=ax).copy()
            mask = np.flip(mask, axis=ax).copy()
    return img, mask


def random_rotate90(img: np.ndarray, mask: np.ndarray, prob: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Rotación 90° aleatoria en el plano XY (ejes 1,2)."""
    if np.random.rand() < prob:
        k = np.random.choice([1, 2, 3])
        img = np.rot90(img, k=k, axes=(1, 2)).copy()
        mask = np.rot90(mask, k=k, axes=(1, 2)).copy()
    return img, mask


def gaussian_noise(img: np.ndarray, std_range: Tuple[float, float] = (0.0, 0.1), prob: float = 0.15) -> np.ndarray:
    """Ruido gaussiano aditivo (solo imagen)."""
    if np.random.rand() < prob:
        std = np.random.uniform(*std_range)
        noise = np.random.normal(0, std, img.shape).astype(np.float32)
        img = img + noise
    return img


def gaussian_blur(img: np.ndarray, sigma_range: Tuple[float, float] = (0.5, 1.5), prob: float = 0.2) -> np.ndarray:
    """Desenfoque gaussiano 3D ligero (solo imagen). Usa scipy si está disponible."""
    if np.random.rand() < prob:
        try:
            from scipy.ndimage import gaussian_filter
            sigma = np.random.uniform(*sigma_range)
            img = gaussian_filter(img, sigma=sigma).astype(np.float32)
        except ImportError:
            pass  # skip if scipy not available
    return img


def brightness_multiplicative(img: np.ndarray, factor_range: Tuple[float, float] = (0.75, 1.25), prob: float = 0.15) -> np.ndarray:
    """Ajuste de brillo multiplicativo (solo imagen)."""
    if np.random.rand() < prob:
        factor = np.random.uniform(*factor_range)
        img = img * factor
    return img


def gamma_correction(img: np.ndarray, gamma_range: Tuple[float, float] = (0.7, 1.5), prob: float = 0.3) -> np.ndarray:
    """Corrección gamma (solo imagen). Opera en rango [0,1]."""
    if np.random.rand() < prob:
        # Escalar a [0,1] si no lo está
        vmin, vmax = float(img.min()), float(img.max())
        rng = vmax - vmin
        if rng > 1e-8:
            img_01 = (img - vmin) / rng
            gamma = np.random.uniform(*gamma_range)
            img_01 = np.power(np.clip(img_01, 0, 1), gamma)
            img = img_01 * rng + vmin
    return img


def apply_augmentations(img: np.ndarray, mask: np.ndarray, cfg=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pipeline completo de augmentaciones.
    img:  (Z, H, W) float32
    mask: (Z, H, W) float32 {0, 1}
    cfg:  config object con flags aug_* opcionales
    """
    # Asegurar float32
    img = img.astype(np.float32, copy=True)
    mask = mask.astype(np.float32, copy=True)

    # Augmentaciones espaciales (imagen + máscara)
    img, mask = random_flip(img, mask, axes=(1, 2), prob=0.5)
    img, mask = random_rotate90(img, mask, prob=0.5)

    # Augmentaciones de intensidad (solo imagen)
    img = gaussian_noise(img, prob=0.15)
    img = gaussian_blur(img, prob=0.2)
    img = brightness_multiplicative(img, prob=0.15)
    img = gamma_correction(img, prob=0.3)

    return img, mask

