# src/metrics.py
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import torch

try:
    from scipy.spatial.distance import directed_hausdorff
    from scipy.ndimage import binary_erosion
    _SCIPY_AVAILABLE = True
except Exception:
    directed_hausdorff = None
    binary_erosion = None
    _SCIPY_AVAILABLE = False

def _align_spatial(a: torch.Tensor, b: torch.Tensor):
    """Recorta ambos tensores al mínimo común en (Z,H,W) si difieren."""
    if a.shape[-3:] == b.shape[-3:]:
        return a, b
    Z = min(a.size(-3), b.size(-3))
    H = min(a.size(-2), b.size(-2))
    W = min(a.size(-1), b.size(-1))
    return a[..., :Z, :H, :W], b[..., :Z, :H, :W]

def _to_bin(y_prob: torch.Tensor, threshold: float) -> torch.Tensor:
    return (y_prob >= threshold).to(y_prob.dtype)


def _surface_points(mask: np.ndarray) -> np.ndarray:
    """
    Extrae puntos de superficie para reducir el cómputo de Hausdorff.
    """
    if not mask.any():
        return np.empty((0, 3), dtype=np.float32)
    eroded = binary_erosion(mask)
    surface = mask ^ eroded
    coords = np.argwhere(surface)
    if coords.size == 0:  # fallback: usa todos los positivos
        coords = np.argwhere(mask)
    return coords.astype(np.float32)

def dice(y_prob: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    y_prob, y_true = _align_spatial(y_prob, y_true)
    y_pred = _to_bin(y_prob, threshold)
    inter = (y_pred * y_true).sum(dim=[1,2,3,4])
    denom = y_pred.sum(dim=[1,2,3,4]) + y_true.sum(dim=[1,2,3,4])
    d = (2 * inter + eps) / (denom + eps)
    return d.mean()

def iou(y_prob: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    y_prob, y_true = _align_spatial(y_prob, y_true)
    y_pred = _to_bin(y_prob, threshold)
    inter = (y_pred * y_true).sum(dim=[1,2,3,4])
    union = y_pred.sum(dim=[1,2,3,4]) + y_true.sum(dim=[1,2,3,4]) - inter
    j = (inter + eps) / (union + eps)
    return j.mean()

def precision(y_prob: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    y_prob, y_true = _align_spatial(y_prob, y_true)
    y_pred = _to_bin(y_prob, threshold)
    tp = (y_pred * y_true).sum(dim=[1,2,3,4])
    fp = (y_pred * (1 - y_true)).sum(dim=[1,2,3,4])
    p = (tp + eps) / (tp + fp + eps)
    return p.mean()

def recall(y_prob: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    y_prob, y_true = _align_spatial(y_prob, y_true)
    y_pred = _to_bin(y_prob, threshold)
    tp = (y_pred * y_true).sum(dim=[1,2,3,4])
    fn = ((1 - y_pred) * y_true).sum(dim=[1,2,3,4])
    r = (tp + eps) / (tp + fn + eps)
    return r.mean()

def f1(y_prob: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    p = precision(y_prob, y_true, threshold, eps)
    r = recall(y_prob, y_true, threshold, eps)
    return 2 * p * r / (p + r + eps)

def accuracy(y_prob: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    y_prob, y_true = _align_spatial(y_prob, y_true)
    y_pred = _to_bin(y_prob, threshold)
    correct = (y_pred == y_true).float().mean(dim=[1,2,3,4])
    return correct.mean()


def _compute_surface_distances(pred_pts: np.ndarray, true_pts: np.ndarray, 
                                max_points: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula distancias de cada punto de superficie a la otra superficie.
    Usa sampling aleatorio si hay demasiados puntos para evitar OOM.
    
    Args:
        pred_pts: Puntos de superficie de predicción (N, 3)
        true_pts: Puntos de superficie de GT (M, 3)
        max_points: Máximo número de puntos a usar (para evitar matrices NxM muy grandes)
    """
    from scipy.spatial.distance import cdist
    if pred_pts.size == 0 or true_pts.size == 0:
        return np.array([]), np.array([])
    
    # Sampling si hay demasiados puntos (evita matrices NxM gigantes)
    if pred_pts.shape[0] > max_points:
        idx = np.random.choice(pred_pts.shape[0], max_points, replace=False)
        pred_pts = pred_pts[idx]
    if true_pts.shape[0] > max_points:
        idx = np.random.choice(true_pts.shape[0], max_points, replace=False)
        true_pts = true_pts[idx]
    
    d_pred_to_true = cdist(pred_pts, true_pts).min(axis=1)
    d_true_to_pred = cdist(true_pts, pred_pts).min(axis=1)
    return d_pred_to_true, d_true_to_pred


def hausdorff_distance(y_prob: torch.Tensor,
                       y_true: torch.Tensor,
                       threshold: float = 0.5,
                       spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> torch.Tensor:
    """
    Hausdorff Distance 3D (bidireccional) calculado sobre superficies binarizadas.
    Devuelve promedio sobre el batch. Si SciPy no está disponible, devuelve NaN.
    """
    if not _SCIPY_AVAILABLE:
        return torch.tensor(float("nan"), device=y_prob.device)

    y_prob, y_true = _align_spatial(y_prob, y_true)
    y_pred_np = _to_bin(y_prob, threshold).detach().cpu().numpy().astype(bool)
    y_true_np = y_true.detach().cpu().numpy().astype(bool)
    spacing_np = np.asarray(spacing, dtype=np.float32)

    # Distancia diagonal del volumen como fallback cuando faltan positivos
    vol_shape = np.array(y_true_np.shape[-3:], dtype=np.float32)
    diag = float(np.linalg.norm(vol_shape * spacing_np))

    hd_vals = []
    for b in range(y_true_np.shape[0]):
        pred = y_pred_np[b, 0]
        true = y_true_np[b, 0]

        if not pred.any() and not true.any():
            hd_vals.append(0.0)
            continue
        if not pred.any() or not true.any():
            hd_vals.append(diag)
            continue

        pred_pts = _surface_points(pred) * spacing_np
        true_pts = _surface_points(true) * spacing_np

        if pred_pts.size == 0 or true_pts.size == 0:
            hd_vals.append(diag)
            continue

        d1 = directed_hausdorff(pred_pts, true_pts)[0]
        d2 = directed_hausdorff(true_pts, pred_pts)[0]
        hd_vals.append(float(max(d1, d2)))

    if not hd_vals:
        return torch.tensor(float("nan"), device=y_prob.device)
    return torch.tensor(hd_vals, device=y_prob.device).mean()


def hausdorff_distance_95(y_prob: torch.Tensor,
                          y_true: torch.Tensor,
                          threshold: float = 0.5,
                          spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> torch.Tensor:
    """
    Hausdorff Distance 95 (percentil 95) - más robusto a outliers que HD máximo.
    HD95 = percentil 95 de las distancias de superficie combinadas.
    """
    if not _SCIPY_AVAILABLE:
        return torch.tensor(float("nan"), device=y_prob.device)

    y_prob, y_true = _align_spatial(y_prob, y_true)
    y_pred_np = _to_bin(y_prob, threshold).detach().cpu().numpy().astype(bool)
    y_true_np = y_true.detach().cpu().numpy().astype(bool)
    spacing_np = np.asarray(spacing, dtype=np.float32)

    vol_shape = np.array(y_true_np.shape[-3:], dtype=np.float32)
    diag = float(np.linalg.norm(vol_shape * spacing_np))

    hd95_vals = []
    for b in range(y_true_np.shape[0]):
        pred = y_pred_np[b, 0]
        true = y_true_np[b, 0]

        if not pred.any() and not true.any():
            hd95_vals.append(0.0)
            continue
        if not pred.any() or not true.any():
            hd95_vals.append(diag)
            continue

        pred_pts = _surface_points(pred) * spacing_np
        true_pts = _surface_points(true) * spacing_np

        if pred_pts.size == 0 or true_pts.size == 0:
            hd95_vals.append(diag)
            continue

        d_pred_to_true, d_true_to_pred = _compute_surface_distances(pred_pts, true_pts)
        all_distances = np.concatenate([d_pred_to_true, d_true_to_pred])
        hd95_vals.append(float(np.percentile(all_distances, 95)))

    if not hd95_vals:
        return torch.tensor(float("nan"), device=y_prob.device)
    return torch.tensor(hd95_vals, device=y_prob.device).mean()


def fp_per_volume(y_prob: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    FP/Volume: Promedio de falsos positivos por volumen.
    Cuenta regiones conectadas FP (predicciones que no solapan con GT).
    """
    y_prob, y_true = _align_spatial(y_prob, y_true)
    y_pred = _to_bin(y_prob, threshold)
    # FP = predicción positiva donde GT es negativo
    fp_mask = (y_pred * (1 - y_true))
    # Contar voxeles FP por batch
    fp_counts = fp_mask.sum(dim=[1, 2, 3, 4]).float()
    # Normalizar por volumen total (proporción)
    total_voxels = torch.tensor(y_pred[0].numel(), dtype=torch.float32, device=y_pred.device)
    fp_per_vol = fp_counts / total_voxels
    return fp_per_vol.mean()


def fp_per_image(y_prob: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    FP/Images: Promedio de falsos positivos por slice/imagen 2D.
    Para volúmenes 3D, calcula FP por slice axial y promedia.
    """
    y_prob, y_true = _align_spatial(y_prob, y_true)
    y_pred = _to_bin(y_prob, threshold)
    # FP por slice (dimension Z = -3)
    fp_mask = (y_pred * (1 - y_true))
    # [B, C, Z, H, W] -> suma sobre H, W para cada slice Z
    fp_per_slice = fp_mask.sum(dim=[-2, -1]).float()  # [B, C, Z]
    # Normalizar por área del slice
    slice_area = y_pred.shape[-2] * y_pred.shape[-1]
    fp_per_slice = fp_per_slice / slice_area
    # Promedio sobre todos los slices y batch
    return fp_per_slice.mean()


def average_precision(y_prob: torch.Tensor, y_true: torch.Tensor,
                      prob_thresholds: Tuple[float, ...] = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
                                                            0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)) -> torch.Tensor:
    """
    Average Precision (AP) para segmentación binaria.
    Calcula el área bajo la curva Precision-Recall variando el umbral de probabilidad.
    
    AP = ∫₀¹ P(R) dR ≈ Σ (ΔR) · P_interp
    
    Donde P_interp usa la envolvente monotónica (interpolación por la derecha).
    """
    y_prob, y_true = _align_spatial(y_prob, y_true)
    
    precisions = []
    recalls = []
    
    # Ordenar umbrales de mayor a menor (más estricto primero)
    for t in sorted(prob_thresholds, reverse=True):
        y_pred = _to_bin(y_prob, t)
        tp = (y_pred * y_true).sum().float()
        fp = (y_pred * (1 - y_true)).sum().float()
        fn = ((1 - y_pred) * y_true).sum().float()
        
        prec = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
        
        precisions.append(prec.item())
        recalls.append(rec.item())
    
    # Interpolar precision (envolvente monotónica: P_interp(k) = max_{j>=k} P(j))
    precisions_interp = []
    max_prec = 0.0
    for p in reversed(precisions):
        max_prec = max(max_prec, p)
        precisions_interp.insert(0, max_prec)
    
    # Calcular AP como suma de (delta_recall * precision_interp)
    ap = 0.0
    prev_recall = 0.0
    for prec, rec in zip(precisions_interp, recalls):
        delta_recall = rec - prev_recall
        if delta_recall > 0:
            ap += delta_recall * prec
        prev_recall = rec
    
    return torch.tensor(ap, device=y_prob.device)


def mean_average_precision(y_prob: torch.Tensor, y_true: torch.Tensor,
                           iou_thresholds: Tuple[float, ...] = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)) -> torch.Tensor:
    """
    Mean Average Precision (mAP) estilo COCO para segmentación.
    Promedio de AP sobre múltiples umbrales de IoU.
    
    mAP = (1/|T|) Σ_{τ∈T} AP@IoU=τ
    
    Donde T = {0.5, 0.55, ..., 0.95} (umbrales de IoU estilo COCO).
    Para cada umbral τ, se considera TP si IoU(pred, GT) >= τ.
    """
    y_prob, y_true = _align_spatial(y_prob, y_true)
    
    ap_values = []
    
    for iou_thresh in iou_thresholds:
        # Para cada umbral de IoU, calcular si es TP o FP
        # En segmentación voxel-wise, usamos el IoU global como criterio
        prob_thresholds = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        
        precisions = []
        recalls = []
        
        for t in sorted(prob_thresholds, reverse=True):
            y_pred = _to_bin(y_prob, t)
            
            # Calcular IoU
            inter = (y_pred * y_true).sum().float()
            union = y_pred.sum() + y_true.sum() - inter
            current_iou = inter / (union + 1e-6)
            
            # TP solo si IoU >= umbral
            if current_iou >= iou_thresh:
                tp = (y_pred * y_true).sum().float()
                fp = (y_pred * (1 - y_true)).sum().float()
                fn = ((1 - y_pred) * y_true).sum().float()
            else:
                # Si IoU < umbral, toda la predicción es FP
                tp = torch.tensor(0.0, device=y_prob.device)
                fp = y_pred.sum().float()
                fn = y_true.sum().float()
            
            prec = tp / (tp + fp + 1e-6)
            rec = tp / (tp + fn + 1e-6)
            
            precisions.append(prec.item())
            recalls.append(rec.item())
        
        # Interpolar y calcular AP para este umbral de IoU
        precisions_interp = []
        max_prec = 0.0
        for p in reversed(precisions):
            max_prec = max(max_prec, p)
            precisions_interp.insert(0, max_prec)
        
        ap = 0.0
        prev_recall = 0.0
        for prec, rec in zip(precisions_interp, recalls):
            delta_recall = rec - prev_recall
            if delta_recall > 0:
                ap += delta_recall * prec
            prev_recall = rec
        
        ap_values.append(ap)
    
    # mAP = promedio sobre todos los umbrales de IoU
    mAP = sum(ap_values) / len(ap_values) if ap_values else 0.0
    return torch.tensor(mAP, device=y_prob.device)

def dice_loss(y_prob: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Dice Loss suave sobre probabilidades (no binariza).
    Alinea espacialmente si difieren shapes por padding/recortes.
    Fuerza float32 para estabilidad con AMP.
    """
    y_prob, y_true = _align_spatial(y_prob, y_true)
    y_prob = y_prob.float()
    y_true = y_true.float()
    inter = (y_prob * y_true).sum(dim=[1,2,3,4])
    denom = y_prob.sum(dim=[1,2,3,4]) + y_true.sum(dim=[1,2,3,4])
    d = (2 * inter + eps) / (denom + eps)
    return 1.0 - d.mean()


def bce_loss(y_prob: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Binary Cross Entropy Loss numéricamente estable y compatible con AMP autocast.
    Convierte probabilidades a logits y usa binary_cross_entropy_with_logits.
    """
    y_prob, y_true = _align_spatial(y_prob, y_true)
    y_prob = y_prob.float()
    y_true = y_true.float()
    eps = 1e-6
    y_prob = torch.clamp(y_prob, eps, 1.0 - eps)
    logits = torch.log(y_prob / (1.0 - y_prob))
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, y_true, reduction="mean")


def dice_ce_loss(y_prob: torch.Tensor, y_true: torch.Tensor, dice_weight: float = 0.5, ce_weight: float = 0.5) -> torch.Tensor:
    """
    Loss combinada: Dice + Binary Cross Entropy.
    Estabiliza el entrenamiento con extreme class imbalance.
    """
    return dice_weight * dice_loss(y_prob, y_true) + ce_weight * bce_loss(y_prob, y_true)

def compute_metrics(y_prob: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """Calcula todas las métricas de segmentación."""
    hd = hausdorff_distance(y_prob, y_true, threshold)
    hd95 = hausdorff_distance_95(y_prob, y_true, threshold)
    fp_vol = fp_per_volume(y_prob, y_true, threshold)
    fp_img = fp_per_image(y_prob, y_true, threshold)
    ap = average_precision(y_prob, y_true)
    mAP = mean_average_precision(y_prob, y_true)
    
    return {
        "dice": float(dice(y_prob, y_true, threshold).item()),
        "iou": float(iou(y_prob, y_true, threshold).item()),
        "precision": float(precision(y_prob, y_true, threshold).item()),
        "recall": float(recall(y_prob, y_true, threshold).item()),
        "f1": float(f1(y_prob, y_true, threshold).item()),
        "accuracy": float(accuracy(y_prob, y_true, threshold).item()),
        "hausdorff": float(hd.item()),
        "hausdorff_95": float(hd95.item()),
        "fp_per_volume": float(fp_vol.item()),
        "fp_per_image": float(fp_img.item()),
        "ap": float(ap.item()),
        "mAP": float(mAP.item()),
    }
