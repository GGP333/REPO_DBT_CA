# src/nnunet_train.py
"""
Funciones de entrenamiento específicas para nnUNet con soporte para deep supervision
"""
from __future__ import annotations
import os
import json
from typing import Dict, Tuple, List, Any, Optional, Union
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from metrics import dice_loss, compute_metrics


def deep_supervision_loss(outputs: List[torch.Tensor], 
                         target: torch.Tensor, 
                         weights: Optional[List[float]] = None) -> torch.Tensor:
    """
    Calcula la pérdida combinada para deep supervision.
    
    Args:
        outputs: Lista de salidas del modelo [main_output, ds_output1, ds_output2, ...]
        target: Máscara objetivo
        weights: Pesos para cada salida (por defecto: [0.5, 0.25, 0.125, ...])
    
    Returns:
        Pérdida combinada ponderada
    """
    if len(outputs) == 1:
        # Sin deep supervision, solo la salida principal
        return dice_loss(outputs[0], target)
    
    # Generar pesos por defecto si no se proporcionan
    if weights is None:
        weights = [0.5]  # Peso principal
        remaining_weight = 0.5
        for i in range(1, len(outputs)):
            weight = remaining_weight / (2 ** i)
            weights.append(weight)
    
    # Normalizar pesos para que sumen 1
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    # Calcular pérdida ponderada
    total_loss = 0.0
    for i, (output, weight) in enumerate(zip(outputs, weights)):
        loss = dice_loss(output, target)
        total_loss += weight * loss
    
    return total_loss


def train_one_epoch_nnunet(model: torch.nn.Module,
                          loader: DataLoader,
                          optimizer: torch.optim.Optimizer,
                          device: torch.device,
                          amp: bool = True,
                          deep_supervision: bool = False,
                          ds_weights: Optional[List[float]] = None) -> float:
    """
    Entrenamiento por una época con soporte para deep supervision.
    
    Args:
        model: Modelo nnUNet
        loader: DataLoader de entrenamiento
        optimizer: Optimizador
        device: Dispositivo (CPU/GPU)
        amp: Si usar Automatic Mixed Precision
        deep_supervision: Si usar deep supervision
        ds_weights: Pesos para deep supervision
    
    Returns:
        Pérdida promedio de la época
    """
    model.train()
    scaler = GradScaler(enabled=amp)
    epoch_loss = 0.0
    n = 0
    
    for batch in tqdm(loader, desc="Train", leave=False):
        imgs: torch.Tensor = batch["image"].to(device)  # (B,1,Z,H,W)
        masks: torch.Tensor = batch["mask"].to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast(enabled=amp):
            if deep_supervision and model.training:
                # Modelo con deep supervision retorna (main_output, [ds_outputs])
                outputs = model(imgs)
                if isinstance(outputs, tuple):
                    main_output, ds_outputs = outputs
                    all_outputs = [main_output] + ds_outputs
                else:
                    # Fallback si el modelo no tiene deep supervision habilitado
                    all_outputs = [outputs]
            else:
                # Modo normal
                output = model(imgs)
                all_outputs = [output]
            
            # Calcular pérdida
            loss = deep_supervision_loss(all_outputs, masks, ds_weights)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += float(loss.item()) * imgs.size(0)
        n += imgs.size(0)
    
    return epoch_loss / max(1, n)


@torch.no_grad()
def validate_one_epoch_nnunet(model: torch.nn.Module,
                             loader: DataLoader,
                             device: torch.device,
                             threshold: float,
                             preds_out_dir: str) -> Dict[str, float]:
    """
    Validación por una época (sin deep supervision en inferencia).
    
    Args:
        model: Modelo nnUNet
        loader: DataLoader de validación
        device: Dispositivo (CPU/GPU)
        threshold: Umbral para binarización
        preds_out_dir: Directorio para guardar predicciones
    
    Returns:
        Diccionario con métricas agregadas
    """
    model.eval()
    agg = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}
    n = 0
    
    os.makedirs(preds_out_dir, exist_ok=True)
    
    for batch in tqdm(loader, desc="Val", leave=False):
        imgs: torch.Tensor = batch["image"].to(device)  # (B,1,Z,H,W)
        masks: torch.Tensor = batch["mask"].to(device)
        
        # Forward pass (solo salida principal durante validación)
        outputs = model(imgs)
        if isinstance(outputs, tuple):
            # Si el modelo retorna tuple (main, ds), usar solo main
            probs = outputs[0]
        else:
            probs = outputs
        
        # Calcular métricas por batch
        for i in range(probs.size(0)):
            prob_i = probs[i]  # (1,Z,H,W)
            mask_i = masks[i]  # (1,Z,H,W)
            
            # Métricas
            metrics_i = compute_metrics(prob_i, mask_i, threshold)
            for k, v in metrics_i.items():
                agg[k] += v
            n += 1
            
            # Guardar predicciones
            study_id = batch["study_id"][i]
            prob_np = prob_i.squeeze(0).cpu().numpy()  # (Z,H,W)
            bin_np = (prob_np > threshold).astype("uint8")
            
            prob_path = os.path.join(preds_out_dir, f"{study_id}_probs.npy")
            bin_path = os.path.join(preds_out_dir, f"{study_id}_bin.npy")
            
            import numpy as np
            np.save(prob_path, prob_np.astype("float32"))
            np.save(bin_path, bin_np)
    
    # Promediar métricas
    for k in agg:
        agg[k] /= max(1, n)
    
    return agg


def fit_nnunet(model: torch.nn.Module,
              train_loader: DataLoader,
              val_loader: DataLoader,
              cfg,
              deep_supervision: bool = False,
              ds_weights: Optional[List[float]] = None,
              out_ckpt_dir: str = "outputs/checkpoints",
              out_logs_dir: str = "outputs/logs") -> Dict[str, Any]:
    """
    Entrenamiento completo de nnUNet con soporte para deep supervision.
    
    Args:
        model: Modelo nnUNet
        train_loader: DataLoader de entrenamiento
        val_loader: DataLoader de validación
        cfg: Configuración
        deep_supervision: Si usar deep supervision durante entrenamiento
        ds_weights: Pesos para deep supervision
        out_ckpt_dir: Directorio para checkpoints
        out_logs_dir: Directorio para logs
    
    Returns:
        Diccionario con historial de entrenamiento
    """
    from train_eval import get_device, _append_logs_csv_json
    
    device = get_device()
    model.to(device)
    
    # Configurar optimizador con learning rate específico para nnUNet
    lr = float(getattr(cfg, 'lr', 1e-3))
    weight_decay = float(getattr(cfg, 'weight_decay', 1e-5))
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Scheduler más agresivo para nnUNet
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode="max", 
        factor=0.5, 
        patience=7,  # Más paciente
        verbose=True,
        min_lr=1e-6
    )
    
    # Early Stopping
    es_enabled = bool(getattr(cfg, "early_stopping", True))
    es_patience = int(getattr(cfg, "early_stop_patience", 25))  # Más paciente para nnUNet
    es_min_delta = float(getattr(cfg, "early_stop_min_delta", 1e-4))
    es_counter = 0
    
    # Configurar deep supervision si está habilitado
    if deep_supervision:
        model.deep_supervision = True
        print(f"Deep supervision habilitado con pesos: {ds_weights}")
    
    os.makedirs(out_ckpt_dir, exist_ok=True)
    os.makedirs(out_logs_dir, exist_ok=True)
    best_path = os.path.join(out_ckpt_dir, "best_nnunet.pt")
    
    best_dice = -1.0
    history = []
    
    for epoch in range(1, int(cfg.max_epochs) + 1):
        # Entrenamiento
        tr_loss = train_one_epoch_nnunet(
            model, train_loader, optimizer, device, 
            amp=bool(cfg.amp), 
            deep_supervision=deep_supervision,
            ds_weights=ds_weights
        )
        
        # Validación
        val_metrics = validate_one_epoch_nnunet(
            model, val_loader, device, 
            threshold=float(cfg.threshold), 
            preds_out_dir="outputs/nnunet_preds_npy"
        )
        
        # Logs
        logs = {
            "train_loss": tr_loss, 
            **{f"val_{k}": v for k, v in val_metrics.items()}, 
            "lr": optimizer.param_groups[0]["lr"]
        }
        _append_logs_csv_json(out_logs_dir, epoch, logs)
        history.append({"epoch": epoch, **logs})
        
        # Print
        try:
            from tqdm import tqdm
            tqdm.write(f"Epoch {epoch:03d}/{int(cfg.max_epochs)} | "
                      f"train_loss={tr_loss:.4f} | "
                      f"val_dice={val_metrics['dice']:.4f} | val_iou={val_metrics['iou']:.4f} | "
                      f"val_f1={val_metrics['f1']:.4f} | val_acc={val_metrics['accuracy']:.4f} | "
                      f"lr={optimizer.param_groups[0]['lr']:.6f}")
        except Exception:
            print(f"Epoch {epoch:03d}/{int(cfg.max_epochs)} | train_loss={tr_loss:.4f} | "
                  f"val_dice={val_metrics['dice']:.4f} | val_iou={val_metrics['iou']:.4f} | "
                  f"val_f1={val_metrics['f1']:.4f} | val_acc={val_metrics['accuracy']:.4f} | "
                  f"lr={optimizer.param_groups[0]['lr']:.6f}")
        
        # Scheduler
        scheduler.step(val_metrics["dice"])
        
        # Guardado del mejor + Early Stopping
        improved = val_metrics["dice"] > best_dice + es_min_delta
        if improved:
            best_dice = val_metrics["dice"]
            torch.save({
                "model_state": model.state_dict(),
                "cfg": vars(cfg) if hasattr(cfg, "__dict__") else dict(cfg),
                "epoch": epoch,
                "best_val_dice": best_dice,
                "deep_supervision": deep_supervision,
                "ds_weights": ds_weights
            }, best_path)
            es_counter = 0
        else:
            if es_enabled:
                es_counter += 1
        
        if es_enabled and es_counter >= es_patience:
            print(f"Early stopping en época {epoch} (sin mejora por {es_patience} épocas)")
            break
    
    return {"best_val_dice": best_dice, "best_ckpt": best_path, "history": history}


@torch.no_grad()
def evaluate_nnunet_from_checkpoint(model_ctor, 
                                   ckpt_path: str, 
                                   val_loader: DataLoader, 
                                   threshold: float = 0.5) -> Dict[str, float]:
    """
    Carga un checkpoint de nnUNet y evalúa en validación.
    
    Args:
        model_ctor: Constructor del modelo
        ckpt_path: Ruta al checkpoint
        val_loader: DataLoader de validación
        threshold: Umbral para binarización
    
    Returns:
        Diccionario con métricas agregadas
    """
    from train_eval import get_device
    
    device = get_device()
    data = torch.load(ckpt_path, map_location=device)
    
    # Crear modelo
    model = model_ctor()
    model.load_state_dict(data["model_state"])
    model.to(device)
    model.eval()
    
    # Si el checkpoint tenía deep supervision, deshabilitarlo para evaluación
    if hasattr(model, 'deep_supervision'):
        model.deep_supervision = False
    
    # Evaluar
    metrics = validate_one_epoch_nnunet(
        model, val_loader, device, 
        threshold=threshold, 
        preds_out_dir="outputs/nnunet_preds_npy"
    )
    
    return metrics 