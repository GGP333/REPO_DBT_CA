# src/train_eval.py
from __future__ import annotations
import os
import json
from typing import Dict, Tuple, List, Any, Optional
import numpy as np
import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from metrics import dice_loss, bce_loss, dice_ce_loss, compute_metrics

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _get_loss_fn(loss_name: str):
    """Devuelve la función de pérdida según el nombre."""
    _loss_map = {"dice": dice_loss, "bce": bce_loss, "dice_ce": dice_ce_loss}
    if loss_name not in _loss_map:
        raise ValueError(f"Loss '{loss_name}' no reconocida. Opciones: {list(_loss_map.keys())}")
    return _loss_map[loss_name]


def train_one_epoch(model: torch.nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    amp: bool = True,
                    loss_name: str = "dice_ce") -> float:
    model.train()
    scaler = GradScaler("cuda", enabled=amp)
    loss_fn = _get_loss_fn(loss_name)
    epoch_loss = 0.0
    n = 0
    for batch in tqdm(loader, desc="Train", leave=False):
        imgs: torch.Tensor = batch["image"].to(device)  # (B,1,Z,H,W)
        masks: torch.Tensor = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=amp):
            probs = model(imgs)
            loss = loss_fn(probs, masks)
        scaler.scale(loss).backward()
        # Gradient clipping para evitar explosión de gradientes (inf/nan)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += float(loss.item()) * imgs.size(0)
        n += imgs.size(0)
    return epoch_loss / max(1, n)

@torch.no_grad()
def validate_one_epoch(model: torch.nn.Module,
                       loader: DataLoader,
                       device: torch.device,
                       threshold: float,
                       preds_out_dir: str,
                       plan: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """
    Validación por época.

    Si plan es proporcionado, usa sliding window inference (auto-configurado).
    Si plan es None, usa el forward directo (legacy / compatible con otros modelos).
    """
    model.eval()
    agg: Dict[str, float] = {}
    n = 0

    use_sw = plan is not None
    if use_sw:
        from patch_utils import sliding_window_inference
        patch_size = tuple(plan["patch_size"])
        overlap = float(plan.get("patch_overlap", 0.5))

    for batch in tqdm(loader, desc="Val", leave=False):
        imgs: torch.Tensor = batch["image"].to(device)  # (B,1,Z,H,W)
        masks: torch.Tensor = batch["mask"].to(device)
        study_ids = batch["study_ids"]
        shapes = batch["shapes"]

        if use_sw:
            # Sliding window inference: procesar volumen a volumen
            batch_probs = []
            for i in range(imgs.size(0)):
                vol = imgs[i]  # (1, Z, H, W)
                Z, H, W = shapes[i]
                # Recortar al shape real (sin padding)
                vol_real = vol[:, :Z, :H, :W]
                probs_i = sliding_window_inference(
                    model, vol_real, patch_size,
                    overlap=overlap, device=device, amp=True,
                )
                # Re-pad para que coincida con mask paddeada
                if probs_i.shape[2:] != masks.shape[2:]:
                    pad_z = masks.size(2) - probs_i.size(2)
                    pad_y = masks.size(3) - probs_i.size(3)
                    pad_x = masks.size(4) - probs_i.size(4)
                    probs_i = torch.nn.functional.pad(probs_i, (0, max(0, pad_x), 0, max(0, pad_y), 0, max(0, pad_z)))
                    probs_i = probs_i[:, :, :masks.size(2), :masks.size(3), :masks.size(4)]
                batch_probs.append(probs_i)
            probs = torch.cat(batch_probs, dim=0)
        else:
            # Forward directo (legacy)
            probs = model(imgs)  # (B,1,Z,H,W)

        # Métricas (usa TODAS las claves devueltas por compute_metrics)
        m = compute_metrics(probs, masks, threshold=threshold)
        if not agg:
            agg = {k: 0.0 for k in m}
        for k in agg:
            agg[k] += m.get(k, 0.0) * imgs.size(0)
        n += imgs.size(0)

        # Guardar predicciones por volumen (sobrescribe por estudio)
        probs_cpu = probs.detach().cpu()
        for i, sid in enumerate(study_ids):
            Z, H, W = shapes[i]
            p = probs_cpu[i, 0, :Z, :H, :W].numpy()
            binp = (p >= threshold).astype("float32")
            os.makedirs(preds_out_dir, exist_ok=True)
            npy_prob = os.path.join(preds_out_dir, f"{sid}_probs.npy")
            npy_bin = os.path.join(preds_out_dir, f"{sid}_bin.npy")
            np.save(npy_prob, p)
            np.save(npy_bin, binp)

    if n == 0:
        return {k: 0.0 for k in agg}
    return {k: v / n for k, v in agg.items()}

def _append_logs_csv_json(out_logs_dir: str, epoch: int, logs: Dict[str, Any]) -> None:
    csv_path = os.path.join(out_logs_dir, "metrics.csv")
    jsonl_path = os.path.join(out_logs_dir, "metrics.jsonl")
    # CSV
    df = pd.DataFrame([{**{"epoch": epoch}, **logs}])
    if os.path.isfile(csv_path):
        df.to_csv(csv_path, mode="a", index=False, header=False)
    else:
        df.to_csv(csv_path, index=False)
    # JSONL
    with open(jsonl_path, "a") as f:
        f.write(json.dumps({"epoch": epoch, **logs}) + "\n")

def fit(model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg,
        out_ckpt_dir: str = "outputs/checkpoints",
        out_logs_dir: str = "outputs/logs",
        preds_out_dir: str = "outputs/preds_npy",
        plan: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Entrena y valida por época, guarda mejor checkpoint por val_dice.

    Parameters
    ----------
    plan : dict, optional
        Si proporcionado, usa sliding window inference en validación.
    """
    # Limpiar logs anteriores para evitar datos superpuestos de runs previos
    for _old in [os.path.join(out_logs_dir, "metrics.csv"),
                 os.path.join(out_logs_dir, "metrics.jsonl")]:
        if os.path.isfile(_old):
            os.remove(_old)

    device = get_device()
    model.to(device)
    max_epochs = int(cfg.max_epochs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    # Scheduler: poly (default) o plateau (legacy)
    sched_name = str(getattr(cfg, "scheduler", "poly")).lower()
    warmup_epochs = int(getattr(cfg, "warmup_epochs", 5))
    if sched_name == "poly":
        # Polynomial LR decay: lr * (1 - epoch/max_epochs)^0.9
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda ep: max(0.0, (1.0 - ep / max(1, max_epochs)) ** 0.9) if ep >= warmup_epochs else (ep + 1) / max(1, warmup_epochs)
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5, verbose=True)

    # Early Stopping (configurable)
    es_enabled = bool(getattr(cfg, "early_stopping", False))
    es_patience = int(getattr(cfg, "early_stop_patience", 50))
    es_min_delta = float(getattr(cfg, "early_stop_min_delta", 0.001))
    es_counter = 0

    best_dice = -1.0
    best_path = os.path.join(out_ckpt_dir, "best.pt")
    last_path = os.path.join(out_ckpt_dir, "last.pt")

    history = []
    for epoch in range(1, max_epochs+1):
        loss_name = str(getattr(cfg, "loss", "dice_ce")).lower()
        tr_loss = train_one_epoch(model, train_loader, optimizer, device, amp=bool(cfg.amp), loss_name=loss_name)
        val_metrics = validate_one_epoch(
            model,
            val_loader,
            device,
            threshold=float(cfg.threshold),
            preds_out_dir=preds_out_dir,
            plan=plan,
        )
        logs = {"train_loss": tr_loss, **{f"val_{k}": v for k,v in val_metrics.items()}, "lr": optimizer.param_groups[0]["lr"]}
        _append_logs_csv_json(out_logs_dir, epoch, logs)
        history.append({"epoch": epoch, **logs})
        # Print limpio por epoch (para notebook)
        try:
            from tqdm import tqdm
            tqdm.write(
                f"Epoch {epoch:03d}/{int(cfg.max_epochs)} | "
                f"train_loss={tr_loss:.4f} | "
                f"val_dice={val_metrics['dice']:.4f} | "
                f"val_iou={val_metrics['iou']:.4f} | "
                f"val_f1={val_metrics['f1']:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} | "
                f"val_hausdorff={val_metrics['hausdorff']:.4f} | "
                f"lr={optimizer.param_groups[0]['lr']:.6f}"
            )
        except Exception:
            print(
                f"Epoch {epoch:03d}/{int(cfg.max_epochs)} | "
                f"train_loss={tr_loss:.4f} | "
                f"val_dice={val_metrics['dice']:.4f} | "
                f"val_iou={val_metrics['iou']:.4f} | "
                f"val_f1={val_metrics['f1']:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} | "
                f"val_hausdorff={val_metrics['hausdorff']:.4f} | "
                f"lr={optimizer.param_groups[0]['lr']:.6f}"
            )
        # Scheduler step
        if sched_name == "poly":
            scheduler.step()
        else:
            scheduler.step(val_metrics["dice"])

        # Guardado del último checkpoint (siempre, para recuperación)
        # Incluir plan en el checkpoint para poder reconstruir el modelo
        ckpt_data = {
            "model_state": model.state_dict(),
            "cfg": vars(cfg) if hasattr(cfg, "__dict__") else dict(cfg),
            "epoch": epoch,
            "val_dice": val_metrics["dice"],
        }
        if plan is not None:
            ckpt_data["plan"] = plan
        torch.save(ckpt_data, last_path)

        # Guardado del mejor + Early Stopping accounting
        improved = val_metrics["dice"] > best_dice + es_min_delta
        if improved:
            best_dice = val_metrics["dice"]
            ckpt_best = {
                "model_state": model.state_dict(),
                "cfg": vars(cfg) if hasattr(cfg, "__dict__") else dict(cfg),
                "epoch": epoch,
                "best_val_dice": best_dice,
            }
            if plan is not None:
                ckpt_best["plan"] = plan
            torch.save(ckpt_best, best_path)
            es_counter = 0
        else:
            if es_enabled:
                es_counter += 1

        if es_enabled and es_counter >= es_patience:
            try:
                from tqdm import tqdm
                tqdm.write(f"Early stopping en epoch {epoch} (sin mejora por {es_patience} epochs, min_delta={es_min_delta}).")
            except Exception:
                print(f"Early stopping en epoch {epoch} (sin mejora por {es_patience} epochs, min_delta={es_min_delta}).")
            break

    return {"best_val_dice": best_dice, "best_ckpt": best_path, "last_ckpt": last_path, "history": history}

@torch.no_grad()
def evaluate_from_checkpoint(model_ctor,
                             ckpt_path: str,
                             val_loader: DataLoader,
                             threshold: float = 0.5,
                             preds_out_dir: str = "outputs/preds_npy",
                             plan: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """
    Carga pesos y evalúa en validación (métricas agregadas).
    Soporta sliding window si se proporciona plan.
    """
    device = get_device()
    data = torch.load(ckpt_path, map_location=device)

    # Si el checkpoint contiene plan y no se proporcionó uno explícito, usarlo
    if plan is None and "plan" in data:
        plan = data["plan"]

    model = model_ctor()
    model.load_state_dict(data["model_state"])
    model.to(device)
    model.eval()
    metrics = validate_one_epoch(
        model,
        val_loader,
        device,
        threshold=threshold,
        preds_out_dir=preds_out_dir,
        plan=plan,
    )
    return metrics
