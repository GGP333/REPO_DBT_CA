#!/usr/bin/env python3
"""
Entrenamiento UNet 3D (U_Net BCE) con pérdida Dice.
Limita epochs via CLI; out_dir configurable.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from config_loader import load_config
from dataset_dbt import make_dataloaders, set_global_seeds
from io_utils import get_logger
from train_eval import fit, evaluate_from_checkpoint
from unet3d import UNet3D


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--out-dir", default="outputs")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.max_epochs: cfg.max_epochs = args.max_epochs
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.num_workers: cfg.num_workers = args.num_workers
    if args.lr: cfg.lr = args.lr
    if args.weight_decay: cfg.weight_decay = args.weight_decay
    if args.threshold: cfg.threshold = args.threshold
    cfg.loss = "dice"

    out_root = Path(args.out_dir)
    ckpt_dir = out_root / "checkpoints"
    logs_dir = out_root / "logs"
    preds_dir = out_root / "preds_npy"
    for d in [ckpt_dir, logs_dir, preds_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger = get_logger("train_unet_dice")
    set_global_seeds(int(cfg.seed))

    train_loader, val_loader, _, _ = make_dataloaders(cfg)

    model = UNet3D(in_ch=1, base_ch=32, levels=4)
    logger.info(f"Parámetros del modelo: {sum(p.numel() for p in model.parameters())/1e6:.3f}M")

    history = fit(model, train_loader, val_loader, cfg, out_ckpt_dir=str(ckpt_dir), out_logs_dir=str(logs_dir))
    best_ckpt = history["best_ckpt"]

    def ctor():
        return UNet3D(in_ch=1, base_ch=32, levels=4)

    val_metrics = evaluate_from_checkpoint(ctor, best_ckpt, val_loader, threshold=float(cfg.threshold))
    with open(logs_dir / "val_metrics.json", "w") as f:
        json.dump(val_metrics, f, indent=2)
    logger.info(f"Val metrics: {val_metrics}")


if __name__ == "__main__":
    main()

