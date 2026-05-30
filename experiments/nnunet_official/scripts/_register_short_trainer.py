"""
Creates a short-epoch trainer subclass on the fly for quick smoke tests.

Usage:
    python _register_short_trainer.py --epochs 5

This drops a small .py inside nnunetv2's user-extensible trainer dir so that
`nnUNetv2_train ... -tr nnUNetTrainer_<EPOCHS>epochs` works.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nnunetv2.training.nnUNetTrainer.variants.training_length as _tl  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, required=True)
    args = ap.parse_args()

    target_dir = Path(_tl.__file__).resolve().parent
    target = target_dir / f"nnUNetTrainer_{args.epochs}epochs.py"
    if target.exists():
        print(f"Already exists: {target}")
        return

    code = f'''from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_{args.epochs}epochs(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=None):
        if device is None:
            import torch
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.num_epochs = {args.epochs}
'''
    target.write_text(code)
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
