"""
Drop the experiment's custom trainers into nnUNetv2's auto-discovery path.

nnUNetv2 looks for trainer classes under
`nnunetv2/training/nnUNetTrainer/variants/` (recursively). We copy our
files there so that `nnUNetv2_train ... -tr <ClassName>` works.

Run once after installing/upgrading nnunetv2.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import nnunetv2.training.nnUNetTrainer.variants.training_length as _tl

TRAINER_FILES = [
    "nnUNetTrainer_PaperStandardized.py",
    "nnUNetTrainer_Seeded42.py",
    "nnUNetTrainer_Seeded42_ES.py",
]


def main():
    src_dir = Path(__file__).resolve().parent / "trainers"
    dst_dir = Path(_tl.__file__).resolve().parent
    for fname in TRAINER_FILES:
        src = src_dir / fname
        dst = dst_dir / fname
        if not src.is_file():
            print(f"[SKIP] missing source: {src}")
            continue
        shutil.copy2(src, dst)
        print(f"installed: {src.name} -> {dst}")


if __name__ == "__main__":
    main()
