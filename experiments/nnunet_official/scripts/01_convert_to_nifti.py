"""
Convert preprocessed .npy DBT studies to NIfTI for nnUNetv2.

For each dataset (1..4) creates:
    nnUNet_raw/DatasetXXX_*/imagesTr/{case}_0000.nii.gz   (training+val)
    nnUNet_raw/DatasetXXX_*/labelsTr/{case}.nii.gz
    nnUNet_raw/DatasetXXX_*/imagesTs/{case}_0000.nii.gz   (test, held out)
    nnUNet_raw/DatasetXXX_*/labelsTs/{case}.nii.gz        (kept for our own eval)
    nnUNet_raw/DatasetXXX_*/dataset.json

The train/val/test partition replicates the repo's nnUNet split (see
02_make_splits.py) so the comparison vs. the repo's nnU-Net-like is honest.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import SimpleITK as sitk

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from datasets import DATASETS, dataset_folder  # noqa: E402
from splits import compute_split  # noqa: E402

SPACING_ZYX = (1.0, 1.0, 1.0)  # data already resampled


def list_studies(src_dir: Path) -> List[str]:
    return sorted(
        d.name for d in src_dir.iterdir()
        if d.is_dir() and (d.name.startswith("dbt_") or d.name.startswith("real_dbt_"))
    )


def load_pair(src_dir: Path, sid: str):
    img = np.load(src_dir / sid / f"{sid}_img.npy").astype(np.float32)
    mask = np.load(src_dir / sid / f"{sid}_mask.npy")
    mask = (mask > 0).astype(np.uint8)
    if img.shape != mask.shape:
        raise RuntimeError(f"shape mismatch {sid}: img={img.shape} mask={mask.shape}")
    return img, mask


def save_nifti(arr_zyx: np.ndarray, out_path: Path) -> None:
    img = sitk.GetImageFromArray(arr_zyx)
    img.SetSpacing(SPACING_ZYX[::-1])  # SimpleITK uses (x,y,z)
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(out_path), useCompression=True)


def write_dataset_json(out_dir: Path, n_training: int) -> None:
    info = {
        "channel_names": {"0": "noNorm"},  # already z-scored externally
        "labels": {"background": 0, "tumor": 1},
        "numTraining": n_training,
        "file_ending": ".nii.gz",
        "name": out_dir.name,
        "description": "DBT 3D tumor segmentation (single-channel volume already z-score normalized).",
    }
    with open(out_dir / "dataset.json", "w") as f:
        json.dump(info, f, indent=2)


def convert_dataset(dsid: int, nnunet_raw: Path, seed: int = 42) -> None:
    meta = DATASETS[dsid]
    src = meta["src"]
    if not src.is_dir():
        raise FileNotFoundError(f"Missing source dataset {src}")

    studies = list_studies(src)
    split = compute_split(studies, seed=seed)
    print(f"[{meta['name']}] total={len(studies)}  train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}")

    out_dir = nnunet_raw / dataset_folder(dsid)
    imgs_tr = out_dir / "imagesTr"
    lbls_tr = out_dir / "labelsTr"
    imgs_ts = out_dir / "imagesTs"
    lbls_ts = out_dir / "labelsTs"
    for d in [imgs_tr, lbls_tr, imgs_ts, lbls_ts]:
        d.mkdir(parents=True, exist_ok=True)

    n_train_total = 0
    for sid in studies:
        img, mask = load_pair(src, sid)
        if sid in split["test"]:
            save_nifti(img, imgs_ts / f"{sid}_0000.nii.gz")
            save_nifti(mask, lbls_ts / f"{sid}.nii.gz")
        else:
            save_nifti(img, imgs_tr / f"{sid}_0000.nii.gz")
            save_nifti(mask, lbls_tr / f"{sid}.nii.gz")
            n_train_total += 1
        print(f"  {sid}: ok")

    write_dataset_json(out_dir, n_train_total)
    print(f"[{meta['name']}] -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nnunet-raw", required=True, help="Path to nnUNet_raw root.")
    ap.add_argument("--datasets", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    nnunet_raw = Path(args.nnunet_raw).resolve()
    nnunet_raw.mkdir(parents=True, exist_ok=True)

    for dsid in args.datasets:
        convert_dataset(dsid, nnunet_raw, seed=args.seed)


if __name__ == "__main__":
    main()
