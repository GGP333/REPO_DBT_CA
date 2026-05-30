"""
Evaluate nnUNetv2 predictions on the held-out test set with the same 12
metrics the repo uses for the rest of the experiments.

Reuses `src/models/nnunet/src/metrics.py::compute_metrics` so the numbers are
1:1 comparable to the repo's reported results.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "models" / "nnunet" / "src"))
from metrics import compute_metrics  # noqa: E402


def load_mask(path: Path) -> np.ndarray:
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    return (arr > 0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    if not pred_files:
        raise RuntimeError(f"No predictions in {pred_dir}")

    per_case = {}
    accum = None
    n_metrics = 0

    for p in pred_files:
        sid = p.name.replace(".nii.gz", "")
        gt_path = gt_dir / f"{sid}.nii.gz"
        if not gt_path.exists():
            print(f"  [SKIP] missing GT for {sid}")
            continue
        pred = load_mask(p)
        gt = load_mask(gt_path)
        # broadcast to (B, C, Z, H, W) = (1, 1, Z, H, W)
        y_prob = torch.from_numpy(pred)[None, None]
        y_true = torch.from_numpy(gt)[None, None]
        m = compute_metrics(y_prob, y_true, threshold=args.threshold)
        per_case[sid] = m
        if accum is None:
            accum = {k: 0.0 for k in m}
            n_metrics = 0
        for k, v in m.items():
            accum[k] += float(v)
        n_metrics += 1
        print(f"  {sid}: dice={m.get('dice', float('nan')):.4f}  iou={m.get('iou', float('nan')):.4f}")

    mean = {k: accum[k] / max(1, n_metrics) for k in accum} if accum else {}
    out = {"mean": mean, "per_case": per_case, "n_cases": n_metrics}
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nMean over {n_metrics} test cases:")
    for k, v in mean.items():
        print(f"  {k:>20s}: {v:.4f}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
