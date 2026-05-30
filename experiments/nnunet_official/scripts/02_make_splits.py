"""
Write splits_final.json for each nnUNetv2 dataset using the repo's exact
train/val partition. Test cases are kept out of splits_final.json (they live
in imagesTs/ and are evaluated post-hoc).

Must be run AFTER plan_and_preprocess, because nnUNetv2 needs the
preprocessed dataset folder to exist for splits_final.json to be honored.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from datasets import DATASETS, dataset_folder  # noqa: E402
from splits import compute_split  # noqa: E402


def list_studies(src_dir: Path):
    return sorted(
        d.name for d in src_dir.iterdir()
        if d.is_dir() and (d.name.startswith("dbt_") or d.name.startswith("real_dbt_"))
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nnunet-preprocessed", required=True)
    ap.add_argument("--datasets", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pre_root = Path(args.nnunet_preprocessed).resolve()

    for dsid in args.datasets:
        meta = DATASETS[dsid]
        studies = list_studies(meta["src"])
        split = compute_split(studies, seed=args.seed)

        # nnUNetv2 expects a list of 5 folds; we replicate the same fold across
        # all positions so any --fold (0..4) trains/validates with the same data.
        fold0 = {"train": split["train"], "val": split["val"]}
        splits = [fold0] * 5

        dst_dir = pre_root / dataset_folder(dsid)
        dst_dir.mkdir(parents=True, exist_ok=True)
        out_path = dst_dir / "splits_final.json"
        with open(out_path, "w") as f:
            json.dump(splits, f, indent=2)
        print(f"[{meta['name']}] -> {out_path}  (train={len(fold0['train'])} val={len(fold0['val'])} test={len(split['test'])})")


if __name__ == "__main__":
    main()
