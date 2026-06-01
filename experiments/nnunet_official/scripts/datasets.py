"""Central registry: nnUNetv2 dataset IDs, names and source paths."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "Dataset_Preprocessed"

DATASETS = {
    1: {"name": "DBTLarge",          "src": SRC_ROOT / "Dataset_large_tumor"},
    2: {"name": "DBTSmall",          "src": SRC_ROOT / "Dataset_small_tumor"},
    3: {"name": "DBTMixedSize",           "src": SRC_ROOT / "Dataset_Mixed_Size"},
    4: {"name": "DBTHybrid",  "src": SRC_ROOT / "Dataset_Hybrid"},
}


def dataset_folder(dsid: int) -> str:
    return f"Dataset{dsid:03d}_{DATASETS[dsid]['name']}"
