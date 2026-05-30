#!/usr/bin/env python3
"""
Prune slices (Z axis) in Dataset_Preprocessed volumes.

Rule requested:
- Training will use at most N slices per volume.
- Remove artifact slices: first K and last M slices (by default K=10, M=8).
- From the remaining, keep at most N slices (default N=20).

This script edits the dataset IN PLACE (with optional backup).

Expected structure per case:
  Dataset_Preprocessed/**/<case_id>_img.npy
    <case_id>_img.npy            (Z,H,W)
    <case_id>_mask.npy           (Z,H,W)
    <case_id>_tissue_mask.npy    (optional) (Z,H,W)
    preprocess_params.json       (optional; will be updated with pruning info)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _natural_key(text: str) -> List[object]:
    import re

    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", text)]


def _find_tissue_mask(case_dir: Path, case_id: str) -> Optional[Path]:
    exact = case_dir / f"{case_id}_tissue_mask.npy"
    if exact.exists():
        return exact
    with_c = case_dir / f"{case_id}C_tissue_mask.npy"
    if with_c.exists():
        return with_c
    candidates = sorted(case_dir.glob(f"{case_id}*tissue_mask.npy"), key=lambda p: _natural_key(p.name))
    return candidates[0] if candidates else None


def _safe_write_npy(dst: Path, arr: np.ndarray) -> None:
    """
    Atomic-ish write: write to temp file then replace.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    # Importante: np.save(pathlib.Path) puede agregar ".npy" si el sufijo no es ".npy".
    # Para evitarlo, escribimos sobre un file-handle.
    with open(tmp, "wb") as f:
        np.save(f, arr)
    tmp.replace(dst)


def _backup_file(path: Path, *, backup_mode: str, backup_dir: Optional[Path]) -> Optional[Path]:
    """
    backup_mode:
      - none: no backup
      - rename: move file to <name>.bak.<timestamp>
      - copy: copy file to backup_dir preserving relative structure
    """
    if backup_mode == "none":
        return None

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if backup_mode == "rename":
        bak = path.with_name(path.name + f".bak.{ts}")
        path.replace(bak)
        return bak

    if backup_mode == "copy":
        if backup_dir is None:
            raise ValueError("backup_mode=copy requiere --backup-dir")
        backup_dir.mkdir(parents=True, exist_ok=True)
        # store full path tail from dataset root? caller passes already-relativized location by using backup_dir/case_id/...
        dst = backup_dir / path.name
        shutil.copy2(path, dst)
        return dst

    raise ValueError(f"backup_mode no reconocido: {backup_mode}")


@dataclass(frozen=True)
class CasePaths:
    case_id: str
    case_dir: Path
    img_path: Path
    mask_path: Path
    tissue_mask_path: Optional[Path]
    params_path: Optional[Path]


def discover_cases(root: Path) -> List[CasePaths]:
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"No existe el directorio: {root}")

    cases: List[CasePaths] = []
    img_paths = sorted(
        [p for p in root.rglob("*_img.npy") if p.is_file()],
        key=lambda p: _natural_key(str(p.relative_to(root))),
    )
    for img in img_paths:
        case_dir = img.parent
        case_id = img.name[:-8]  # remove "_img.npy"
        msk = case_dir / f"{case_id}_mask.npy"
        if not msk.exists():
            continue
        tissue = _find_tissue_mask(case_dir, case_id)
        params = case_dir / "preprocess_params.json"
        cases.append(
            CasePaths(
                case_id=case_id,
                case_dir=case_dir,
                img_path=img,
                mask_path=msk,
                tissue_mask_path=tissue,
                params_path=params if params.exists() else None,
            )
        )
    if not cases:
        raise RuntimeError(f"No se encontraron casos con '*_img.npy' y '*_mask.npy' dentro de {root}.")
    return cases


def compute_keep_range(
    z_total: int,
    *,
    remove_first: int,
    remove_last: int,
    max_slices: int,
    select: str,
) -> Optional[Tuple[int, int]]:
    """
    Returns (z0, z1) to keep, half-open, or None if cannot compute.
    """
    start = int(remove_first)
    end = int(z_total) - int(remove_last)
    if z_total <= 0:
        return None
    if end <= start:
        return None

    remaining = end - start
    if remaining <= max_slices:
        return start, end

    if select == "first":
        z0 = start
        z1 = start + max_slices
        return z0, z1

    if select == "centered":
        extra = remaining - max_slices
        z0 = start + (extra // 2)
        z1 = z0 + max_slices
        return z0, z1

    raise ValueError(f"select no reconocido: {select}")


def update_params_json(
    params_path: Path,
    *,
    case_id: str,
    z_total_before: int,
    z_keep: Tuple[int, int],
    z_total_after: int,
    remove_first: int,
    remove_last: int,
    max_slices: int,
    select: str,
    dry_run: bool,
    backup_mode: str,
    backup_dir_for_case: Optional[Path],
) -> None:
    try:
        with open(params_path, "r", encoding="utf-8") as f:
            data: Dict[str, object] = json.load(f)
    except Exception:
        data = {}

    data["slice_pruning"] = {
        "enabled": True,
        "axis": 0,
        "remove_first": int(remove_first),
        "remove_last": int(remove_last),
        "max_slices": int(max_slices),
        "select": str(select),
        "z_total_before": int(z_total_before),
        "z_keep_range": [int(z_keep[0]), int(z_keep[1])],
        "z_total_after": int(z_total_after),
    }

    if dry_run:
        return

    # backup json, then write
    _backup_file(
        params_path,
        backup_mode=backup_mode,
        backup_dir=(backup_dir_for_case / "json") if (backup_mode == "copy" and backup_dir_for_case) else None,
    )
    params_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = params_path.with_suffix(params_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(params_path)


def process_case(
    cp: CasePaths,
    *,
    remove_first: int,
    remove_last: int,
    max_slices: int,
    select: str,
    dry_run: bool,
    backup_mode: str,
    backup_dir: Optional[Path],
    verbose: bool,
) -> Tuple[bool, str]:
    # Use memmap for lower RAM; slicing will create a view then we materialize.
    img = np.load(cp.img_path, mmap_mode="r")
    mask = np.load(cp.mask_path, mmap_mode="r")
    tissue = np.load(cp.tissue_mask_path, mmap_mode="r") if cp.tissue_mask_path else None

    # Squeeze if needed
    img = np.squeeze(img)
    mask = np.squeeze(mask)
    tissue = np.squeeze(tissue) if tissue is not None else None

    if img.ndim != 3 or mask.ndim != 3 or (tissue is not None and tissue.ndim != 3):
        return False, f"{cp.case_id}: ndim inesperado img{img.ndim} mask{mask.ndim}" + (f" tissue{tissue.ndim}" if tissue is not None else "")
    if img.shape != mask.shape:
        return False, f"{cp.case_id}: shape mismatch img{img.shape} vs mask{mask.shape}"
    if tissue is not None and tissue.shape != img.shape:
        return False, f"{cp.case_id}: shape mismatch img{img.shape} vs tissue{tissue.shape}"

    z_total = int(img.shape[0])
    keep = compute_keep_range(
        z_total,
        remove_first=remove_first,
        remove_last=remove_last,
        max_slices=max_slices,
        select=select,
    )
    if keep is None:
        return False, f"{cp.case_id}: no se puede recortar (z_total={z_total}, remove_first={remove_first}, remove_last={remove_last})"

    z0, z1 = keep
    new_img = np.asarray(img[z0:z1]).copy()
    new_mask = np.asarray(mask[z0:z1]).copy()
    new_tissue = np.asarray(tissue[z0:z1]).copy() if tissue is not None else None

    if new_img.shape != new_mask.shape:
        return False, f"{cp.case_id}: post-shape mismatch img{new_img.shape} vs mask{new_mask.shape}"
    if new_tissue is not None and new_tissue.shape != new_img.shape:
        return False, f"{cp.case_id}: post-shape mismatch img{new_img.shape} vs tissue{new_tissue.shape}"

    # Prepare per-case backup dir if needed
    backup_dir_for_case = (backup_dir / cp.case_id) if (backup_mode == "copy" and backup_dir is not None) else None
    if backup_dir_for_case is not None:
        (backup_dir_for_case / "npy").mkdir(parents=True, exist_ok=True)
        (backup_dir_for_case / "json").mkdir(parents=True, exist_ok=True)

    if not dry_run:
        # backup + write img/mask/tissue
        if backup_mode == "copy":
            # copy originals first
            for p in [cp.img_path, cp.mask_path] + ([cp.tissue_mask_path] if cp.tissue_mask_path else []):
                if p is None:
                    continue
                dst = (backup_dir_for_case / "npy" / p.name) if backup_dir_for_case else None
                if dst is None:
                    raise RuntimeError("backup_dir_for_case no pudo construirse")
                shutil.copy2(p, dst)
        elif backup_mode == "rename":
            _backup_file(cp.img_path, backup_mode="rename", backup_dir=None)
            _backup_file(cp.mask_path, backup_mode="rename", backup_dir=None)
            if cp.tissue_mask_path:
                _backup_file(cp.tissue_mask_path, backup_mode="rename", backup_dir=None)
        elif backup_mode == "none":
            pass
        else:
            raise ValueError(f"backup_mode no reconocido: {backup_mode}")

        _safe_write_npy(cp.img_path, new_img)
        _safe_write_npy(cp.mask_path, new_mask)
        if cp.tissue_mask_path and new_tissue is not None:
            _safe_write_npy(cp.tissue_mask_path, new_tissue)

    # Update preprocess_params.json (optional)
    if cp.params_path is not None:
        update_params_json(
            cp.params_path,
            case_id=cp.case_id,
            z_total_before=z_total,
            z_keep=(z0, z1),
            z_total_after=int(new_img.shape[0]),
            remove_first=remove_first,
            remove_last=remove_last,
            max_slices=max_slices,
            select=select,
            dry_run=dry_run,
            backup_mode=backup_mode,
            backup_dir_for_case=backup_dir_for_case,
        )

    msg = (
        f"{cp.case_id}: OK "
        f"(Z {z_total} -> {new_img.shape[0]}) "
        f"keep=[{z0}:{z1}] "
        f"rm_first={remove_first} rm_last={remove_last} max={max_slices} select={select}"
    )
    if verbose and cp.tissue_mask_path is None:
        msg += " (sin tissue_mask)"
    return True, msg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recorta slices Z en Dataset_Preprocessed (npy) para entrenamiento.")
    p.add_argument(
        "--root",
        type=Path,
        default=Path("Dataset_Preprocessed"),
        help="Ruta a Dataset_Preprocessed/",
    )
    p.add_argument("--remove-first", type=int, default=10, help="Cantidad de slices iniciales a eliminar.")
    p.add_argument("--remove-last", type=int, default=8, help="Cantidad de slices finales a eliminar.")
    p.add_argument("--max-slices", type=int, default=20, help="Máximo de slices a conservar por volumen.")
    p.add_argument(
        "--select",
        choices=["centered", "first"],
        default="centered",
        help="Cómo escoger los N slices si sobran: centrados o los primeros tras el recorte.",
    )
    p.add_argument(
        "--backup",
        choices=["rename", "copy", "none"],
        default="rename",
        help="Qué hacer con los archivos originales antes de sobrescribirlos.",
    )
    p.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="Directorio donde guardar backups si --backup=copy. Se recomienda ruta absoluta.",
    )
    p.add_argument("--dry-run", action="store_true", help="No escribe nada; solo reporta lo que haría.")
    p.add_argument("--limit", type=int, default=0, help="Procesa solo los primeros N casos (0 = todos).")
    p.add_argument("--verbose", action="store_true", help="Imprime más detalles.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root: Path = args.root

    if args.backup == "copy" and args.backup_dir is None:
        print("[ERROR] --backup=copy requiere --backup-dir", file=sys.stderr)
        return 2

    cases = discover_cases(root)
    if args.limit and args.limit > 0:
        cases = cases[: int(args.limit)]

    if args.backup == "copy":
        # Create a timestamped subdir to avoid collisions and keep a clean "run" record
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_root = args.backup_dir / f"slice_prune_backup_{ts}"
    else:
        backup_root = None

    ok = 0
    fail = 0
    for cp in cases:
        try:
            success, msg = process_case(
                cp,
                remove_first=args.remove_first,
                remove_last=args.remove_last,
                max_slices=args.max_slices,
                select=args.select,
                dry_run=bool(args.dry_run),
                backup_mode=str(args.backup),
                backup_dir=backup_root,
                verbose=bool(args.verbose),
            )
            if success:
                ok += 1
                print(msg)
            else:
                fail += 1
                print(f"{cp.case_id}: FAIL - {msg}", file=sys.stderr)
        except Exception as exc:
            fail += 1
            print(f"{cp.case_id}: FAIL - excepción: {exc}", file=sys.stderr)

    print(f"[RESUMEN] ok={ok} fail={fail} root={root} dry_run={bool(args.dry_run)} backup={args.backup}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())


