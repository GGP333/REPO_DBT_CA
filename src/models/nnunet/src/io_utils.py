# src/io_utils.py
from __future__ import annotations
import os
import re
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class StudyRecord:
    study_id: str
    img_path: str
    mask_path: str

def get_logger(name: str = "dbt_io") -> logging.Logger:
    """Crea un logger simple (no configura root)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter("[%(levelname)s] %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger

_ID_RE = re.compile(r"dbt[_\-]?(\d+)", re.IGNORECASE)

def _extract_id(text: str) -> Optional[str]:
    m = _ID_RE.search(text)
    return m.group(1) if m else None

def _pick_npy(npy_files: List[str], target_id: Optional[str], kind_hint: str) -> Optional[str]:
    """
    Selecciona un .npy apropiado dentro de una lista, priorizando:
      1) que contenga el id del estudio
      2) que contenga 'img' o 'mask' acorde a kind_hint
      3) el único archivo si solo hay uno
      4) si hay varios, el más largo (heurística) como fallback
    """
    if not npy_files:
        return None
    candidates = []
    if target_id:
        candidates = [p for p in npy_files if target_id in os.path.basename(p)]
        if candidates and kind_hint:
            by_kind = [p for p in candidates if kind_hint.lower() in os.path.basename(p).lower()]
            if by_kind:
                candidates = by_kind
    if not candidates and kind_hint:
        candidates = [p for p in npy_files if kind_hint.lower() in os.path.basename(p).lower()]
    if not candidates and len(npy_files) == 1:
        return npy_files[0]
    if not candidates:
        # fallback: mayor longitud de nombre
        candidates = sorted(npy_files, key=lambda x: len(os.path.basename(x)), reverse=True)
    return candidates[0] if candidates else None

def _scan_for(kind_dir: str) -> List[str]:
    """Escanea recursivamente una carpeta buscando .npy."""
    found = []
    for root, _, files in os.walk(kind_dir):
        for f in files:
            if f.lower().endswith(".npy"):
                found.append(os.path.join(root, f))
    return sorted(found)

def discover_studies(dataset_root: str, logger: Optional[logging.Logger] = None) -> List[StudyRecord]:
    """
    Descubre estudios `dbt_*` y `real_dbt_*` bajo `dataset_root` y empareja img/mask con lógica defensiva.

    - Tolera typos en nombres de archivos (e.g., `img_dbt_18.npy` dentro de `dbt_018`).
    - Advierte y excluye estudios sin par completo.
    """
    logger = logger or get_logger()
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"dataset_root no existe: {dataset_root}")

    # Listar directorios tipo dbt_*, real_dbt_* o img_dbt_* (versiones preprocesadas)
    candidates = [os.path.join(dataset_root, d) for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    studies_dirs = [
        d
        for d in candidates
        if re.search(r"^(real_dbt|dbt|img_dbt)[_\-]?\d+$", os.path.basename(d), re.IGNORECASE)
    ]
    studies_dirs = sorted(studies_dirs)

    records: List[StudyRecord] = []
    if not studies_dirs:
        logger.warning("No se encontraron carpetas tipo 'dbt_XXX', 'real_dbt_XXX' o 'img_dbt_XXX' en el dataset_root.")

    for sdir in studies_dirs:
        sid_folder = os.path.basename(sdir)
        sid_digits = _extract_id(sid_folder)  # e.g., "018"

        # Encontrar subcarpetas img_* y mask_* (o heurísticas)
        subdirs = [os.path.join(sdir, sd) for sd in os.listdir(sdir) if os.path.isdir(os.path.join(sdir, sd))]
        img_dirs = [d for d in subdirs if os.path.basename(d).lower().startswith("img")]
        mask_dirs = [d for d in subdirs if os.path.basename(d).lower().startswith("mask")]

        # Si no hay subcarpetas img/mask pero el estudio contiene directamente
        # archivos *_img.npy y *_mask.npy, úsalo sin warnings.
        root_npy = _scan_for(sdir)
        has_img_root = any("img" in os.path.basename(p).lower() for p in root_npy)
        has_mask_root = any("mask" in os.path.basename(p).lower() for p in root_npy)
        if not img_dirs and not mask_dirs and has_img_root and has_mask_root:
            img_dirs = [sdir]
            mask_dirs = [sdir]

        if not img_dirs:
            img_dirs = [sdir]
            logger.warning(f"[{sid_folder}] No se encontró carpeta 'img_*'. Buscando .npy en el directorio del estudio.")
        if not mask_dirs:
            mask_dirs = [sdir]
            logger.warning(f"[{sid_folder}] No se encontró carpeta 'mask_*'. Buscando .npy en el directorio del estudio.")

        img_npy = []
        for d in img_dirs:
            img_npy += _scan_for(d)
        mask_npy = []
        for d in mask_dirs:
            mask_npy += _scan_for(d)

        img_path = _pick_npy(img_npy, sid_digits, "img")
        mask_path = _pick_npy(mask_npy, sid_digits, "mask")

        if not img_path or not mask_path:
            logger.warning(f"[{sid_folder}] Estudio excluido por falta de par img/mask (.npy). img: {bool(img_path)}, mask: {bool(mask_path)}")
            continue

        # Warnings por desalineaciones de nombres
        for (p, kind) in [(img_path, "img"), (mask_path, "mask")]:
            base = os.path.basename(p)
            if sid_digits and sid_digits not in base:
                logger.warning(f"[{sid_folder}] {kind} con id no coincidente: '{base}'. Se usará igualmente.")

        records.append(StudyRecord(study_id=sid_folder, img_path=img_path, mask_path=mask_path))

    # Fallback: estructura tipo img_dbt_XXX con archivos *_img.npy y *_mask.npy
    if not records:
        fallback_dirs = [d for d in candidates if re.search(r"^img_dbt[_\-]?\d+$", os.path.basename(d), re.IGNORECASE)]
        fallback_dirs = sorted(fallback_dirs)
        for fdir in fallback_dirs:
            sid = os.path.basename(fdir)
            files = [os.path.join(fdir, f) for f in os.listdir(fdir) if f.lower().endswith(".npy")]
            img_path = _pick_npy(files, _extract_id(sid), "img")
            mask_path = _pick_npy(files, _extract_id(sid), "mask")
            if img_path and mask_path:
                records.append(StudyRecord(study_id=sid, img_path=img_path, mask_path=mask_path))
            else:
                logger.warning(f"[{sid}] Fallback sin par img/mask (*.npy).")

    if not records:
        logger.warning("No quedó ningún estudio emparejado. Revisa la estructura y nombres.")
    else:
        logger.info(f"Estudios emparejados: {len(records)}")
        for rec in records:
            logger.info(f"[{rec.study_id}] img={rec.img_path} | mask={rec.mask_path}")

    return records
