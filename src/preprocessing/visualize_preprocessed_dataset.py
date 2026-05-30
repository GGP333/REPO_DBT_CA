#!/usr/bin/env python3
"""Visor interactivo para Dataset_Preprocessed (volúmenes .npy).

Muestra por caso:
- Imagen original (float32)
- Máscara GT (uint8/bool)
- Máscara tissue (uint8/bool) si existe
- Overlay (imagen + máscaras)

Navegacin:
- Rueda del mouse: cambia slice (Z)
- Flechas arriba/abajo: cambia slice (Z)
- Flechas izquierda/derecha: cambia caso
- Slider inferior: cambia slice (Z)
- Slider superior: cambia caso

Lectura de pixel:
- Al mover el mouse sobre cualquiera de los paneles, muestra (x,y,z) y valores
  de imagen/mscara en el ttulo.

Requisitos:
- numpy
- matplotlib

Ejemplo:
  python scripts/visualize_preprocessed_dataset.py --root Dataset_Preprocessed
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


def _natural_key(text: str) -> List[object]:
    import re

    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", text)]


@dataclass(frozen=True)
class CasePaths:
    case_id: str
    img_path: Path
    mask_path: Path
    tissue_mask_path: Optional[Path] = None


def _find_tissue_mask(case_dir: Path, case_id: str) -> Optional[Path]:
    """
    Soporta distintos nombres:
    - {case_id}_tissue_mask.npy
    - {case_id}C_tissue_mask.npy (observado en tu dataset)
    - cualquier {case_id}*tissue_mask.npy como fallback
    """
    exact = case_dir / f"{case_id}_tissue_mask.npy"
    if exact.exists():
        return exact

    with_c = case_dir / f"{case_id}C_tissue_mask.npy"
    if with_c.exists():
        return with_c

    # Fallback: busca cualquier variante compatible
    candidates = sorted(case_dir.glob(f"{case_id}*tissue_mask.npy"), key=lambda p: _natural_key(p.name))
    return candidates[0] if candidates else None


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
        cases.append(CasePaths(case_id=case_id, img_path=img, mask_path=msk, tissue_mask_path=tissue))

    if not cases:
        raise RuntimeError(
            "No se encontraron casos con '*_img.npy' y '*_mask.npy' dentro de "
            f"{root}. Esperaba estructura tipo Dataset_Preprocessed/**/img_dbt_XXX_img.npy"
        )

    return cases


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visor interactivo de Dataset_Preprocessed (npy)")
    p.add_argument(
        "--root",
        type=Path,
        default=Path("Dataset_Preprocessed"),
        help="Ruta a Dataset_Preprocessed/",
    )
    p.add_argument(
        "--start_case",
        type=str,
        default=None,
        help="Nombre de carpeta de caso para iniciar (ej: img_dbt_004).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.35,
        help="Alpha del overlay de máscaras sobre la imagen (0-1).",
    )
    return p.parse_args()


def robust_window(vol, fallback: Tuple[float, float] = (-2.0, 2.0)) -> Tuple[float, float]:
    """Devuelve (vmin,vmax) robustos para mostrar imgen normalizada."""
    try:
        import numpy as np

        v = vol
        if hasattr(v, "ndim") and v.ndim == 3:
            # usar solo voxels no cero para evitar que el fondo domine
            nz = v[v != 0]
            if nz.size > 0:
                lo, hi = np.percentile(nz, [1, 99])
            else:
                lo, hi = np.percentile(v, [1, 99])
        else:
            lo, hi = float(np.min(v)), float(np.max(v))

        if not (lo < hi):
            return fallback
        # evita ventanas absurdas
        if (hi - lo) < 1e-6:
            return fallback
        return float(lo), float(hi)
    except Exception:
        return fallback


def main() -> None:
    args = parse_args()

    # imports tardos para dar mejor error si faltan dependencias
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError(
            "Falta numpy. Instala con: pip install numpy matplotlib\n"
            "(o activa tu entorno con las dependencias del proyecto)."
        ) from exc

    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    root = args.root
    cases = discover_cases(root)

    # ndice inicial
    case_idx = 0
    if args.start_case is not None:
        for i, c in enumerate(cases):
            if c.case_id == args.start_case:
                case_idx = i
                break

    # estado mutable
    state = {
        "case_idx": case_idx,
        "z": 0,
        "img": None,
        "mask": None,          # GT
        "tissue_mask": None,   # Tissue (opcional)
        "vmin": None,
        "vmax": None,
        "shape": None,
    }

    def load_case(i: int) -> None:
        i = int(max(0, min(len(cases) - 1, i)))
        cp = cases[i]

        img = np.load(cp.img_path)
        mask = np.load(cp.mask_path)
        tissue = np.load(cp.tissue_mask_path) if cp.tissue_mask_path else None

        if img.ndim != 3 or mask.ndim != 3 or (tissue is not None and tissue.ndim != 3):
            raise ValueError(
                f"Se esperaba 3D (z,y,x). Got img {img.shape} mask {mask.shape}"
                + (f" tissue {tissue.shape}" if tissue is not None else "")
                + f" en {cp.case_id}"
            )
        if img.shape != mask.shape:
            raise ValueError(f"Shape imagen {img.shape} != mask {mask.shape} en {cp.case_id}")
        if tissue is not None and tissue.shape != img.shape:
            raise ValueError(f"Shape imagen {img.shape} != tissue_mask {tissue.shape} en {cp.case_id}")

        state["case_idx"] = i
        state["img"] = img
        state["mask"] = (mask > 0).astype(np.uint8)
        state["tissue_mask"] = (tissue > 0).astype(np.uint8) if tissue is not None else None
        state["shape"] = img.shape
        state["z"] = int(max(0, min(state["z"], img.shape[0] - 1)))
        vmin, vmax = robust_window(img)
        state["vmin"], state["vmax"] = vmin, vmax

    # carga inicial
    load_case(state["case_idx"])

    matplotlib.rcParams["keymap.back"] = []
    matplotlib.rcParams["keymap.forward"] = []

    fig, axes = plt.subplots(1, 4, figsize=(17.5, 5.0))
    ax_img, ax_msk, ax_tis, ax_ovr = axes

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    img_artist = ax_img.imshow(state["img"][state["z"]], cmap="gray", vmin=state["vmin"], vmax=state["vmax"])
    ax_img.set_title("Imagen")

    msk_artist = ax_msk.imshow(state["mask"][state["z"]], cmap="gray", vmin=0, vmax=1)
    ax_msk.set_title("Máscara (GT)")

    tis0 = state["tissue_mask"][state["z"]] if state["tissue_mask"] is not None else np.zeros_like(state["mask"][state["z"]])
    tis_artist = ax_tis.imshow(tis0, cmap="gray", vmin=0, vmax=1)
    ax_tis.set_title("Máscara (tissue)")

    ovr_base = ax_ovr.imshow(state["img"][state["z"]], cmap="gray", vmin=state["vmin"], vmax=state["vmax"])
    ovr_mask_gt = ax_ovr.imshow(state["mask"][state["z"]], cmap="autumn", vmin=0, vmax=1, alpha=args.alpha)
    ovr_mask_tis = ax_ovr.imshow(tis0, cmap="winter", vmin=0, vmax=1, alpha=args.alpha)
    ax_ovr.set_title("Overlay")

    # Sliders
    fig.subplots_adjust(bottom=0.18, top=0.88)

    ax_case = fig.add_axes([0.15, 0.92, 0.7, 0.03])
    case_slider = Slider(
        ax=ax_case,
        label="Caso",
        valmin=0,
        valmax=max(0, len(cases) - 1),
        valinit=state["case_idx"],
        valstep=1,
    )

    ax_z = fig.add_axes([0.15, 0.08, 0.7, 0.03])
    z_slider = Slider(
        ax=ax_z,
        label="Slice Z",
        valmin=0,
        valmax=max(0, state["shape"][0] - 1),
        valinit=state["z"],
        valstep=1,
    )

    hover_text = {"last": ""}

    def set_suptitle(extra: str = "") -> None:
        cp = cases[state["case_idx"]]
        z = state["z"]
        zmax = state["shape"][0] - 1
        base = f"{cp.case_id} | z={z}/{zmax}"
        if extra:
            base += f" | {extra}"
        fig.suptitle(base)

    set_suptitle()

    def redraw_slice() -> None:
        z = int(state["z"])
        img_artist.set_data(state["img"][z])
        msk_artist.set_data(state["mask"][z])
        tis = state["tissue_mask"][z] if state["tissue_mask"] is not None else np.zeros_like(state["mask"][z])
        tis_artist.set_data(tis)
        ovr_base.set_data(state["img"][z])
        ovr_mask_gt.set_data(state["mask"][z])
        ovr_mask_tis.set_data(tis)
        fig.canvas.draw_idle()

    def on_case_change(val) -> None:
        i = int(val)
        load_case(i)
        # actualizar rango de z
        z_slider.valmax = max(0, state["shape"][0] - 1)
        z_slider.ax.set_xlim(z_slider.valmin, z_slider.valmax)
        z_slider.set_val(int(min(state["z"], z_slider.valmax)))

        # actualizar ventanas
        img_artist.set_clim(state["vmin"], state["vmax"])
        ovr_base.set_clim(state["vmin"], state["vmax"])

        set_suptitle()
        redraw_slice()

    def on_z_change(val) -> None:
        state["z"] = int(val)
        set_suptitle(hover_text["last"])
        redraw_slice()

    case_slider.on_changed(on_case_change)
    z_slider.on_changed(on_z_change)

    def step_case(delta: int) -> None:
        new_idx = int(max(0, min(len(cases) - 1, state["case_idx"] + delta)))
        if new_idx != state["case_idx"]:
            case_slider.set_val(new_idx)

    def step_z(delta: int) -> None:
        new_z = int(max(0, min(state["shape"][0] - 1, state["z"] + delta)))
        if new_z != state["z"]:
            z_slider.set_val(new_z)

    def on_scroll(event) -> None:
        if event.inaxes not in (ax_img, ax_msk, ax_tis, ax_ovr):
            return
        if event.button == "up":
            step_z(-1)
        elif event.button == "down":
            step_z(+1)

    def on_key(event) -> None:
        if event.key == "left":
            step_case(-1)
        elif event.key == "right":
            step_case(+1)
        elif event.key == "up":
            step_z(-1)
        elif event.key == "down":
            step_z(+1)

    def on_move(event) -> None:
        if event.inaxes not in (ax_img, ax_msk, ax_tis, ax_ovr):
            hover_text["last"] = ""
            set_suptitle()
            fig.canvas.draw_idle()
            return
        if event.xdata is None or event.ydata is None:
            return

        x = int(round(event.xdata))
        y = int(round(event.ydata))
        z = int(state["z"])
        _, h, w = state["shape"]
        if not (0 <= x < w and 0 <= y < h):
            return

        img_val = float(state["img"][z, y, x])
        msk_val = int(state["mask"][z, y, x])
        tis_val = int(state["tissue_mask"][z, y, x]) if state["tissue_mask"] is not None else -1
        txt = f"(x,y)=({x},{y}) img={img_val:.4g} gt={msk_val} tissue={tis_val}"
        hover_text["last"] = txt
        set_suptitle(txt)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("motion_notify_event", on_move)

    plt.show()


if __name__ == "__main__":
    main()
