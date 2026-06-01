#!/usr/bin/env python3
"""
Visor interactivo de volúmenes DBT — Dataset (TIFF) y Dataset_Preprocessed (npy).

Modos de visualización:
  1) Ver un volumen individual (imagen + máscara + overlay)
  2) Comparar Raw vs Preprocessed lado a lado para el mismo caso

Navegación:
  - Rueda del ratón / flechas arriba-abajo   → cambiar slice (Z)
  - Flechas izquierda-derecha                 → cambiar caso
  - Slider inferior                           → cambiar slice
  - Slider superior                           → cambiar caso
  - Hover del mouse                           → leer valores de píxel

Requisitos:
  pip install numpy matplotlib tifffile

Ejemplo:
  python scripts/visualize_volumes.py
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ──────────────────────────────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_ROOT = BASE_DIR / "Dataset"
PREP_ROOT = BASE_DIR / "Dataset_Preprocessed"

SUBDATASETS = [
    "Dataset_Hybrid",
    "Dataset_Mixed_Size",
    "Dataset_small_tumor",
    "Dataset_large_tumor",
]


# ──────────────────────────────────────────────────────────────────────
# Utilidades
# ──────────────────────────────────────────────────────────────────────
def _natural_key(text: str) -> list:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", text)]


def _cls() -> None:
    print("\033[2J\033[H", end="")


def _pick(options: list[str], title: str, allow_back: bool = True) -> Optional[int]:
    """Menú interactivo en terminal. Devuelve índice elegido o None (atrás)."""
    while True:
        _cls()
        print(f"\n{'═' * 60}")
        print(f"  {title}")
        print(f"{'═' * 60}")
        for i, opt in enumerate(options, 1):
            print(f"  [{i:3d}]  {opt}")
        if allow_back:
            print(f"\n  [  0]  ← Volver / Salir")
        print(f"{'─' * 60}")
        try:
            raw = input("  Elige una opción: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if not raw:
            continue
        try:
            val = int(raw)
        except ValueError:
            continue
        if val == 0 and allow_back:
            return None
        if 1 <= val <= len(options):
            return val - 1
    return None


# ──────────────────────────────────────────────────────────────────────
# Descubrimiento de casos
# ──────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class RawCase:
    case_id: str
    img_tiff: Path
    mask_tiff: Path


@dataclass(frozen=True)
class PrepCase:
    case_id: str
    img_npy: Path
    mask_npy: Path


def discover_raw_cases(subdataset: str) -> List[RawCase]:
    root = RAW_ROOT / subdataset
    if not root.is_dir():
        return []
    cases: List[RawCase] = []
    for case_dir in sorted(root.iterdir(), key=lambda p: _natural_key(p.name)):
        if not case_dir.is_dir():
            continue
        cid = case_dir.name
        img_dir = case_dir / f"img_{cid}"
        mask_dir = case_dir / f"mask_{cid}"
        if not img_dir.is_dir() or not mask_dir.is_dir():
            continue
        # buscar tiff
        img_tiffs = sorted(img_dir.glob("*.tif*"), key=lambda p: _natural_key(p.name))
        mask_tiffs = sorted(mask_dir.glob("*.tif*"), key=lambda p: _natural_key(p.name))
        if img_tiffs and mask_tiffs:
            cases.append(RawCase(cid, img_tiffs[0], mask_tiffs[0]))
    return cases


def discover_prep_cases(subdataset: str) -> List[PrepCase]:
    root = PREP_ROOT / subdataset
    if not root.is_dir():
        return []
    cases: List[PrepCase] = []
    for img_path in sorted(root.rglob("*_img.npy"), key=lambda p: _natural_key(str(p))):
        case_dir = img_path.parent
        cid = img_path.name[:-8]  # quita _img.npy
        mask_path = case_dir / f"{cid}_mask.npy"
        if mask_path.exists():
            cases.append(PrepCase(cid, img_path, mask_path))
    return cases


# ──────────────────────────────────────────────────────────────────────
# Carga de volúmenes
# ──────────────────────────────────────────────────────────────────────
def load_tiff_volume(tiff_path: Path) -> np.ndarray:
    """Carga un TIFF (multi-página o single) como array 3D float32."""
    try:
        import tifffile
    except ImportError:
        print("  ⚠ Instala tifffile:  pip install tifffile")
        sys.exit(1)
    vol = tifffile.imread(str(tiff_path))
    if vol.ndim == 2:
        vol = vol[np.newaxis]
    if vol.ndim == 4 and vol.shape[0] == 1:
        vol = vol[0]
    if vol.ndim == 4 and vol.shape[-1] == 1:
        vol = vol[..., 0]
    return vol.astype(np.float32)


def load_npy_volume(npy_path: Path) -> np.ndarray:
    vol = np.load(npy_path)
    if vol.ndim == 2:
        vol = vol[np.newaxis]
    return vol.astype(np.float32)


def robust_window(vol: np.ndarray) -> Tuple[float, float]:
    nz = vol[vol != 0]
    if nz.size > 0:
        lo, hi = float(np.percentile(nz, [1, 99])[0]), float(np.percentile(nz, [1, 99])[1])
    else:
        lo, hi = float(vol.min()), float(vol.max())
    if hi - lo < 1e-6:
        lo, hi = float(vol.min()), float(vol.max()) + 1.0
    return lo, hi


# ──────────────────────────────────────────────────────────────────────
# Modo 1: Visor individual
# ──────────────────────────────────────────────────────────────────────
def view_single(cases, load_img_fn, load_mask_fn, alpha: float = 0.35) -> None:
    """
    Visor interactivo con slider de caso y slice.
    cases = lista de objetos con .case_id
    load_img_fn(case)  -> np.ndarray 3D
    load_mask_fn(case) -> np.ndarray 3D
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    matplotlib.rcParams["keymap.back"] = []
    matplotlib.rcParams["keymap.forward"] = []

    state = {"ci": 0, "z": 0, "img": None, "mask": None, "shape": None, "vmin": 0, "vmax": 1}

    def load(i: int) -> None:
        i = max(0, min(len(cases) - 1, i))
        c = cases[i]
        print(f"  Cargando {c.case_id} …", end=" ", flush=True)
        img = load_img_fn(c)
        mask = load_mask_fn(c)
        print(f"shape={img.shape}")
        state["ci"] = i
        state["img"] = img
        state["mask"] = (mask > 0).astype(np.float32)
        state["shape"] = img.shape
        state["z"] = min(state["z"], img.shape[0] - 1)
        state["vmin"], state["vmax"] = robust_window(img)

    load(0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    ax_img, ax_msk, ax_ovr = axes
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    z = state["z"]
    im_art = ax_img.imshow(state["img"][z], cmap="gray", vmin=state["vmin"], vmax=state["vmax"])
    ax_img.set_title("Imagen")
    mk_art = ax_msk.imshow(state["mask"][z], cmap="gray", vmin=0, vmax=1)
    ax_msk.set_title("Máscara")
    ov_base = ax_ovr.imshow(state["img"][z], cmap="gray", vmin=state["vmin"], vmax=state["vmax"])
    ov_mask = ax_ovr.imshow(state["mask"][z], cmap="autumn", vmin=0, vmax=1, alpha=alpha)
    ax_ovr.set_title("Overlay")

    fig.subplots_adjust(bottom=0.18, top=0.88)

    ax_cs = fig.add_axes([0.15, 0.93, 0.7, 0.03])
    sl_case = Slider(ax_cs, "Caso", 0, max(len(cases) - 1, 1), valinit=0, valstep=1)
    ax_zs = fig.add_axes([0.15, 0.06, 0.7, 0.03])
    sl_z = Slider(ax_zs, "Slice Z", 0, max(state["shape"][0] - 1, 1), valinit=0, valstep=1)

    hover = {"txt": ""}

    def suptitle(extra: str = "") -> None:
        c = cases[state["ci"]]
        t = f"{c.case_id}  |  z={state['z']}/{state['shape'][0]-1}  |  shape={state['shape']}"
        if extra:
            t += f"  |  {extra}"
        fig.suptitle(t, fontsize=10)

    suptitle()

    def redraw() -> None:
        z = state["z"]
        im_art.set_data(state["img"][z])
        mk_art.set_data(state["mask"][z])
        ov_base.set_data(state["img"][z])
        ov_mask.set_data(state["mask"][z])
        fig.canvas.draw_idle()

    def on_case(val):
        load(int(val))
        sl_z.valmax = max(state["shape"][0] - 1, 1)
        sl_z.ax.set_xlim(0, sl_z.valmax)
        sl_z.set_val(min(state["z"], sl_z.valmax))
        im_art.set_clim(state["vmin"], state["vmax"])
        ov_base.set_clim(state["vmin"], state["vmax"])
        suptitle(); redraw()

    def on_z(val):
        state["z"] = int(val)
        suptitle(hover["txt"]); redraw()

    sl_case.on_changed(on_case)
    sl_z.on_changed(on_z)

    def step_case(d):
        n = max(0, min(len(cases) - 1, state["ci"] + d))
        if n != state["ci"]:
            sl_case.set_val(n)

    def step_z(d):
        n = max(0, min(state["shape"][0] - 1, state["z"] + d))
        if n != state["z"]:
            sl_z.set_val(n)

    def on_scroll(e):
        if e.inaxes in axes:
            step_z(-1 if e.button == "up" else 1)

    def on_key(e):
        if e.key == "left": step_case(-1)
        elif e.key == "right": step_case(1)
        elif e.key == "up": step_z(-1)
        elif e.key == "down": step_z(1)

    def on_move(e):
        if e.inaxes not in axes or e.xdata is None:
            hover["txt"] = ""; suptitle(); fig.canvas.draw_idle(); return
        x, y = int(round(e.xdata)), int(round(e.ydata))
        _, h, w = state["shape"]
        if 0 <= x < w and 0 <= y < h:
            iv = float(state["img"][state["z"], y, x])
            mv = int(state["mask"][state["z"], y, x])
            hover["txt"] = f"(x={x}, y={y}) img={iv:.4g} mask={mv}"
            suptitle(hover["txt"]); fig.canvas.draw_idle()

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    plt.show()


# ──────────────────────────────────────────────────────────────────────
# Modo 2: Comparar Raw vs Preprocessed
# ──────────────────────────────────────────────────────────────────────
def view_compare(raw_cases: List[RawCase], prep_cases: List[PrepCase],
                 alpha: float = 0.35) -> None:
    """Side-by-side: Raw (izq) vs Preprocessed (der) para casos comunes."""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    matplotlib.rcParams["keymap.back"] = []
    matplotlib.rcParams["keymap.forward"] = []

    # Buscar casos comunes
    prep_dict = {c.case_id: c for c in prep_cases}
    pairs = [(r, prep_dict[r.case_id]) for r in raw_cases if r.case_id in prep_dict]
    if not pairs:
        print("  ⚠ No se encontraron casos comunes entre Dataset y Dataset_Preprocessed.")
        input("  Presiona Enter para volver …")
        return

    pairs.sort(key=lambda p: _natural_key(p[0].case_id))
    print(f"  {len(pairs)} casos comunes encontrados.")

    state = {
        "ci": 0,
        # raw
        "r_img": None, "r_mask": None, "r_shape": None, "r_z": 0,
        "r_vmin": 0, "r_vmax": 1,
        # prep
        "p_img": None, "p_mask": None, "p_shape": None, "p_z": 0,
        "p_vmin": 0, "p_vmax": 1,
    }

    def load(i: int) -> None:
        i = max(0, min(len(pairs) - 1, i))
        rc, pc = pairs[i]
        print(f"  Cargando {rc.case_id} (raw+prep) …", end=" ", flush=True)
        ri = load_tiff_volume(rc.img_tiff)
        rm = load_tiff_volume(rc.mask_tiff)
        pi = load_npy_volume(pc.img_npy)
        pm = load_npy_volume(pc.mask_npy)
        print(f"raw={ri.shape}  prep={pi.shape}")
        state["ci"] = i
        state["r_img"], state["r_mask"], state["r_shape"] = ri, (rm > 0).astype(np.float32), ri.shape
        state["p_img"], state["p_mask"], state["p_shape"] = pi, (pm > 0).astype(np.float32), pi.shape
        state["r_z"] = min(state["r_z"], ri.shape[0] - 1)
        state["p_z"] = min(state["p_z"], pi.shape[0] - 1)
        state["r_vmin"], state["r_vmax"] = robust_window(ri)
        state["p_vmin"], state["p_vmax"] = robust_window(pi)

    load(0)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    (ax_ri, ax_rm, ax_ro), (ax_pi, ax_pm, ax_po) = axes
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])

    rz, pz = state["r_z"], state["p_z"]

    ri_art = ax_ri.imshow(state["r_img"][rz], cmap="gray", vmin=state["r_vmin"], vmax=state["r_vmax"])
    ax_ri.set_title("Raw — Imagen", fontsize=9)
    rm_art = ax_rm.imshow(state["r_mask"][rz], cmap="gray", vmin=0, vmax=1)
    ax_rm.set_title("Raw — Máscara", fontsize=9)
    ro_b = ax_ro.imshow(state["r_img"][rz], cmap="gray", vmin=state["r_vmin"], vmax=state["r_vmax"])
    ro_m = ax_ro.imshow(state["r_mask"][rz], cmap="autumn", vmin=0, vmax=1, alpha=alpha)
    ax_ro.set_title("Raw — Overlay", fontsize=9)

    pi_art = ax_pi.imshow(state["p_img"][pz], cmap="gray", vmin=state["p_vmin"], vmax=state["p_vmax"])
    ax_pi.set_title("Prep — Imagen", fontsize=9)
    pm_art = ax_pm.imshow(state["p_mask"][pz], cmap="gray", vmin=0, vmax=1)
    ax_pm.set_title("Prep — Máscara", fontsize=9)
    po_b = ax_po.imshow(state["p_img"][pz], cmap="gray", vmin=state["p_vmin"], vmax=state["p_vmax"])
    po_m = ax_po.imshow(state["p_mask"][pz], cmap="autumn", vmin=0, vmax=1, alpha=alpha)
    ax_po.set_title("Prep — Overlay", fontsize=9)

    fig.subplots_adjust(bottom=0.20, top=0.88, hspace=0.28)

    ax_cs = fig.add_axes([0.15, 0.93, 0.7, 0.03])
    sl_case = Slider(ax_cs, "Caso", 0, max(len(pairs) - 1, 1), valinit=0, valstep=1)

    ax_rz = fig.add_axes([0.15, 0.10, 0.7, 0.03])
    sl_rz = Slider(ax_rz, "Raw Z", 0, max(state["r_shape"][0] - 1, 1), valinit=0, valstep=1)

    ax_pz = fig.add_axes([0.15, 0.04, 0.7, 0.03])
    sl_pz = Slider(ax_pz, "Prep Z", 0, max(state["p_shape"][0] - 1, 1), valinit=0, valstep=1)

    def suptitle() -> None:
        rc, _ = pairs[state["ci"]]
        fig.suptitle(
            f"{rc.case_id}  |  Raw z={state['r_z']}/{state['r_shape'][0]-1} shape={state['r_shape']}"
            f"  ·  Prep z={state['p_z']}/{state['p_shape'][0]-1} shape={state['p_shape']}",
            fontsize=10,
        )

    suptitle()

    def redraw() -> None:
        rz, pz = state["r_z"], state["p_z"]
        ri_art.set_data(state["r_img"][rz]); rm_art.set_data(state["r_mask"][rz])
        ro_b.set_data(state["r_img"][rz]); ro_m.set_data(state["r_mask"][rz])
        pi_art.set_data(state["p_img"][pz]); pm_art.set_data(state["p_mask"][pz])
        po_b.set_data(state["p_img"][pz]); po_m.set_data(state["p_mask"][pz])
        fig.canvas.draw_idle()

    def on_case(val):
        load(int(val))
        sl_rz.valmax = max(state["r_shape"][0] - 1, 1)
        sl_rz.ax.set_xlim(0, sl_rz.valmax)
        sl_rz.set_val(min(state["r_z"], sl_rz.valmax))
        sl_pz.valmax = max(state["p_shape"][0] - 1, 1)
        sl_pz.ax.set_xlim(0, sl_pz.valmax)
        sl_pz.set_val(min(state["p_z"], sl_pz.valmax))
        ri_art.set_clim(state["r_vmin"], state["r_vmax"])
        ro_b.set_clim(state["r_vmin"], state["r_vmax"])
        pi_art.set_clim(state["p_vmin"], state["p_vmax"])
        po_b.set_clim(state["p_vmin"], state["p_vmax"])
        suptitle(); redraw()

    def on_rz(val):
        state["r_z"] = int(val); suptitle(); redraw()

    def on_pz(val):
        state["p_z"] = int(val); suptitle(); redraw()

    sl_case.on_changed(on_case)
    sl_rz.on_changed(on_rz)
    sl_pz.on_changed(on_pz)

    def step_case(d):
        n = max(0, min(len(pairs) - 1, state["ci"] + d))
        if n != state["ci"]: sl_case.set_val(n)

    def step_z(d, which):
        key = f"{which}_z"
        skey = f"{which}_shape"
        sl = sl_rz if which == "r" else sl_pz
        n = max(0, min(state[skey][0] - 1, state[key] + d))
        if n != state[key]: sl.set_val(n)

    def on_scroll(e):
        if e.inaxes in axes[0]:
            step_z(-1 if e.button == "up" else 1, "r")
        elif e.inaxes in axes[1]:
            step_z(-1 if e.button == "up" else 1, "p")

    def on_key(e):
        if e.key == "left": step_case(-1)
        elif e.key == "right": step_case(1)
        elif e.key == "up": step_z(-1, "r"); step_z(-1, "p")
        elif e.key == "down": step_z(1, "r"); step_z(1, "p")

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


# ──────────────────────────────────────────────────────────────────────
# Menú principal
# ──────────────────────────────────────────────────────────────────────
def menu_subdataset() -> Optional[str]:
    available = []
    for sd in SUBDATASETS:
        raw_ok = (RAW_ROOT / sd).is_dir()
        prep_ok = (PREP_ROOT / sd).is_dir()
        if raw_ok or prep_ok:
            tag = []
            if raw_ok: tag.append("Raw")
            if prep_ok: tag.append("Prep")
            available.append((sd, ", ".join(tag)))
    if not available:
        print("  ⚠ No se encontraron subdatasets.")
        return None
    options = [f"{sd}  ({tags})" for sd, tags in available]
    idx = _pick(options, "Elige un subdataset")
    return available[idx][0] if idx is not None else None


def menu_source(subdataset: str) -> Optional[str]:
    options = []
    raw_cases = discover_raw_cases(subdataset)
    prep_cases = discover_prep_cases(subdataset)

    if raw_cases:
        options.append(("raw", f"Dataset (Raw TIFF)  — {len(raw_cases)} casos"))
    if prep_cases:
        options.append(("prep", f"Dataset_Preprocessed (npy)  — {len(prep_cases)} casos"))
    if raw_cases and prep_cases:
        # contar comunes
        prep_ids = {c.case_id for c in prep_cases}
        common = sum(1 for c in raw_cases if c.case_id in prep_ids)
        if common > 0:
            options.append(("compare", f"Comparar Raw vs Preprocessed  — {common} casos comunes"))

    if not options:
        print(f"  ⚠ No hay datos en {subdataset}.")
        input("  Presiona Enter …")
        return None

    idx = _pick([o[1] for o in options], f"Fuente de datos  [{subdataset}]")
    return options[idx][0] if idx is not None else None


def menu_filter_cases(case_ids: List[str], label: str) -> Optional[List[int]]:
    """Muestra la lista de casos y deja elegir uno, varios o todos."""
    options = [f"▶ TODOS ({len(case_ids)} casos)"] + case_ids
    idx = _pick(options, f"Casos disponibles — {label}")
    if idx is None:
        return None
    if idx == 0:
        return list(range(len(case_ids)))
    return [idx - 1]


def main() -> None:
    print("\n  ╔══════════════════════════════════════════════╗")
    print("  ║   Visor de Volúmenes DBT                    ║")
    print("  ║   Dataset (Raw) & Dataset_Preprocessed      ║")
    print("  ╚══════════════════════════════════════════════╝\n")

    while True:
        # 1) Elegir subdataset
        subdataset = menu_subdataset()
        if subdataset is None:
            print("\n  ¡Hasta luego!")
            break

        # 2) Elegir fuente
        source = menu_source(subdataset)
        if source is None:
            continue

        if source == "raw":
            cases = discover_raw_cases(subdataset)
            sel = menu_filter_cases([c.case_id for c in cases], f"Raw — {subdataset}")
            if sel is None:
                continue
            chosen = [cases[i] for i in sel]
            print(f"\n  Abriendo visor con {len(chosen)} caso(s) …\n")
            view_single(
                chosen,
                load_img_fn=lambda c: load_tiff_volume(c.img_tiff),
                load_mask_fn=lambda c: load_tiff_volume(c.mask_tiff),
            )

        elif source == "prep":
            cases = discover_prep_cases(subdataset)
            sel = menu_filter_cases([c.case_id for c in cases], f"Preprocessed — {subdataset}")
            if sel is None:
                continue
            chosen = [cases[i] for i in sel]
            print(f"\n  Abriendo visor con {len(chosen)} caso(s) …\n")
            view_single(
                chosen,
                load_img_fn=lambda c: load_npy_volume(c.img_npy),
                load_mask_fn=lambda c: load_npy_volume(c.mask_npy),
            )

        elif source == "compare":
            raw_cases = discover_raw_cases(subdataset)
            prep_cases = discover_prep_cases(subdataset)
            prep_dict = {c.case_id: c for c in prep_cases}
            common_raw = [r for r in raw_cases if r.case_id in prep_dict]
            common_prep = [prep_dict[r.case_id] for r in common_raw]

            sel = menu_filter_cases(
                [c.case_id for c in common_raw],
                f"Comparar — {subdataset}",
            )
            if sel is None:
                continue
            chosen_raw = [common_raw[i] for i in sel]
            chosen_prep = [common_prep[i] for i in sel]
            print(f"\n  Abriendo comparador con {len(chosen_raw)} caso(s) …\n")
            view_compare(chosen_raw, chosen_prep)


if __name__ == "__main__":
    main()

