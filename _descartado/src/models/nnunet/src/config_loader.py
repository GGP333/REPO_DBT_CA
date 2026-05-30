# src/config_loader.py
from __future__ import annotations
import os
import yaml
from types import SimpleNamespace
from typing import Any, Dict

def _as_namespace(d: Dict[str, Any]) -> SimpleNamespace:
    """Convierte un dict recursivo a SimpleNamespace."""
    ns = SimpleNamespace(**{k: _as_namespace(v) if isinstance(v, dict) else v for k, v in d.items()})
    return ns

def load_config(path: str = "configs/config.yaml") -> SimpleNamespace:
    """
    Carga y valida el archivo YAML de configuración.
    
    Parameters
    ----------
    path : str
        Ruta al archivo YAML.
    
    Returns
    -------
    SimpleNamespace
        Objeto de configuración con atributos accesibles por punto.
    
    Raises
    ------
    FileNotFoundError
        Si no existe el YAML o la ruta dataset_root.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No se encontró config YAML en: {path}")
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    cfg = _as_namespace(raw)

    # Validaciones básicas
    if not hasattr(cfg, "dataset_root"):
        raise ValueError("Falta 'dataset_root' en config.")
    dataset_root = cfg.dataset_root
    if not isinstance(dataset_root, str):
        raise ValueError("'dataset_root' debe ser string.")
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(
            f"No existe dataset_root: {dataset_root} (recuerda que la ruta contiene un espacio y debe ir entre comillas)."
        )

    # Salidas
    outputs_root = "outputs"
    subdirs = ["checkpoints", "logs", "visualizations", "preds_npy"]
    for sd in subdirs:
        os.makedirs(os.path.join(outputs_root, sd), exist_ok=True)

    return cfg
