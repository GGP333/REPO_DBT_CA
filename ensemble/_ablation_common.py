"""
Utilidades compartidas por el ablation study del ensemble y el analisis estadistico.

Carga de probabilidades, metricas de segmentacion y bootstrap. Las metricas
replican exactamente las de ensemble/make_paper_assets.py (mismo convenio para el
caso degenerado prediccion-vacia-y-GT-vacio = 1.0) para que todos los numeros del
manuscrito sean comparables 1:1.
"""
import os
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DP = f"{REPO}/data_probs"
GT = os.environ.get("DBT_DATA", f"{REPO}/data/real")

# Splits del paper (fijos). El 5-fold sobre dev genero las predicciones out-of-fold.
DEV = ["001", "002", "003", "004", "008", "009", "011", "012", "017", "018"]
TEST = ["005", "006", "007", "010", "013", "014", "015", "016", "019", "020"]
FOLDS = {0: ["001", "012"], 1: ["004", "018"], 2: ["008", "011"],
         3: ["002", "017"], 4: ["003", "009"]}

MEMBERS = ["nn", "att", "bce"]
MEMBER_LABEL = {"nn": "nnU-Net", "att": "Attention U-Net", "bce": "3D U-Net (BCE)"}

# Nombre de la carpeta de probabilidades por split
_SPLIT_DIR = {"test": "test", "dev": "OOF_dev"}


# ---------------------------------------------------------------------------
# Carga
def load_prob(member, case, split):
    """Probabilidad de tumor por voxel, (Z,Y,X) float32."""
    sub = _SPLIT_DIR[split]
    if member == "nn":
        # nnU-Net guarda (2,Z,Y,X); canal 1 = tumor
        return np.load(f"{DP}/nnunet_{sub}/real_dbt_{case}.npz")["probabilities"][1].astype(np.float32)
    net = {"att": "attention", "bce": "unet_bce"}[member]
    return np.load(f"{DP}/{net}_{sub}/real_dbt_{case}.npz")["prob"].astype(np.float32)


def load_mask(case):
    return np.load(f"{GT}/real_dbt_{case}/real_dbt_{case}_mask.npy") > 0


def load_split(split, cases=None):
    """Precarga probs + GT de un split. Devuelve (probs, masks).

    probs[member][case] -> array (Z,Y,X); masks[case] -> array bool.
    Los 10 volumenes son de 15x256x512, asi que las tres redes + GT caben
    holgadamente en RAM (~250 MB) y el barrido se vuelve puramente aritmetico.
    """
    cases = cases or (DEV if split == "dev" else TEST)
    masks = {c: load_mask(c) for c in cases}
    probs = {m: {c: load_prob(m, c, split) for c in cases} for m in MEMBERS}
    for m in MEMBERS:
        for c in cases:
            if probs[m][c].shape != masks[c].shape:
                raise ValueError(
                    f"shape mismatch en {m}/real_dbt_{c}: "
                    f"{probs[m][c].shape} vs GT {masks[c].shape}")
    return probs, masks


# ---------------------------------------------------------------------------
# Metricas
def metrics(pred, gt):
    """Dice, IoU, precision y recall de una mascara binaria contra el GT."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    union = tp + fp + fn
    denom = pred.sum() + gt.sum()
    return dict(
        dice=1.0 if denom == 0 else 2 * tp / denom,
        iou=1.0 if union == 0 else tp / union,
        precision=1.0 if (tp + fp) == 0 else tp / (tp + fp),
        recall=1.0 if (tp + fn) == 0 else tp / (tp + fn),
    )


METRIC_KEYS = ["dice", "iou", "precision", "recall"]


def combine(probs, case, weights):
    """Promedio ponderado de las probabilidades de los miembros con peso > 0."""
    num = None
    den = 0.0
    for m, w in zip(MEMBERS, weights):
        if w == 0:
            continue
        p = probs[m][case]
        num = w * p if num is None else num + w * p
        den += w
    if num is None:
        raise ValueError("todos los pesos son cero")
    return num / den


def evaluate_config(probs, masks, cases, weights, thr):
    """Metricas por caso de una configuracion (pesos, umbral)."""
    return {c: metrics(combine(probs, c, weights) >= thr, masks[c]) for c in cases}


def summarize(per_case, cases):
    """Agrega metricas por caso: media, SD, mediana, min, max y casos nulos."""
    out = {}
    for k in METRIC_KEYS:
        v = np.array([per_case[c][k] for c in cases], dtype=float)
        out[f"{k}_mean"] = float(v.mean())
        out[f"{k}_sd"] = float(v.std(ddof=1))
        out[f"{k}_median"] = float(np.median(v))
        out[f"{k}_min"] = float(v.min())
        out[f"{k}_max"] = float(v.max())
    dices = np.array([per_case[c]["dice"] for c in cases], dtype=float)
    out["n_zero_dice"] = int((dices < 1e-6).sum())
    out["n_cases"] = len(cases)
    return out


# ---------------------------------------------------------------------------
# Bootstrap
def bootstrap_ci(values, n_boot=10000, seed=42, alpha=0.05, stat=np.mean):
    """IC percentil por bootstrap sobre los casos (remuestreo con reemplazo)."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot = stat(values[idx], axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def fmt_weights(weights):
    """(2,2,3) -> '2:2:3'."""
    return ":".join(str(int(w)) for w in weights)
