"""
Rendimiento de los modelos preentrenados en sintetico sobre la cohorte clinica,
SIN ninguna etapa de adaptacion (zero-shot).

Los registros de fine-tuning solo persistieron la media agregada, asi que los
valores por caso no existian. Esto es re-inferencia desde los checkpoints
sinteticos archivados, no reentrenamiento.

La inferencia replica la de ensemble/_gen_custom_probs.py (mismo padding a
multiplo de 16, mismo recorte posterior) y las metricas usan la misma funcion
que el resto del analisis, para que los numeros sean comparables 1:1 con los de
las tablas del manuscrito.

Uso:  python src/eval_zeroshot_clinical.py
"""
import importlib.util
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{REPO}/ensemble")
from _ablation_common import metrics, TEST, load_mask          # noqa: E402

DATA = os.environ.get("DBT_DATA",
                      f"{REPO}/data/Dataset_Preprocessed/Dataset_Hybrid")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THR = 0.5

NETS = {
    "Attention U-Net": (f"{REPO}/src/models/attention_unet3d.py", "AttentionUNet3D",
                        dict(in_ch=1, base_ch=32, levels=4, use_checkpointing=True),
                        f"{REPO}/models/synthetic_base/Attention_UNet_Both_best.pt"),
    "3D U-Net":        (f"{REPO}/src/models/unet3d.py", "UNet3D",
                        dict(in_ch=1, base_ch=32, levels=4),
                        f"{REPO}/models/synthetic_base/UNet_BCE_Both_best.pt"),
}


def pad16(x, m=16):
    D, H, W = x.shape[-3:]
    ru = lambda v: ((v + m - 1) // m) * m
    return torch.nn.functional.pad(x, (0, ru(W)-W, 0, ru(H)-H, 0, ru(D)-D))


def load_net(file, cls, kw, ckpt):
    spec = importlib.util.spec_from_file_location("m", file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    net = getattr(mod, cls)(**kw).to(DEVICE)
    state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    net.load_state_dict(state["model_state"] if "model_state" in state else state)
    net.eval()
    return net


@torch.no_grad()
def predict(net, case):
    raw = np.load(f"{DATA}/{case}/{case}_img.npy").astype(np.float32)
    D, H, W = raw.shape
    out = net(pad16(torch.from_numpy(raw[None, None]).to(DEVICE)))
    return out[0, 0, :D, :H, :W].cpu().numpy()


def main():
    out = {"threshold": THR, "n_cases": len(TEST), "networks": {}}
    for name, (file, cls, kw, ckpt) in NETS.items():
        net = load_net(file, cls, kw, ckpt)
        per_case = {}
        for c in TEST:
            case = f"real_dbt_{c}"
            p = predict(net, case)
            per_case[case] = metrics(p >= THR, load_mask(c))
        del net
        torch.cuda.empty_cache()

        agg = {k: float(np.mean([v[k] for v in per_case.values()]))
               for k in ("dice", "iou", "precision", "recall")}
        agg["n_zero_dice"] = int(sum(v["dice"] == 0 for v in per_case.values()))
        out["networks"][name] = {"checkpoint": os.path.basename(ckpt),
                                 "mean": agg, "per_case": per_case}
        print(f"{name:<16} Dice medio = {agg['dice']:.4f}  "
              f"ceros = {agg['n_zero_dice']}/{len(TEST)}", flush=True)

    out_dir = f"{REPO}/results/statistics/tables"
    os.makedirs(out_dir, exist_ok=True)
    dst = f"{out_dir}/zeroshot_clinical.json"
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
