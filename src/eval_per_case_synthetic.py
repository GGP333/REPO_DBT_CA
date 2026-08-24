"""
Recupera metricas POR CASO en los conjuntos de test de los cuatro datasets, para
el 3D U-Net (BCE) y el Attention U-Net.

Motivacion: los entrenamientos originales solo persistieron la media agregada en
logs/test_metrics.json, de modo que la Tabla 1 del manuscrito no puede acompanarse
de desviaciones estandar ni intervalos de confianza. Los checkpoints de las ocho
corridas si se conservan, asi que basta re-inferir: no se reentrena nada.

La reproduccion es deliberadamente fiel al pipeline original
(src/models/*/src/train_eval.py::validate_one_epoch):

  - mismos IDs de test, leidos del run_config.json de cada corrida;
  - mismo Dataset y mismo collate_pad, que rellena cada volumen a multiplos de 16
    (Z: 15 -> 16). Las metricas se calculan sobre el volumen RELLENADO, igual que
    en el entrenamiento, porque de lo contrario no coincidirian;
  - mismo compute_metrics y mismo umbral 0.5.

Criterio de aceptacion: la media recomputada debe coincidir con el
logs/test_metrics.json publicado. Si no coincide, el checkpoint o el preprocesado
no son los del manuscrito y hay que detenerse, no ajustar el numero.

Sobre la tolerancia: la coincidencia no puede ser bit a bit. La inferencia corre
en float32 sobre cuDNN, cuya seleccion de algoritmo no es determinista entre
ejecuciones ni entre versiones de la libreria. Se comprobo ademas que la corrida
original uso TF32 activado (el valor por defecto de cudnn.allow_tf32): forzar
TF32 = False aleja el resultado en lugar de acercarlo, asi que aqui se conservan
los ajustes por defecto. En la practica las metricas de solapamiento reproducen
hasta ~1e-5 y las de distancia hasta ~0.2 voxeles sobre valores de decenas.

Uso:
  python src/eval_per_case_synthetic.py --ckpt-root /home/gabriel/Escritorio/Paper_DBT
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = f"{REPO}/data/Dataset_Preprocessed"

# Nombres de carpeta antiguos en el arbol de checkpoints:
#   Both = Mixed_Size,  Both_RealWorld = Hybrid
RUNS = [
    # (modelo, dataset, run_dir actual, subarbol y nombre antiguo del checkpoint)
    ("UNet_BCE", "large_tumor",
     "results/outputs_clean/UNet_BCE_Dataset_large_tumor",
     "outputs_clean/UNet_BCE_Dataset_large_tumor"),
    ("UNet_BCE", "small_tumor",
     "results/outputs_clean/UNet_BCE_Dataset_small_tumor",
     "outputs_clean/UNet_BCE_Dataset_small_tumor"),
    ("UNet_BCE", "Mixed_Size",
     "results/outputs_clean/UNet_BCE_Dataset_Mixed_Size",
     "outputs_clean/UNet_BCE_Dataset_Both"),
    ("UNet_BCE", "Hybrid",
     "results/outputs_improved/UNet_BCE_Dataset_Hybrid",
     "outputs_improved/UNet_BCE_Dataset_Both_RealWorld"),
    ("Attention_UNet", "large_tumor",
     "results/outputs_improved/Attention_UNet_Dataset_large_tumor",
     "outputs_improved/Attention_UNet_Dataset_large_tumor"),
    ("Attention_UNet", "small_tumor",
     "results/outputs_improved/Attention_UNet_Dataset_small_tumor",
     "outputs_improved/Attention_UNet_Dataset_small_tumor"),
    ("Attention_UNet", "Mixed_Size",
     "results/outputs_improved/Attention_UNet_Dataset_Mixed_Size",
     "outputs_improved/Attention_UNet_Dataset_Both"),
    ("Attention_UNet", "Hybrid",
     "results/outputs_improved/Attention_UNet_Dataset_Hybrid",
     "outputs_improved/Attention_UNet_Dataset_Both_RealWorld"),
]

SRC_DIR = {"UNet_BCE": f"{REPO}/src/models/unet_bce/src",
           "Attention_UNet": f"{REPO}/src/models/attention_unet/src"}


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # Registrar antes de ejecutar: @dataclass resuelve sus anotaciones via
    # sys.modules[cls.__module__] y fallaria si el modulo aun no esta ahi.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_model(model_name, src):
    """Constructor identico al usado en el entrenamiento."""
    if model_name == "UNet_BCE":
        mod = _load_module("unet3d", f"{src}/unet3d.py")
        return mod.UNet3D(in_ch=1, base_ch=32, levels=4)
    mod = _load_module("attention_unet3d", f"{src}/attention_unet3d.py")
    return mod.AttentionUNet3D(in_ch=1, base_ch=32, levels=4,
                               dropout_rate=0.1, use_checkpointing=False)


@torch.no_grad()
def evaluate_run(model_name, dataset, run_dir, ckpt_path, device, verbose=True):
    src = SRC_DIR[model_name]
    # dataset_dbt.py importa io_utils desde su propio src, y augmentations desde
    # "shared". La ruta que calcula para shared quedo obsoleta cuando el modulo
    # paso de src/models/shared a src/shared, asi que la anadimos aqui en vez de
    # modificar el codigo original de entrenamiento.
    sys.path.insert(0, src)
    sys.path.insert(0, f"{REPO}/src/shared")
    ds_mod = _load_module("dataset_dbt", f"{src}/dataset_dbt.py")
    me_mod = _load_module("metrics", f"{src}/metrics.py")

    cfg = json.load(open(f"{REPO}/{run_dir}/run_config.json"))
    test_ids = list(cfg["dataset"]["test"]["study_ids"])
    threshold = float(cfg["hyperparameters"]["threshold"])
    expected_params = int(cfg["model"]["total_params"])

    root = f"{DATA}/Dataset_{dataset}"
    records = {r.study_id: r for r in ds_mod.discover_studies(root)}
    missing = [i for i in test_ids if i not in records]
    if missing:
        raise RuntimeError(f"{model_name}/{dataset}: faltan estudios {missing}")
    recs = [records[i] for i in sorted(test_ids)]

    ds = ds_mod.DBTVolumeDataset(recs, augment=False, cfg=None)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=ds_mod.collate_pad)

    model = build_model(model_name, src)
    n_params = sum(p.numel() for p in model.parameters())
    if n_params != expected_params:
        raise RuntimeError(f"{model_name}/{dataset}: el modelo reconstruido tiene "
                           f"{n_params} parametros, run_config declara {expected_params}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.to(device).eval()

    per_case = {}
    for batch in loader:
        imgs = batch["image"].to(device)
        masks = batch["mask"].to(device)
        probs = model(imgs)
        # Metricas sobre el volumen rellenado, igual que validate_one_epoch
        per_case[batch["study_ids"][0]] = me_mod.compute_metrics(
            probs, masks, threshold=threshold)

    keys = list(next(iter(per_case.values())))
    mean = {k: float(np.mean([per_case[c][k] for c in per_case])) for k in keys}

    del model
    torch.cuda.empty_cache()
    sys.path.remove(src)
    sys.path.remove(f"{REPO}/src/shared")
    for m in ("dataset_dbt", "metrics", "unet3d", "attention_unet3d"):
        sys.modules.pop(m, None)

    return dict(mean=mean, per_case=per_case, n_cases=len(per_case),
                threshold=threshold, checkpoint=ckpt_path)


# Las metricas se agrupan por como se comportan ante una perturbacion minima de
# la mascara binaria, que es lo que produce el no determinismo de cuDNN:
#   overlap  -> continuas y acotadas en [0,1]; deben reproducir casi exactamente.
#              Son las cuatro que reporta la Tabla 1 del manuscrito.
#   detection-> continuas pero de magnitud pequena; se comparan en relativo.
#   distance -> Hausdorff y HD95 son DISCONTINUAS: un unico voxel que cruza el
#              umbral puede desplazarlas un voxel entero. No admiten una
#              tolerancia estrecha por su propia definicion.
OVERLAP_METRICS = ["dice", "iou", "precision", "recall", "f1", "accuracy"]
DISTANCE_METRICS = ["hausdorff", "hausdorff_95"]


def check_reproduction(recomputed, published, tol_overlap, tol_rel,
                       tol_distance_vox=1.0, tol_distance_rel=0.05):
    """Compara la media recomputada con la publicada, por clase de metrica."""
    worst = {"overlap": (0.0, None), "detection": (0.0, None),
             "distance": (0.0, None)}
    for k, pub in published.items():
        if k not in recomputed:
            continue
        d = abs(recomputed[k] - pub)
        if k in OVERLAP_METRICS:
            cls, val = "overlap", d
        elif k in DISTANCE_METRICS:
            cls, val = "distance", d
        else:
            cls, val = "detection", d / max(abs(pub), 1e-6)
        if val > worst[cls][0]:
            worst[cls] = (val, k)

    ok_ov = worst["overlap"][0] <= tol_overlap
    ok_det = worst["detection"][0] <= tol_rel
    # Distancias: se acepta hasta un voxel, o el 5 % del valor publicado
    dist_allow = max(tol_distance_vox,
                     tol_distance_rel * abs(published.get("hausdorff_95", 0.0)))
    ok_dist = worst["distance"][0] <= dist_allow

    return (ok_ov and ok_det and ok_dist), dict(
        published_dice=published["dice"], recomputed_dice=recomputed["dice"],
        worst_overlap_abs_diff=worst["overlap"][0],
        worst_overlap_metric=worst["overlap"][1],
        worst_detection_rel_diff=worst["detection"][0],
        worst_detection_metric=worst["detection"][1],
        worst_distance_abs_diff=worst["distance"][0],
        worst_distance_metric=worst["distance"][1],
        tol_overlap_abs=tol_overlap, tol_detection_rel=tol_rel,
        tol_distance_abs=dist_allow,
        passed_overlap=bool(ok_ov), passed_detection=bool(ok_det),
        passed_distance=bool(ok_dist),
        passed=bool(ok_ov and ok_det and ok_dist),
        note=("La coincidencia no puede ser bit a bit: la inferencia float32 "
              "sobre cuDNN no es determinista entre ejecuciones ni versiones de "
              "la libreria. Se conservan los ajustes por defecto (TF32 activado), "
              "que son los de la corrida original; forzar TF32=False aleja el "
              "resultado. Hausdorff y HD95 son discontinuas respecto de la "
              "mascara binaria y por eso llevan una tolerancia de un voxel."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", default="/home/gabriel/Escritorio/Paper_DBT",
                    help="arbol que contiene outputs_clean/ y outputs_improved/ "
                         "con los checkpoints best.pt de las corridas originales")
    ap.add_argument("--tol-overlap", type=float, default=1e-3,
                    help="tolerancia absoluta para dice/iou/precision/recall/f1/accuracy")
    ap.add_argument("--tol-rel", type=float, default=0.01,
                    help="tolerancia relativa para distancias, AP y falsos positivos")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}\n")

    ok_all = True
    for model_name, dataset, run_dir, ckpt_sub in RUNS:
        ckpt = f"{args.ckpt_root}/{ckpt_sub}/checkpoints/best.pt"
        tag = f"{model_name} / {dataset}"
        if not os.path.exists(ckpt):
            print(f"[FALTA]  {tag}: no existe {ckpt}")
            ok_all = False
            continue

        res = evaluate_run(model_name, dataset, run_dir, ckpt, device)

        published = json.load(open(f"{REPO}/{run_dir}/logs/test_metrics.json"))
        ok, detail = check_reproduction(res["mean"], published,
                                        args.tol_overlap, args.tol_rel)
        ok_all &= ok

        out = f"{REPO}/{run_dir}/logs/test_metrics_per_case.json"
        res["reproduction_check"] = detail
        with open(out, "w") as f:
            json.dump(res, f, indent=2)

        print(f"[{'OK ' if ok else 'FAIL'}]  {tag:32s} n={res['n_cases']:2d}  "
              f"Dice {res['mean']['dice']:.6f} vs {published['dice']:.6f}  "
              f"| peor dif: solapam. {detail['worst_overlap_abs_diff']:.1e} abs, "
              f"detec. {detail['worst_detection_rel_diff']:.1e} rel, "
              f"dist. {detail['worst_distance_abs_diff']:.2f} vox")

    print("\n" + ("Todas las corridas reproducen el valor publicado."
                  if ok_all else
                  "ATENCION: alguna corrida NO reproduce el valor publicado."))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
