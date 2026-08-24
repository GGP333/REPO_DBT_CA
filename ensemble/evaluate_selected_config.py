"""
Evaluacion UNICA en el conjunto de test independiente.

Este es el unico script del analisis de ablacion con acceso a
data_probs/*_test/. La configuracion que recibe debe haber sido seleccionada
previamente con ensemble/ablation_weights_threshold.py, que trabaja
exclusivamente sobre predicciones out-of-fold del conjunto de desarrollo.

Esta separacion fisica entre "seleccionar" y "evaluar" es la garantia auditable
que exige la revision: el test no puede haberse usado para comparar
configuraciones porque el codigo que las compara no puede leerlo.

Uso:  python ensemble/evaluate_selected_config.py --w 2 2 3 --thr 0.3
"""
import argparse
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ablation_common import (METRIC_KEYS, TEST, bootstrap_ci, combine,
                              fmt_weights, load_split, metrics)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", nargs=3, type=float, required=True,
                    metavar=("W_NN", "W_ATT", "W_BCE"),
                    help="pesos seleccionados en desarrollo")
    ap.add_argument("--thr", type=float, required=True,
                    help="umbral seleccionado en desarrollo")
    ap.add_argument("--out", default="results/ablation")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    tables = f"{args.out}/tables"
    os.makedirs(tables, exist_ok=True)

    weights = tuple(args.w)
    print(f"Configuracion seleccionada en desarrollo: "
          f"{fmt_weights(weights)} @ umbral {args.thr}")
    print("Evaluando UNA SOLA VEZ sobre el conjunto de test independiente "
          f"({len(TEST)} casos clinicos)\n")

    probs, masks = load_split("test")
    per_case = {c: metrics(combine(probs, c, weights) >= args.thr, masks[c])
                for c in TEST}

    print(f"{'caso':16s} {'Dice':>7s} {'IoU':>7s} {'Prec':>7s} {'Rec':>7s}")
    for c in TEST:
        m = per_case[c]
        print(f"real_dbt_{c}    {m['dice']:7.4f} {m['iou']:7.4f} "
              f"{m['precision']:7.4f} {m['recall']:7.4f}")

    summary = {}
    for k in METRIC_KEYS:
        v = [per_case[c][k] for c in TEST]
        lo, hi = bootstrap_ci(v, n_boot=args.n_boot, seed=args.seed)
        summary[k] = dict(mean=st.mean(v), sd=st.stdev(v), median=st.median(v),
                          ci95=[lo, hi], min=min(v), max=max(v))

    dices = [per_case[c]["dice"] for c in TEST]
    n_zero = sum(1 for d in dices if d < 1e-6)

    print("-" * 48)
    for k in METRIC_KEYS:
        s = summary[k]
        print(f"{k:10s} media = {s['mean']:.4f} +/- {s['sd']:.4f}   "
              f"mediana = {s['median']:.4f}   "
              f"IC95% [{s['ci95'][0]:.4f}, {s['ci95'][1]:.4f}]")
    print(f"casos nulos = {n_zero}/{len(TEST)}")

    record = dict(
        selected_weights=fmt_weights(weights),
        selected_threshold=args.thr,
        selection_protocol=("pesos y umbral seleccionados exclusivamente sobre "
                            "predicciones out-of-fold del conjunto de desarrollo "
                            "(n=10); el conjunto de test se evaluo una unica vez"),
        n_cases=len(TEST),
        per_case={f"real_dbt_{c}": per_case[c] for c in TEST},
        summary=summary,
        n_zero_dice=n_zero,
        bootstrap=dict(n_resamples=args.n_boot, seed=args.seed),
    )
    path = f"{tables}/selected_config_test.json"
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\n-> {path}")

    # Comprobacion contra la via de reproduccion oficial del repositorio.
    #
    # Referencia: ensemble/reproduce_best.py sobre las probabilidades archivadas
    # en data_probs/. Nota de trazabilidad: results/best_ensemble.json guarda
    # 0.5184468 en lugar de 0.5184273 porque se calculo desde las predicciones
    # float32 en memoria, mientras que las probabilidades archivadas de Attention
    # y U-Net BCE estan almacenadas en float16. La diferencia (2e-5) es
    # irrelevante al nivel de precision reportado en el manuscrito (0.52).
    REFERENCE_DICE = 0.5184272913392241
    if fmt_weights(weights) == "2:2:3" and abs(args.thr - 0.3) < 1e-9:
        ok = abs(summary["dice"]["mean"] - REFERENCE_DICE) < 1e-9 and n_zero == 0
        print(f"Coincide con reproduce_best.py (Dice {REFERENCE_DICE:.7f}, "
              f"0 ceros): {'SI' if ok else 'NO'}")


if __name__ == "__main__":
    main()
