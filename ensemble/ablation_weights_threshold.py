"""
Ablation study del ensemble — SOLO SOBRE EL CONJUNTO DE DESARROLLO (OOF).

Responde a la "Required experimental revision" de los revisores:

  A1  Rejilla obligatoria: 13 esquemas de pesos x 3 umbrales (0.3, 0.5, 0.6)
      - mapas de probabilidad individuales (nn / att / bce)
      - ensembles por pares (nn+att, nn+bce, att+bce), peso igual
      - pesos iguales 1:1:1
      - el esquema seleccionado 2:2:3
      - alternativas 2:1:1, 1:2:1, 1:1:2, 3:2:2, 2:3:2
  A2  Rejilla fina complementaria: pesos en {0,1,2,3}^3 (49 unicos tras colapsar
      multiplos escalares) x umbrales 0.05..0.95 -> 931 configuraciones.
  A3  Verificacion del ranking de la configuracion declarada (2:2:3 @ 0.3).
  A4  Estabilidad de la seleccion: bootstrap y leave-one-case-out sobre dev.
  A5  Reglas de agregacion alternativas.

RESTRICCION DE DISENO (exigida por el revisor): este script no accede en ningun
momento a las probabilidades del conjunto de test. Solo lee data_probs/*_OOF_dev/.
La evaluacion en test se hace una unica vez, para la configuracion seleccionada
aqui, mediante ensemble/evaluate_selected_config.py.

Uso:  python ensemble/ablation_weights_threshold.py --out results/ablation
"""
import argparse
import csv
import json
import os
import sys
from itertools import product
from math import gcd

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ablation_common import (DEV, MEMBERS, METRIC_KEYS, bootstrap_ci,
                              fmt_weights, load_split)

# El split esta fijado a nivel de modulo: este script no puede evaluar test.
SPLIT = "dev"

MANDATORY_THRESHOLDS = [0.3, 0.5, 0.6]
FINE_THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05 .. 0.95
DECLARED_WEIGHTS = (2, 2, 3)
DECLARED_THR = 0.3

# A1 — rejilla obligatoria, en el orden en que la pide el revisor
MANDATORY_GRID = [
    ("nnU-Net alone",            (1, 0, 0), "single-model"),
    ("Attention U-Net alone",    (0, 1, 0), "single-model"),
    ("3D U-Net alone",           (0, 0, 1), "single-model"),
    ("nnU-Net + Attention",      (1, 1, 0), "pairwise"),
    ("nnU-Net + 3D U-Net",       (1, 0, 1), "pairwise"),
    ("Attention + 3D U-Net",     (0, 1, 1), "pairwise"),
    ("Equal weights",            (1, 1, 1), "equal"),
    ("Selected scheme",          (2, 2, 3), "selected"),
    ("Alternative",              (2, 1, 1), "alternative"),
    ("Alternative",              (1, 2, 1), "alternative"),
    ("Alternative",              (1, 1, 2), "alternative"),
    ("Alternative",              (3, 2, 2), "alternative"),
    ("Alternative",              (2, 3, 2), "alternative"),
]


# ---------------------------------------------------------------------------
# Evaluacion multi-umbral eficiente y exacta.
#
# Para una combinacion de pesos fija, se ordenan las probabilidades de los
# voxeles positivos y negativos por separado; entonces TP/FP a cualquier umbral
# salen de una busqueda binaria, y las 19 metricas se obtienen en una pasada en
# lugar de re-binarizar el volumen 19 veces.
def _sorted_pools(prob, gt):
    return np.sort(prob[gt]), np.sort(prob[~gt])


def _counts_at(pos_sorted, neg_sorted, thr):
    tp = len(pos_sorted) - int(np.searchsorted(pos_sorted, thr, side="left"))
    fp = len(neg_sorted) - int(np.searchsorted(neg_sorted, thr, side="left"))
    fn = len(pos_sorted) - tp
    return tp, fp, fn


def _metrics_from_counts(tp, fp, fn):
    """Identico convenio que _ablation_common.metrics (degenerado -> 1.0)."""
    pred_sum = tp + fp
    gt_sum = tp + fn
    union = tp + fp + fn
    return dict(
        dice=1.0 if (pred_sum + gt_sum) == 0 else 2 * tp / (pred_sum + gt_sum),
        iou=1.0 if union == 0 else tp / union,
        precision=1.0 if pred_sum == 0 else tp / pred_sum,
        recall=1.0 if gt_sum == 0 else tp / gt_sum,
    )


def eval_weights_all_thresholds(probs, masks, weights, thresholds):
    """{thr: {case: metrics}} para una combinacion de pesos."""
    out = {t: {} for t in thresholds}
    for c in DEV:
        num, den = None, 0.0
        for m, w in zip(MEMBERS, weights):
            if w == 0:
                continue
            num = w * probs[m][c] if num is None else num + w * probs[m][c]
            den += w
        pos, neg = _sorted_pools(num / den, masks[c])
        for t in thresholds:
            out[t][c] = _metrics_from_counts(*_counts_at(pos, neg, t))
    return out


def summarize_run(per_case, n_boot, seed):
    """Agrega y anade IC bootstrap del Dice medio."""
    row = {}
    for k in METRIC_KEYS:
        v = np.array([per_case[c][k] for c in DEV], dtype=float)
        row[f"{k}_mean"] = float(v.mean())
        row[f"{k}_sd"] = float(v.std(ddof=1))
        row[f"{k}_median"] = float(np.median(v))
    dice = np.array([per_case[c]["dice"] for c in DEV], dtype=float)
    lo, hi = bootstrap_ci(dice, n_boot=n_boot, seed=seed)
    row["dice_ci_lo"], row["dice_ci_hi"] = lo, hi
    row["n_zero_dice"] = int((dice < 1e-6).sum())
    return row


def unique_weight_tuples():
    """Pesos en {0,1,2,3}^3 colapsando multiplos escalares: 63 -> 49."""
    seen = {}
    for w in product(range(4), repeat=3):
        if sum(w) == 0:
            continue
        g = 0
        for x in w:
            g = gcd(g, x)
        seen.setdefault(tuple(x // g for x in w), None)
    return sorted(seen)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/ablation")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    tables = f"{args.out}/tables"
    os.makedirs(tables, exist_ok=True)

    print(f"Cargando probabilidades OOF de desarrollo ({len(DEV)} casos)...")
    probs, masks = load_split(SPLIT)

    # --- A2: rejilla fina (superconjunto de A1) --------------------------
    tuples = unique_weight_tuples()
    all_thr = sorted(set(FINE_THRESHOLDS) | set(MANDATORY_THRESHOLDS))
    print(f"A2: {len(tuples)} combinaciones de pesos x {len(all_thr)} umbrales "
          f"= {len(tuples) * len(all_thr)} configuraciones (solo dev)...")

    results = {}      # (weights, thr) -> per_case metrics
    for i, w in enumerate(tuples, 1):
        for thr, per_case in eval_weights_all_thresholds(
                probs, masks, w, all_thr).items():
            results[(w, thr)] = per_case
        if i % 10 == 0 or i == len(tuples):
            print(f"  {i}/{len(tuples)} combinaciones de pesos")

    summaries = {key: summarize_run(pc, args.n_boot, args.seed)
                 for key, pc in results.items()}

    fine_rows = []
    for (w, thr), s in sorted(summaries.items(),
                              key=lambda kv: -kv[1]["dice_mean"]):
        fine_rows.append(dict(weights=fmt_weights(w), threshold=thr,
                              n_members=sum(1 for x in w if x > 0), **s))
    _write_csv(f"{tables}/ablation_grid_dev.csv", fine_rows, args.seed)
    print(f"  -> ablation_grid_dev.csv ({len(fine_rows)} filas)")

    # --- A1: rejilla obligatoria ----------------------------------------
    print("A1: rejilla obligatoria (13 esquemas x 3 umbrales)...")
    mand_rows = []
    for label, w, cat in MANDATORY_GRID:
        for thr in MANDATORY_THRESHOLDS:
            mand_rows.append(dict(
                configuration=label, category=cat, weights=fmt_weights(w),
                threshold=thr, **summaries[(w, thr)]))
    _write_csv(f"{tables}/table3_ensemble_ablation_dev.csv", mand_rows, args.seed)
    print(f"  -> table3_ensemble_ablation_dev.csv ({len(mand_rows)} filas)")

    # --- A3: ranking de la configuracion declarada ----------------------
    declared_key = (DECLARED_WEIGHTS, DECLARED_THR)
    ranking = _ranking_report(summaries, mand_rows, declared_key, args)
    with open(f"{tables}/ablation_ranking_check.json", "w") as f:
        json.dump(ranking, f, indent=2)
    print(f"  -> ablation_ranking_check.json")

    # --- A4: estabilidad de la seleccion --------------------------------
    print("A4: estabilidad de la seleccion (bootstrap + leave-one-out)...")
    stability = _stability(results, summaries, args)
    _write_csv(f"{tables}/ablation_stability_dev.csv",
               stability["rows"], args.seed)
    with open(f"{tables}/ablation_stability_dev.json", "w") as f:
        json.dump(stability["report"], f, indent=2)
    print(f"  -> ablation_stability_dev.csv / .json")

    # --- A5: reglas de agregacion alternativas --------------------------
    print("A5: reglas de agregacion alternativas...")
    agg_rows = _aggregation_rules(probs, masks, all_thr, args)
    _write_csv(f"{tables}/ablation_aggregation_dev.csv", agg_rows, args.seed)
    print(f"  -> ablation_aggregation_dev.csv ({len(agg_rows)} filas)")

    _print_summary(ranking, mand_rows)


# ---------------------------------------------------------------------------
def _ranking_report(summaries, mand_rows, declared_key, args):
    """Posicion de 2:2:3 @ 0.3 en la rejilla obligatoria y en la fina."""
    def rank_within(keys):
        ordered = sorted(keys, key=lambda k: -summaries[k]["dice_mean"])
        return ordered, ordered.index(declared_key) + 1

    mand_keys = [(w, t) for _, w, _ in MANDATORY_GRID for t in MANDATORY_THRESHOLDS]
    mand_order, mand_rank = rank_within(mand_keys)
    fine_order, fine_rank = rank_within(list(summaries))

    decl = summaries[declared_key]
    best_m, best_f = mand_order[0], fine_order[0]

    def describe(key):
        s = summaries[key]
        return dict(weights=fmt_weights(key[0]), threshold=key[1],
                    dice_mean=s["dice_mean"], dice_sd=s["dice_sd"],
                    dice_ci=[s["dice_ci_lo"], s["dice_ci_hi"]],
                    n_zero_dice=s["n_zero_dice"])

    # Solapamiento de IC entre la declarada y el mejor de cada rejilla
    def overlaps(a, b):
        sa, sb = summaries[a], summaries[b]
        return not (sa["dice_ci_hi"] < sb["dice_ci_lo"]
                    or sb["dice_ci_hi"] < sa["dice_ci_lo"])

    return dict(
        declared=describe(declared_key),
        mandatory_grid=dict(n_configs=len(mand_keys), declared_rank=mand_rank,
                            best=describe(best_m),
                            ci_overlap_with_best=overlaps(declared_key, best_m),
                            top5=[describe(k) for k in mand_order[:5]]),
        fine_grid=dict(n_configs=len(fine_order), declared_rank=fine_rank,
                       best=describe(best_f),
                       ci_overlap_with_best=overlaps(declared_key, best_f),
                       top5=[describe(k) for k in fine_order[:5]]),
        note=("Todos los valores provienen de predicciones out-of-fold del "
              "conjunto de desarrollo (n=10). El conjunto de test no se ha "
              "utilizado en ningun punto de este analisis."),
    )


def _stability(results, summaries, args):
    """Bootstrap y leave-one-out de la seleccion sobre la rejilla obligatoria."""
    mand_keys = [(w, t) for _, w, _ in MANDATORY_GRID for t in MANDATORY_THRESHOLDS]
    dice = {k: np.array([results[k][c]["dice"] for c in DEV]) for k in mand_keys}
    mat = np.stack([dice[k] for k in mand_keys])          # (n_config, n_cases)

    rng = np.random.default_rng(args.seed)
    n_boot = 1000
    idx = rng.integers(0, len(DEV), size=(n_boot, len(DEV)))
    boot_means = mat[:, idx].mean(axis=2)                 # (n_config, n_boot)
    winners = boot_means.argmax(axis=0)
    win_count = np.bincount(winners, minlength=len(mand_keys))

    # Frecuencia con que la declarada queda en el top-5% de cada remuestreo
    declared_i = mand_keys.index((DECLARED_WEIGHTS, DECLARED_THR))
    ranks = (boot_means > boot_means[declared_i]).sum(axis=0) + 1
    top_frac = float((ranks <= max(1, round(0.05 * len(mand_keys)))).mean())

    # Leave-one-case-out
    loo = []
    for j, case in enumerate(DEV):
        keep = [i for i in range(len(DEV)) if i != j]
        best = int(mat[:, keep].mean(axis=1).argmax())
        w, t = mand_keys[best]
        loo.append(dict(left_out=f"real_dbt_{case}",
                        selected_weights=fmt_weights(w), selected_threshold=t))

    rows = []
    for i, (w, t) in enumerate(mand_keys):
        rows.append(dict(weights=fmt_weights(w), threshold=t,
                         dice_mean=summaries[(w, t)]["dice_mean"],
                         bootstrap_win_count=int(win_count[i]),
                         bootstrap_win_frac=float(win_count[i] / n_boot)))
    rows.sort(key=lambda r: -r["bootstrap_win_frac"])

    three_member = [k for k in mand_keys if all(k[0])]
    env = [summaries[k]["dice_mean"] for k in three_member]

    return dict(rows=rows, report=dict(
        n_bootstrap=n_boot, seed=args.seed,
        declared=dict(weights=fmt_weights(DECLARED_WEIGHTS), threshold=DECLARED_THR,
                      bootstrap_win_frac=float(win_count[declared_i] / n_boot),
                      bootstrap_top5pct_frac=top_frac,
                      median_bootstrap_rank=float(np.median(ranks))),
        leave_one_out=loo,
        n_distinct_loo_selections=len({(d["selected_weights"],
                                        d["selected_threshold"]) for d in loo}),
        three_member_dev_envelope=dict(min=float(min(env)), max=float(max(env)),
                                       n_configs=len(three_member)),
    ))


def _aggregation_rules(probs, masks, thresholds, args):
    """Reglas de fusion alternativas, con umbral seleccionado en dev."""
    from _ablation_common import metrics as _metrics

    w = DECLARED_WEIGHTS

    def weighted(c):
        num = sum(wi * probs[m][c] for m, wi in zip(MEMBERS, w))
        return num / sum(w)

    def unweighted(c):
        return sum(probs[m][c] for m in MEMBERS) / 3.0

    def logit_mean(c):
        eps = 1e-6
        z = [np.log(np.clip(probs[m][c], eps, 1 - eps)
                    / (1 - np.clip(probs[m][c], eps, 1 - eps))) for m in MEMBERS]
        return 1.0 / (1.0 + np.exp(-sum(z) / 3.0))

    def vmax(c):
        return np.maximum.reduce([probs[m][c] for m in MEMBERS])

    def vmin(c):
        return np.minimum.reduce([probs[m][c] for m in MEMBERS])

    rules = [("Weighted mean (2:2:3)", weighted),
             ("Unweighted mean (1:1:1)", unweighted),
             ("Logit (log-odds) mean", logit_mean),
             ("Voxel-wise maximum", vmax),
             ("Voxel-wise minimum", vmin)]

    rows = []
    for name, fn in rules:
        pools = {c: _sorted_pools(fn(c), masks[c]) for c in DEV}
        best = None
        for thr in thresholds:
            per_case = {c: _metrics_from_counts(*_counts_at(*pools[c], thr))
                        for c in DEV}
            s = summarize_run(per_case, args.n_boot, args.seed)
            if best is None or s["dice_mean"] > best[1]["dice_mean"]:
                best = (thr, s)
        rows.append(dict(rule=name, selected_threshold=best[0], **best[1]))

    # Voto por mayoria: >=2 de 3 redes activas al umbral estandar de 0.5
    per_case = {}
    for c in DEV:
        votes = sum((probs[m][c] >= 0.5).astype(np.uint8) for m in MEMBERS)
        per_case[c] = _metrics(votes >= 2, masks[c])
    rows.append(dict(rule="Majority vote (2/3 at 0.5)", selected_threshold="n/a",
                     **summarize_run(per_case, args.n_boot, args.seed)))

    # Post-proceso: mayor componente conexa sobre el ensemble seleccionado
    try:
        from scipy.ndimage import label
        per_case = {}
        for c in DEV:
            m = weighted(c) >= DECLARED_THR
            lab, n = label(m)
            if n > 0:
                sizes = np.bincount(lab.ravel())
                sizes[0] = 0
                m = lab == sizes.argmax()
            per_case[c] = _metrics(m, masks[c])
        rows.append(dict(rule="Weighted mean + largest connected component",
                         selected_threshold=DECLARED_THR,
                         **summarize_run(per_case, args.n_boot, args.seed)))
    except ImportError:
        pass

    return rows


def _write_csv(path, rows, seed):
    if not rows:
        return
    fields = list(rows[0]) + ["split", "bootstrap_seed"]
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            wr.writerow({**r, "split": "development (out-of-fold)",
                         "bootstrap_seed": seed})


def _print_summary(ranking, mand_rows):
    d = ranking["declared"]
    print("\n" + "=" * 72)
    print("VERIFICACION DE LA CONFIGURACION DECLARADA  (2:2:3 @ 0.3)")
    print("=" * 72)
    print(f"  Dice dev = {d['dice_mean']:.4f} +/- {d['dice_sd']:.4f}  "
          f"IC95% [{d['dice_ci'][0]:.4f}, {d['dice_ci'][1]:.4f}]  "
          f"ceros = {d['n_zero_dice']}/10")
    for grid, label in [("mandatory_grid", "rejilla obligatoria (39 configs)"),
                        ("fine_grid", "rejilla fina (931 configs)")]:
        g = ranking[grid]
        b = g["best"]
        print(f"\n  {label}:")
        print(f"    ranking de la declarada : #{g['declared_rank']} de {g['n_configs']}")
        print(f"    mejor en dev            : {b['weights']} @ {b['threshold']}  "
              f"Dice = {b['dice_mean']:.4f}")
        print(f"    IC se solapa con la mejor: "
              f"{'SI' if g['ci_overlap_with_best'] else 'NO'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
