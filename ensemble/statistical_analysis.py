"""
Analisis estadistico de la revision (comentario 4 de los revisores).

  B1  Descriptivos de la cohorte clinica de test: media +/- DE, mediana [RIC],
      IC 95 % bootstrap percentil y rango. Alimenta la Tabla 2 revisada.
  B2  Comparaciones pareadas entre metodos sobre los mismos 10 volumenes:
      Friedman omnibus, Wilcoxon de rangos con signo del ensemble frente a cada
      modelo individual, correccion de Holm-Bonferroni sobre las tres
      comparaciones preespecificadas, tamano del efecto rank-biserial y
      diferencia mediana con IC bootstrap.
  B3  Descriptivos por dataset sintetico e hibrido para las tres arquitecturas
      (Tabla 1), a partir de las metricas por caso recuperadas.
  B4  Complementariedad entre arquitecturas: Spearman entre los Dice por caso y
      recuento de casos en que el ensemble supera al mejor modelo individual.

Nota sobre potencia: con n = 10 pares el p-valor bilateral mas pequeno que puede
alcanzar Wilcoxon es 2/2^10 ~ 0.002. El analisis cuantifica la consistencia de
las tendencias observadas; no es confirmatorio.

Uso:  python ensemble/statistical_analysis.py --out results/statistics
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ablation_common import bootstrap_ci

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

METHODS = ["nnU-Net", "Attention", "U-Net BCE", "Ensemble"]
REFERENCE = "Ensemble"
COMPARATORS = ["nnU-Net", "Attention", "U-Net BCE"]
METRICS = ["Dice", "IoU", "Precision", "Recall"]
PRIMARY = "Dice"

# Corridas sinteticas: (dataset, modelo, fichero de metricas por caso)
SYNTHETIC_RUNS = [
    ("large_tumor", "nnU-Net",
     "experiments/nnunet_official/eval/results_Dataset001_DBTLarge__nnUNetTrainer_Seeded42.json"),
    ("small_tumor", "nnU-Net",
     "experiments/nnunet_official/eval/results_Dataset002_DBTSmall__nnUNetTrainer_Seeded42.json"),
    ("Mixed_Size", "nnU-Net",
     "experiments/nnunet_official/eval/results_Dataset003_DBTMixedSize__nnUNetTrainer_Seeded42.json"),
    ("Hybrid", "nnU-Net",
     "experiments/nnunet_official/eval/results_Dataset004_DBTHybrid__nnUNetTrainer_Seeded42.json"),
    ("large_tumor", "3D U-Net",
     "results/outputs_clean/UNet_BCE_Dataset_large_tumor/logs/test_metrics_per_case.json"),
    ("small_tumor", "3D U-Net",
     "results/outputs_clean/UNet_BCE_Dataset_small_tumor/logs/test_metrics_per_case.json"),
    ("Mixed_Size", "3D U-Net",
     "results/outputs_clean/UNet_BCE_Dataset_Mixed_Size/logs/test_metrics_per_case.json"),
    ("Hybrid", "3D U-Net",
     "results/outputs_improved/UNet_BCE_Dataset_Hybrid/logs/test_metrics_per_case.json"),
    ("large_tumor", "Attention U-Net",
     "results/outputs_improved/Attention_UNet_Dataset_large_tumor/logs/test_metrics_per_case.json"),
    ("small_tumor", "Attention U-Net",
     "results/outputs_improved/Attention_UNet_Dataset_small_tumor/logs/test_metrics_per_case.json"),
    ("Mixed_Size", "Attention U-Net",
     "results/outputs_improved/Attention_UNet_Dataset_Mixed_Size/logs/test_metrics_per_case.json"),
    ("Hybrid", "Attention U-Net",
     "results/outputs_improved/Attention_UNet_Dataset_Hybrid/logs/test_metrics_per_case.json"),
]


# ---------------------------------------------------------------------------
def fmt_p(p):
    """p con tres decimales; nunca 'p = 0.000'."""
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def holm_bonferroni(pvals):
    """Devuelve los p ajustados por Holm, en el orden de entrada."""
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n, dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        val = (n - rank) * pvals[i]
        running = max(running, val)          # monotonia
        adj[i] = min(1.0, running)
    return adj


def rank_biserial(x, y):
    """Correlacion rank-biserial pareada (tamano del efecto de Wilcoxon).

    r = (W+ - W-) / (W+ + W-), sobre los rangos de las diferencias no nulas.
    Positiva cuando x tiende a superar a y.
    """
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    wp, wm = r[d > 0].sum(), r[d < 0].sum()
    return float((wp - wm) / (wp + wm))


def descriptives(values, n_boot, seed):
    v = np.asarray(values, float)
    lo, hi = bootstrap_ci(v, n_boot=n_boot, seed=seed)
    q1, q3 = np.percentile(v, [25, 75])
    return dict(n=len(v), mean=v.mean(), sd=v.std(ddof=1), median=np.median(v),
                q1=q1, q3=q3, ci95_lo=lo, ci95_hi=hi, min=v.min(), max=v.max())


# ---------------------------------------------------------------------------
def b1_descriptive(df, args):
    rows = []
    for m in METHODS:
        for k in METRICS:
            v = df[df.method == m].sort_values("case")[k].values
            rows.append(dict(method=m, metric=k,
                             **descriptives(v, args.n_boot, args.seed)))
    return pd.DataFrame(rows)


def b2_paired(df, args):
    friedman, wilcoxon = [], []
    for k in METRICS:
        wide = df.pivot(index="case", columns="method", values=k).sort_index()
        # Omnibus
        chi2, p = stats.friedmanchisquare(*[wide[m].values for m in METHODS])
        friedman.append(dict(metric=k, n=len(wide), k_methods=len(METHODS),
                             chi2=chi2, df=len(METHODS) - 1, p_value=p,
                             p_formatted=fmt_p(p),
                             role="primary" if k == PRIMARY else "secondary"))
        # Comparaciones pareadas contra el ensemble
        raw = []
        for c in COMPARATORS:
            a, b = wide[REFERENCE].values, wide[c].values
            st_, p_ = stats.wilcoxon(a, b, alternative="two-sided",
                                     zero_method="wilcox")
            d = a - b
            lo, hi = bootstrap_ci(d, n_boot=args.n_boot, seed=args.seed,
                                  stat=np.median)
            raw.append(dict(metric=k, comparison=f"{REFERENCE} vs {c}",
                            n_pairs=len(a), statistic=float(st_), p_value=float(p_),
                            median_difference=float(np.median(d)),
                            diff_ci95_lo=lo, diff_ci95_hi=hi,
                            rank_biserial=rank_biserial(a, b),
                            n_favouring_reference=int((d > 0).sum()),
                            n_favouring_comparator=int((d < 0).sum())))
        # Holm solo dentro de la familia de tres comparaciones de cada metrica
        adj = holm_bonferroni([r["p_value"] for r in raw])
        for r, pa in zip(raw, adj):
            r["p_holm"] = float(pa)
            r["p_formatted"] = fmt_p(r["p_value"])
            r["p_holm_formatted"] = fmt_p(pa)
            r["significant_holm_005"] = bool(pa < 0.05)
            r["role"] = "primary" if k == PRIMARY else "secondary (exploratory)"
        wilcoxon.extend(raw)
    return pd.DataFrame(friedman), pd.DataFrame(wilcoxon)


def b3_synthetic(args):
    rows = []
    for dataset, model, path in SYNTHETIC_RUNS:
        full = f"{REPO}/{path}"
        if not os.path.exists(full):
            print(f"  [aviso] falta {path}")
            continue
        per_case = json.load(open(full))["per_case"]
        for k, key in zip(METRICS, ["dice", "iou", "precision", "recall"]):
            v = [per_case[c][key] for c in sorted(per_case)]
            d = descriptives(v, args.n_boot, args.seed)
            # Con n = 3 no se hace inferencia; solo descriptivos
            d["inference_advised"] = len(v) >= 8
            rows.append(dict(dataset=dataset, model=model, metric=k, **d))
    return pd.DataFrame(rows)


def b4_complementarity(df, args):
    wide = df.pivot(index="case", columns="method", values=PRIMARY).sort_index()
    rows = []
    for i, a in enumerate(COMPARATORS):
        for b in COMPARATORS[i + 1:]:
            rho, p = stats.spearmanr(wide[a], wide[b])
            rows.append(dict(model_a=a, model_b=b, spearman_rho=float(rho),
                             p_value=float(p), p_formatted=fmt_p(p)))
    corr = pd.DataFrame(rows)

    # Modos de fallo: la ventaja del ensemble en la media procede sobre todo de
    # que rescata los casos en que un miembro colapsa por completo, no de una
    # superioridad uniforme caso a caso. Conviene cuantificarlo por separado.
    failures = []
    for m in METHODS:
        v = wide[m].values
        failures.append(dict(method=m,
                             n_zero_dice=int((v < 1e-6).sum()),
                             n_dice_below_0p10=int((v < 0.10).sum()),
                             n_dice_below_0p30=int((v < 0.30).sum()),
                             min_dice=float(v.min()),
                             mean_dice=float(v.mean()),
                             median_dice=float(np.median(v))))

    best_single = wide[COMPARATORS].max(axis=1)
    summary = dict(
        n_cases=len(wide),
        n_ensemble_beats_best_single=int((wide[REFERENCE] > best_single).sum()),
        n_ensemble_beats_all_singles=int(
            (wide[REFERENCE].values[:, None] > wide[COMPARATORS].values)
            .all(axis=1).sum()),
        n_ensemble_worst=int(
            (wide[REFERENCE].values[:, None] < wide[COMPARATORS].values)
            .all(axis=1).sum()),
        mean_best_single=float(best_single.mean()),
        mean_ensemble=float(wide[REFERENCE].mean()),
        mean_spearman_between_members=float(corr.spearman_rho.mean()),
    )
    return corr, summary, pd.DataFrame(failures)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores",
                    default="results/paper_assets/scores/scores_per_case_test.csv")
    ap.add_argument("--out", default="results/statistics")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(f"{REPO}/{args.scores}")
    print(f"Cohorte clinica de test: {df.case.nunique()} casos x "
          f"{df.method.nunique()} metodos\n")

    print("B1  descriptivos...")
    desc = b1_descriptive(df, args)
    desc.to_csv(f"{args.out}/descriptive_test.csv", index=False)

    print("B2  Friedman + Wilcoxon con correccion de Holm...")
    fried, wilc = b2_paired(df, args)
    fried.to_csv(f"{args.out}/friedman_omnibus.csv", index=False)
    wilc.to_csv(f"{args.out}/wilcoxon_pairwise.csv", index=False)

    print("B3  descriptivos de los datasets sinteticos e hibrido...")
    syn = b3_synthetic(args)
    syn.to_csv(f"{args.out}/synthetic_descriptive.csv", index=False)

    print("B4  complementariedad y modos de fallo...")
    corr, comp, fail = b4_complementarity(df, args)
    corr.to_csv(f"{args.out}/complementarity_spearman.csv", index=False)
    fail.to_csv(f"{args.out}/failure_modes_test.csv", index=False)
    with open(f"{args.out}/complementarity_summary.json", "w") as f:
        json.dump(comp, f, indent=2)

    with open(f"{args.out}/analysis_config.json", "w") as f:
        json.dump(dict(n_bootstrap=args.n_boot, seed=args.seed,
                       primary_endpoint=PRIMARY,
                       multiplicity="Holm-Bonferroni over the three "
                                     "pre-specified comparisons per metric",
                       sd_convention="sample standard deviation (ddof = 1)",
                       min_attainable_two_sided_p=2 / 2 ** 10), f, indent=2)

    _report(desc, fried, wilc, comp, fail)


def _report(desc, fried, wilc, comp, fail):
    print("\n" + "=" * 78)
    print("COHORTE CLINICA DE TEST (n = 10) — Dice")
    print("=" * 78)
    d = desc[desc.metric == PRIMARY]
    for _, r in d.iterrows():
        print(f"  {r['method']:12s} {r['mean']:.4f} +/- {r['sd']:.4f}   "
              f"mediana {r['median']:.4f} [{r['q1']:.4f}-{r['q3']:.4f}]   "
              f"IC95% [{r['ci95_lo']:.4f}, {r['ci95_hi']:.4f}]")

    f = fried[fried.metric == PRIMARY].iloc[0]
    print(f"\n  Friedman (4 metodos): chi2 = {f['chi2']:.3f}, "
          f"gl = {f['df']}, p = {f['p_formatted']}")

    print(f"\n  Wilcoxon pareado (Dice), correccion de Holm:")
    w = wilc[wilc.metric == PRIMARY]
    for _, r in w.iterrows():
        print(f"    {r['comparison']:26s} dif. mediana "
              f"{r['median_difference']:+.4f} "
              f"[{r['diff_ci95_lo']:+.4f}, {r['diff_ci95_hi']:+.4f}]  "
              f"p = {r['p_formatted']}  p_Holm = {r['p_holm_formatted']}  "
              f"r = {r['rank_biserial']:+.3f}  "
              f"({r['n_favouring_reference']}/{r['n_pairs']} casos a favor)")

    print(f"\n  Modos de fallo (Dice por caso):")
    for _, r in fail.iterrows():
        print(f"    {r['method']:12s} Dice = 0 en {r['n_zero_dice']}/10 casos, "
              f"< 0.10 en {r['n_dice_below_0p10']}/10, "
              f"minimo = {r['min_dice']:.4f}")

    print(f"\n  Complementariedad: rho de Spearman medio entre miembros = "
          f"{comp['mean_spearman_between_members']:.3f}")
    print(f"  El ensemble supera al mejor individual en "
          f"{comp['n_ensemble_beats_best_single']}/{comp['n_cases']} casos; "
          f"supera a los tres a la vez en "
          f"{comp['n_ensemble_beats_all_singles']}/{comp['n_cases']}")
    print("=" * 78)


if __name__ == "__main__":
    main()
