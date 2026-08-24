"""
Tablas de detalle: per-caso, zero-shot, rangos con signo y splits.

Mismo principio que make_latex_tables.py: ningun numero se escribe a mano, todo
sale de los CSV/JSON de results/ y de los run_config.json de cada entrenamiento.
Cada archivo emite un entorno tabular completo, no filas sueltas.

Uso:  python docs/make_report_tables.py
"""
import csv, glob, json, os
import numpy as np
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{REPO}/docs/tables/detalle"
os.makedirs(OUT, exist_ok=True)

METHODS = ["nnU-Net", "Attention", "U-Net BCE", "Ensemble"]
LABEL = {"nnU-Net": "nnU-Net", "Attention": "Attention U-Net",
         "U-Net BCE": "3D U-Net", "Ensemble": "Ensemble"}


def write(name, body):
    with open(f"{OUT}/{name}.tex", "w") as f:
        f.write(body)
    print(f"-> {OUT}/{name}.tex")


def per_case():
    d = {}
    for r in csv.DictReader(open(f"{REPO}/results/paper_assets/scores/"
                                 "scores_per_case_test.csv")):
        d.setdefault(r["method"], {})[r["case"]] = float(r["Dice"])
    return d, sorted(d["Ensemble"])


# --------------------------------------------------------------- splits
def t_splits():
    rows = []
    for f in sorted(glob.glob(f"{REPO}/results/outputs_*/*/run_config.json")):
        c = json.load(open(f))
        ds, hp = c["dataset"], c["hyperparameters"]
        run = os.path.basename(os.path.dirname(f))
        net = "3D U-Net" if run.startswith("UNet_BCE") else "Attention U-Net"
        cfg = hp["dataset_root"].split("/")[-1].replace("Dataset_", "")
        rows.append((cfg, net, ds["train"]["n_studies"],
                     ds["val"]["n_studies"], ds["test"]["n_studies"]))
    order = {"small_tumor": 0, "large_tumor": 1, "Mixed_Size": 2, "Hybrid": 3}
    rows.sort(key=lambda r: (order[r[0]], r[1]))
    nice = {"small_tumor": "Tumor pequeño", "large_tumor": "Tumor grande",
            "Mixed_Size": "Tamaño mixto", "Hybrid": "Híbrido"}
    b = ["\\begin{tabular}{llrrrr}", "\\toprule",
         "Configuración & Red & Train & Val & Test & Total \\\\", "\\midrule"]
    prev = None
    for cfg, net, tr, va, te in rows:
        if prev is not None and cfg != prev:
            b.append("\\addlinespace")
        b.append(f"{nice[cfg] if cfg != prev else ''} & {net} & {tr} & {va} "
                 f"& {te} & {tr+va+te} \\\\")
        prev = cfg
    b += ["\\bottomrule", "\\end{tabular}%"]
    write("splits_sinteticos", "\n".join(b))


# --------------------------------------------------- Dice por caso (test)
def t_percase():
    d, cases = per_case()
    b = ["\\begin{tabular}{lrrrrr}", "\\toprule",
         "Caso & nnU-Net & Attention & 3D U-Net & Ensemble & "
         "Ens.$-$nnU-Net \\\\", "\\midrule"]
    for c in cases:
        diff = d["Ensemble"][c] - d["nnU-Net"][c]
        mark = "\\bfseries " if abs(diff) > 0.5 else ""
        b.append(f"{mark}{c.replace('real_dbt_', '')} & "
                 + " & ".join(f"{mark}{d[m][c]:.3f}" for m in METHODS)
                 + f" & {mark}${diff:+.3f}$ \\\\")
    b.append("\\midrule")
    b.append("Media & " + " & ".join(
        f"{np.mean([d[m][c] for c in cases]):.3f}" for m in METHODS)
        + f" & ${np.mean([d['Ensemble'][c]-d['nnU-Net'][c] for c in cases]):+.3f}$ \\\\")
    b += ["\\bottomrule", "\\end{tabular}%"]
    write("percase_test", "\n".join(b))


# ------------------------------------------------ rangos de Wilcoxon
def t_wilcoxon(other, name):
    d, cases = per_case()
    a = np.array([d["Ensemble"][c] for c in cases])
    o = np.array([d[other][c] for c in cases])
    diff = a - o
    rk = stats.rankdata(np.abs(diff))
    idx = np.argsort(np.abs(diff))
    b = ["\\begin{tabular}{lrrrrrc}", "\\toprule",
         f"Caso & Ensemble & {LABEL[other]} & Dif. & $|$Dif.$|$ & Rango & Signo \\\\",
         "\\midrule"]
    for i in idx:
        b.append(f"{cases[i].replace('real_dbt_', '')} & {a[i]:.3f} & {o[i]:.3f} "
                 f"& ${diff[i]:+.3f}$ & {abs(diff[i]):.3f} & {rk[i]:.0f} "
                 f"& ${'+' if diff[i] > 0 else '-'}$ \\\\")
    rp, rn = rk[diff > 0].sum(), rk[diff < 0].sum()
    res = stats.wilcoxon(a, o, alternative="two-sided")
    b += ["\\midrule",
          f"\\multicolumn{{7}}{{l}}{{$R^+ = {rp:.0f}$ \\quad $R^- = {rn:.0f}$ "
          f"\\quad $W = {min(rp, rn):.0f}$ \\quad $p = {res.pvalue:.4f}$ "
          f"\\quad $r = {(rp-rn)/55:+.2f}$}} \\\\",
          "\\bottomrule", "\\end{tabular}%"]
    write(name, "\n".join(b))


# ------------------------------------------------------- rangos Friedman
def t_friedman():
    d, cases = per_case()
    X = np.array([[d[m][c] for m in METHODS] for c in cases])
    R = np.apply_along_axis(lambda r: stats.rankdata(-r), 1, X)
    res = stats.friedmanchisquare(*X.T)
    b = ["\\begin{tabular}{lrrrr}", "\\toprule",
         "Caso & nnU-Net & Attention & 3D U-Net & Ensemble \\\\", "\\midrule"]
    for i, c in enumerate(cases):
        tie = len(set(R[i])) < 4
        b.append(f"{c.replace('real_dbt_', '')}{'$^\\dagger$' if tie else ''} & "
                 + " & ".join(f"{R[i][j]:.1f}" for j in range(4)) + " \\\\")
    b += ["\\midrule",
          "\\bfseries Suma & " + " & ".join(
              f"\\bfseries {v:.1f}" for v in R.sum(axis=0)) + " \\\\",
          "\\midrule",
          f"\\multicolumn{{5}}{{l}}{{$\\chi^2 = {res.statistic:.2f}$, "
          f"df $= 3$, $p = {res.pvalue:.3f}$}} \\\\",
          "\\bottomrule", "\\end{tabular}%"]
    write("friedman_rangos", "\n".join(b))


# ------------------------------- todas las pruebas alternativas (no significan)
def t_alternativas():
    d, cases = per_case()
    a = np.array([d["Ensemble"][c] for c in cases])
    o = np.array([d["nnU-Net"][c] for c in cases])
    diff = a - o
    from itertools import product
    obs = diff.mean()
    perm = sum(1 for s in product([1, -1], repeat=10)
               if (diff*np.array(s)).mean() >= obs) / 1024
    fe, fn = (a == 0).astype(int), (o == 0).astype(int)
    nb = int(((fn == 1) & (fe == 0)).sum()); nc = int(((fn == 0) & (fe == 1)).sum())
    tests = [
        ("Wilcoxon bilateral \\emph{(la reportada)}",
         stats.wilcoxon(a, o, alternative="two-sided").pvalue),
        ("Wilcoxon unilateral, a favor del ensemble",
         stats.wilcoxon(a, o, alternative="greater").pvalue),
        ("$t$-test pareado", stats.ttest_rel(a, o).pvalue),
        ("Test de signos", stats.binomtest(int((diff > 0).sum()), 10, 0.5).pvalue),
        ("Permutación exacta ($2^{10}$ combinaciones)", perm),
        ("McNemar exacto sobre fallos completos",
         stats.binomtest(nb, nb+nc, 0.5).pvalue),
    ]
    b = ["\\begin{tabular}{lr}", "\\toprule", "Prueba & $p$ \\\\", "\\midrule"]
    for n, p in tests:
        b.append(f"{n} & {p:.3f} \\\\")
    b += ["\\bottomrule", "\\end{tabular}%"]
    write("pruebas_alternativas", "\n".join(b))


# --------------------------------------- rejilla por categoria (5 categorias)
def t_categorias():
    rows = list(csv.DictReader(open(f"{REPO}/results/ablation/tables/"
                                    "table3_ensemble_ablation_dev.csv")))
    nice = {"single-model": "Modelo individual", "pairwise": "Por pares",
            "equal": "Pesos iguales", "alternative": "Alternativas",
            "selected": "\\bfseries Seleccionada"}
    b = ["\\begin{tabular}{llcr}", "\\toprule",
         "Categoría & Mejor configuración & Umbral & Dice (dev) \\\\", "\\midrule"]
    for cat in ["single-model", "pairwise", "equal", "alternative", "selected"]:
        sub = sorted([r for r in rows if r["category"] == cat],
                     key=lambda r: -float(r["dice_mean"]))[0]
        bf = "\\bfseries " if cat == "selected" else ""
        cfg = sub["configuration"] if cat != "selected" else "Esquema seleccionado"
        b.append(f"{nice[cat]} & {bf}{cfg} ({sub['weights']}) & {bf}{sub['threshold']} "
                 f"& {bf}{float(sub['dice_mean']):.4f} \\\\")
    b += ["\\bottomrule", "\\end{tabular}%"]
    write("rejilla_categorias", "\n".join(b))


# ------------------------------------ reglas de agregacion, con ceros
def t_reglas():
    rows = list(csv.DictReader(open(f"{REPO}/results/ablation/tables/"
                                    "ablation_aggregation_dev.csv")))
    nice = {"Weighted mean (2:2:3)": "Media ponderada (2:2:3)",
            "Logit (log-odds) mean": "Media de log-odds",
            "Weighted mean + largest connected component":
                "Ponderada $+$ componente conexa mayor",
            "Unweighted mean (1:1:1)": "Media no ponderada (1:1:1)",
            "Majority vote (2/3 at 0.5)": "Voto por mayoría (2/3 a 0.5)",
            "Voxel-wise maximum": "Máximo por vóxel",
            "Voxel-wise minimum": "Mínimo por vóxel"}
    rows.sort(key=lambda r: -float(r["dice_mean"]))
    b = ["\\begin{tabular}{lcrrrc}", "\\toprule",
         "Regla & Umbral & Dice & Precisión & Recall & Dice $=0$ \\\\",
         "\\midrule"]
    for r in rows:
        z = int(r["n_zero_dice"])
        b.append(f"{nice[r['rule']]} & {r['selected_threshold']} & "
                 f"{float(r['dice_mean']):.4f} & {float(r['precision_mean']):.3f} & "
                 f"{float(r['recall_mean']):.3f} & "
                 f"{'\\bfseries ' if z else ''}{z} \\\\")
    b += ["\\bottomrule", "\\end{tabular}%"]
    write("reglas_agregacion", "\n".join(b))


# ------------------------------------------- hybrid vs fine-tuning
def t_hybrid():
    d, cases = per_case()
    hyb = {}
    for run, lab in [("UNet_BCE_Dataset_Hybrid", "U-Net BCE"),
                     ("Attention_UNet_Dataset_Hybrid", "Attention")]:
        j = json.load(open(f"{REPO}/results/outputs_improved/{run}/logs/"
                           "test_metrics_per_case.json"))
        hyb[lab] = {k: v["dice"] for k, v in j["per_case"].items()}
    b = ["\\begin{tabular}{lrr}", "\\toprule",
         "Red & Preentrenar $+$ \\emph{fine-tuning} & Entrenamiento conjunto "
         "(Híbrido) \\\\", "\\midrule"]
    for m, lab in [("U-Net BCE", "3D U-Net"), ("Attention", "Attention U-Net")]:
        ft = np.mean([d[m][c] for c in cases])
        hy = np.mean(list(hyb[m].values()))
        b.append(f"{lab} & {ft:.3f} & \\bfseries {hy:.3f} \\\\")
    b.append("\\midrule")
    b.append(f"Ensemble de las tres FT & {np.mean([d['Ensemble'][c] for c in cases]):.3f} & --- \\\\")
    b += ["\\bottomrule", "\\end{tabular}%"]
    write("hybrid_vs_ft", "\n".join(b))



# ------------------------------------------- zero-shot sobre cohorte clinica
def t_zeroshot():
    d = json.load(open(f"{REPO}/results/statistics/tables/zeroshot_clinical.json"))
    nets = list(d["networks"])
    cases = list(d["networks"][nets[0]]["per_case"])
    b = ["\\begin{tabular}{lrr}", "\\toprule",
         "Caso & " + " & ".join(nets) + " \\\\", "\\midrule"]
    for c in cases:
        b.append(c.replace("real_dbt_", "") + " & " + " & ".join(
            f"{d['networks'][n]['per_case'][c]['dice']:.4f}" for n in nets) + " \\\\")
    b.append("\\midrule")
    for lab, k in [("Dice medio", "dice"), ("IoU medio", "iou"),
                   ("Precisión media", "precision"), ("Recall medio", "recall")]:
        bf = "\\bfseries " if k == "dice" else ""
        b.append(f"{bf}{lab} & " + " & ".join(
            f"{bf}{d['networks'][n]['mean'][k]:.4f}" for n in nets) + " \\\\")
    b.append("Casos con Dice $=0$ & " + " & ".join(
        f"{d['networks'][n]['mean']['n_zero_dice']}/10" for n in nets) + " \\\\")
    b += ["\\bottomrule", "\\end{tabular}%"]
    write("zeroshot", "\n".join(b))


if __name__ == "__main__":
    t_splits(); t_percase()
    t_wilcoxon("nnU-Net", "wilcoxon_nnunet")
    t_wilcoxon("Attention", "wilcoxon_attention")
    t_friedman(); t_alternativas(); t_categorias(); t_reglas(); t_hybrid()
    t_zeroshot()
