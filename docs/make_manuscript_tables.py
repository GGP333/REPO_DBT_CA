"""
Genera, en INGLES, los cuerpos de tabla LaTeX destinados al manuscrito.

Se distingue de make_latex_tables.py (que emite las tablas en espanol para el
informe) en el idioma y en el formato: estas son las tablas listas para
sustituir o anadir en el manuscrito de Diagnostics, y por tanto siguen la
numeracion del paper (Tabla 1, 2, 3, 4).

Ningun numero se teclea a mano: todo procede de los CSV de results/, de modo
que al reejecutar el analisis las tablas se actualizan solas.

Salida: docs/tables/manuscrito/*.tex

Uso:  python docs/make_manuscript_tables.py
"""
import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{REPO}/docs/tables/manuscrito"
os.makedirs(OUT, exist_ok=True)

# Etiquetas del manuscrito. El CSV usa el nombre interno del modelo.
MODEL_EN = {"nnU-Net": "nnU-Net", "Attention": "Attention U-Net",
            "U-Net BCE": "3D U-Net (baseline)", "Ensemble": "Ensemble",
            "Attention U-Net": "Attention U-Net", "3D U-Net": "3D U-Net (baseline)"}
DATASET_EN = {"large_tumor": "Large tumour dataset",
              "small_tumor": "Small tumour dataset",
              "Mixed_Size": "Mixed-size tumour dataset",
              "Hybrid": "Hybrid synthetic--clinical dataset"}
CATEGORY_EN = {"single-model": "Single model", "pairwise": "Pairwise",
               "equal": "Equal weights", "selected": "Selected",
               "alternative": "Alternative"}
ORDER_MODELS = ["nnU-Net", "3D U-Net (baseline)", "Attention U-Net"]


def write(name, lines):
    with open(f"{OUT}/{name}", "w") as f:
        f.write("\n".join(lines).rstrip("\n") + "%\n")
    print(f"-> docs/tables/manuscrito/{name}")


def pm(mean, sd):
    return f"{mean:.3f} $\\pm$ {sd:.3f}"


def fmt_p(p):
    """p con tres decimales; nunca 'p = 0.000'. Se reformatea desde el valor
    numerico porque al releer el CSV pandas convierte '1.000' en float."""
    return "\\textless\\,0.001" if p < 0.001 else f"{p:.3f}"


def p_stmt(p):
    """'$p$ < 0.001' o '$p$ = 0.041'. El signo cambia con el valor, de modo que
    no puede emitirse un '=' fijo delante de fmt_p()."""
    return "$p$ \\textless\\,0.001" if p < 0.001 else f"$p$ = {p:.3f}"


# ---------------------------------------------------------------------------
def table1_synthetic():
    """Tabla 1 rediseniada: paneles (a) sintetico y (b) hibrido.

    El problema de la version enviada es de maquetacion: la primera columna
    repite el nombre del dataset y lo parte en varias lineas. Aqui el dataset
    pasa a ser una fila de cabecera que ocupa el ancho completo, con lo que
    ninguna celda necesita partirse.
    """
    d = pd.read_csv(f"{REPO}/results/statistics/tables/synthetic_descriptive.csv")
    d["model"] = d["model"].map(lambda m: MODEL_EN.get(m, m))
    wide = d.pivot_table(index=["dataset", "model"], columns="metric",
                         values=["mean", "sd", "n", "ci95_lo", "ci95_hi"],
                         aggfunc="first")

    lines = ["\\footnotesize",
             "\\setlength{\\tabcolsep}{5pt}",
             "\\renewcommand{\\arraystretch}{1.15}",
             "\\begin{tabular}{@{}lccccc@{}}",
             "\\toprule",
             "Model & Dice & IoU & Precision & Recall & 95\\,\\% CI (Dice) \\\\",
             "\\midrule"]

    panels = [("(a) Synthetic datasets --- held-out synthetic test sets",
               ["large_tumor", "small_tumor", "Mixed_Size"]),
              ("(b) Hybrid synthetic--clinical dataset --- held-out clinical cohort "
               "($n$ = 10)",
               ["Hybrid"])]

    for p_i, (title, datasets) in enumerate(panels):
        if p_i:
            lines += ["\\addlinespace[2pt]", "\\midrule"]
        lines.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\textbf{{{title}}}}} \\\\")
        for ds in datasets:
            sub = wide.loc[ds]
            n = int(sub[("n", "Dice")].iloc[0])
            lines.append("\\addlinespace[2pt]")
            if len(datasets) > 1:
                # Con un unico dataset en el panel, el titulo del panel ya lo
                # nombra: repetirlo solo anade una fila vacia de informacion.
                lines.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{DATASET_EN[ds]}}} "
                             f"($n$ = {n} test volumes)}} \\\\")
            best = sub[("mean", "Dice")].idxmax()
            for model in ORDER_MODELS:
                if model not in sub.index:
                    continue
                r = sub.loc[model]
                cells = [pm(r[("mean", m)], r[("sd", m)])
                         for m in ("Dice", "IoU", "Precision", "Recall")]
                if model == best:
                    cells[0] = f"\\textbf{{{cells[0]}}}"
                ci = f"[{r[('ci95_lo', 'Dice')]:.3f}, {r[('ci95_hi', 'Dice')]:.3f}]"
                lines.append(f"\\quad {model} & " + " & ".join(cells) + f" & {ci} \\\\")

    lines += ["\\bottomrule", "\\end{tabular}"]
    write("table1_synthetic.tex", lines)


# ---------------------------------------------------------------------------
def table2_test():
    """Tabla 2 revisada: cohorte clinica de prueba, con dispersion e IC."""
    d = pd.read_csv(f"{REPO}/results/statistics/tables/descriptive_test.csv")
    lines = ["\\small",
             "\\setlength{\\tabcolsep}{5pt}",
             "\\begin{tabular}{@{}llcccc@{}}",
             "\\toprule",
             "Model & Metric & Mean $\\pm$ SD & Median [IQR] & 95\\,\\% CI & Range \\\\",
             "\\midrule"]
    for i, (method, sub) in enumerate(d.groupby("method", sort=False)):
        if i:
            lines.append("\\addlinespace")
        label = MODEL_EN.get(method, method)
        for j, (_, r) in enumerate(sub.iterrows()):
            name = f"\\textbf{{{label}}}" if (j == 0 and method == "Ensemble") else (label if j == 0 else "")
            lines.append(" & ".join([
                name, r["metric"], pm(r["mean"], r["sd"]),
                f"{r['median']:.3f} [{r['q1']:.3f}--{r['q3']:.3f}]",
                f"[{r['ci95_lo']:.3f}, {r['ci95_hi']:.3f}]",
                f"{r['min']:.3f}--{r['max']:.3f}"]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("table2_test.tex", lines)


# ---------------------------------------------------------------------------
def table3_ablation():
    """Tabla 3 (nueva): rejilla obligatoria de ablacion, solo desarrollo."""
    d = pd.read_csv(f"{REPO}/results/ablation/tables/table3_ensemble_ablation_dev.csv")
    lines = ["\\footnotesize",
             "\\setlength{\\tabcolsep}{4pt}",
             "\\begin{tabular}{@{}llcccccc c@{}}",
             "\\toprule",
             "Configuration & Weights & Thr. & Dice $\\pm$ SD & 95\\,\\% CI & IoU & "
             "Prec. & Rec. & Zeros \\\\",
             "\\midrule"]
    prev, prev_w = None, None
    for _, r in d.iterrows():
        if r["category"] != prev:
            if prev is not None:
                lines.append("\\addlinespace[2pt]")
            lines.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\emph{{{CATEGORY_EN[r['category']]}}}}} \\\\")
            prev, prev_w = r["category"], None
        selected = r["weights"] == "2:2:3" and abs(r["threshold"] - 0.3) < 1e-9
        # El nombre solo se repite cuando cambia el esquema de pesos: las tres
        # filas de una misma configuracion son los tres umbrales exigidos.
        name = f"\\quad {r['configuration']}" if r["weights"] != prev_w else ""
        prev_w = r["weights"]
        cells = [name, r["weights"], f"{r['threshold']:g}",
                 pm(r["dice_mean"], r["dice_sd"]),
                 f"[{r['dice_ci_lo']:.3f}, {r['dice_ci_hi']:.3f}]",
                 f"{r['iou_mean']:.3f}", f"{r['precision_mean']:.3f}",
                 f"{r['recall_mean']:.3f}", f"{int(r['n_zero_dice'])}"]
        if selected:
            cells = [f"\\textbf{{{c}}}" for c in cells]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("table3_ablation.tex", lines)


# ---------------------------------------------------------------------------
def table4_paired():
    """Tabla 4 (nueva): contrastes pareados sobre la cohorte de prueba."""
    w = pd.read_csv(f"{REPO}/results/statistics/tables/wilcoxon_pairwise.csv")
    fr = pd.read_csv(f"{REPO}/results/statistics/tables/friedman_omnibus.csv").set_index("metric")
    lines = ["\\small",
             "\\setlength{\\tabcolsep}{5pt}",
             "\\begin{tabular}{@{}lccccc c@{}}",
             "\\toprule",
             "Comparison & Median diff. & 95\\,\\% CI & $p$ & $p$ (Holm) & $r$ & Favours \\\\",
             "\\midrule"]
    for i, (metric, sub) in enumerate(w.groupby("metric", sort=False)):
        if i:
            lines.append("\\addlinespace")
        f = fr.loc[metric]
        lines.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{metric}}} --- Friedman "
                     f"$\\chi^2$ = {f['chi2']:.2f}, df = {int(f['df'])}, "
                     f"{p_stmt(f['p_value'])}}} \\\\")
        for _, r in sub.iterrows():
            star = "$^{*}$" if r["significant_holm_005"] else ""
            comp = MODEL_EN.get(r["comparison"].replace("Ensemble vs ", ""),
                                r["comparison"].replace("Ensemble vs ", ""))
            lines.append(" & ".join([
                f"\\quad Ensemble vs.\\ {comp}",
                f"{r['median_difference']:+.3f}",
                f"[{r['diff_ci95_lo']:+.3f}, {r['diff_ci95_hi']:+.3f}]",
                fmt_p(r["p_value"]), f"{fmt_p(r['p_holm'])}{star}",
                f"{r['rank_biserial']:+.2f}",
                f"{int(r['n_favouring_reference'])}/{int(r['n_pairs'])}"]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("table4_paired.tex", lines)


if __name__ == "__main__":
    table1_synthetic()
    table2_test()
    table3_ablation()
    table4_paired()
