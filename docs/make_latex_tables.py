"""
Genera los cuerpos de tabla LaTeX del informe de revision a partir de los CSV.

Ningun numero del documento se teclea a mano: cada tabla se emite aqui desde el
artefacto que la produjo, de modo que al reejecutar el analisis el PDF se
actualiza solo. Salida en docs/tables/analisis/*.tex, incluida con \\input{}.

Uso:  python docs/make_latex_tables.py
"""
import json
import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{REPO}/docs/tables/analisis"
os.makedirs(OUT, exist_ok=True)

MODEL_ES = {"nnU-Net": "nnU-Net", "Attention": "Attention U-Net",
            "U-Net BCE": "3D U-Net", "Ensemble": "Ensemble"}
DATASET_ES = {"large_tumor": "Tumor grande", "small_tumor": "Tumor pequeño",
              "Mixed_Size": "Tamaño mixto", "Hybrid": "Híbrido"}
LABEL_ES = {
    "nnU-Net alone": "nnU-Net solo",
    "Attention U-Net alone": "Attention U-Net solo",
    "3D U-Net alone": "3D U-Net solo",
    "nnU-Net + Attention": "nnU-Net + Attention",
    "nnU-Net + 3D U-Net": "nnU-Net + 3D U-Net",
    "Attention + 3D U-Net": "Attention + 3D U-Net",
    "Equal weights": "Pesos iguales",
    "Selected scheme": "\\textbf{Esquema seleccionado}",
    "Alternative": "Alternativa",
}
CAT_ES = {"single-model": "Modelo individual", "pairwise": "Par",
          "equal": "Pesos iguales", "selected": "Seleccionado",
          "alternative": "Alternativa"}


def w(name, body):
    """Escribe un fragmento tal cual (macros)."""
    with open(f"{OUT}/{name}", "w") as f:
        f.write(body.rstrip("\n") + "%\n")
    print(f"-> docs/tables/analisis/{name}")


def w_table(name, colspec, header, rows, size="\\small"):
    """Escribe un entorno tabular COMPLETO.

    El cuerpo no puede incluirse con \input desde dentro de un tabular: TeX
    trata el fin de fichero como frontera de fila y \bottomrule acaba mal
    colocado ("Misplaced \noalign"). Emitir el entorno entero evita el problema
    y ademas mantiene juntas la estructura y los datos de cada tabla.
    """
    body = [size, f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
            header + " \\\\", "\\midrule"]
    body += rows
    body += ["\\bottomrule", "\\end{tabular}"]
    w(name, "\n".join(body))


def esc(s):
    return str(s).replace("&", "\\&").replace("_", "\\_")


# ---------------------------------------------------------------------------
def table_ablation():
    """Tabla 3: rejilla obligatoria de ablacion, solo desarrollo."""
    d = pd.read_csv(f"{REPO}/results/ablation/tables/table3_ensemble_ablation_dev.csv")
    lines, prev = [], None
    for _, r in d.iterrows():
        cat = CAT_ES[r["category"]]
        if cat != prev:
            if prev is not None:
                lines.append("\\addlinespace")
            prev = cat
        sel = r["weights"] == "2:2:3" and abs(r["threshold"] - 0.3) < 1e-9
        name = LABEL_ES.get(r["configuration"], r["configuration"])
        cells = [name, r["weights"], f"{r['threshold']:g}",
                 f"{r['dice_mean']:.3f}", f"{r['dice_sd']:.3f}",
                 f"[{r['dice_ci_lo']:.3f}, {r['dice_ci_hi']:.3f}]",
                 f"{r['iou_mean']:.3f}", f"{r['precision_mean']:.3f}",
                 f"{r['recall_mean']:.3f}", f"{int(r['n_zero_dice'])}"]
        if sel:
            cells = [f"\\textbf{{{c}}}" for c in cells]
        lines.append(" & ".join(cells) + " \\\\")
    w_table("ablation_grid.tex", "llccccccc c",
            "Configuración & Pesos & Umbral & Dice & DE & IC 95\\,\\% & IoU & "
            "Prec. & Rec. & Ceros", lines, size="\\footnotesize")


def table_stability():
    """Estabilidad de la seleccion: bootstrap y leave-one-out."""
    r = json.load(open(f"{REPO}/results/ablation/tables/ablation_stability_dev.json"))
    d = pd.read_csv(f"{REPO}/results/ablation/tables/ablation_stability_dev.csv")
    lines = []
    for _, x in d[d.bootstrap_win_frac > 0].head(6).iterrows():
        sel = x["weights"] == "2:2:3" and abs(x["threshold"] - 0.3) < 1e-9
        cells = [x["weights"], f"{x['threshold']:g}", f"{x['dice_mean']:.3f}",
                 f"{100 * x['bootstrap_win_frac']:.1f}\\,\\%"]
        if sel:
            cells = [f"\\textbf{{{c}}}" for c in cells]
        lines.append(" & ".join(cells) + " \\\\")
    w_table("stability.tex", "lccc",
            "Pesos & Umbral & Dice (desarrollo) & Frecuencia de selección", lines)

    loo = r["leave_one_out"]
    n_sel = sum(1 for x in loo if x["selected_weights"] == "2:2:3"
                and abs(x["selected_threshold"] - 0.3) < 1e-9)
    macros = [
        f"\\newcommand{{\\declBootWin}}{{{100 * r['declared']['bootstrap_win_frac']:.1f}}}",
        f"\\newcommand{{\\declBootRank}}{{{r['declared']['median_bootstrap_rank']:.0f}}}",
        f"\\newcommand{{\\declLooCount}}{{{n_sel}}}",
        f"\\newcommand{{\\declLooTotal}}{{{len(loo)}}}",
        f"\\newcommand{{\\envMin}}{{{r['three_member_dev_envelope']['min']:.3f}}}",
        f"\\newcommand{{\\envMax}}{{{r['three_member_dev_envelope']['max']:.3f}}}",
        f"\\newcommand{{\\envN}}{{{r['three_member_dev_envelope']['n_configs']}}}",
    ]
    rank = json.load(open(f"{REPO}/results/ablation/tables/ablation_ranking_check.json"))
    macros += [
        f"\\newcommand{{\\rankMand}}{{{rank['mandatory_grid']['declared_rank']}}}",
        f"\\newcommand{{\\nMand}}{{{rank['mandatory_grid']['n_configs']}}}",
        f"\\newcommand{{\\rankFine}}{{{rank['fine_grid']['declared_rank']}}}",
        f"\\newcommand{{\\nFine}}{{{rank['fine_grid']['n_configs']}}}",
        f"\\newcommand{{\\declDice}}{{{rank['declared']['dice_mean']:.4f}}}",
        f"\\newcommand{{\\bestFineW}}{{{rank['fine_grid']['best']['weights']}}}",
        f"\\newcommand{{\\bestFineThr}}{{{rank['fine_grid']['best']['threshold']:g}}}",
        f"\\newcommand{{\\bestFineDice}}{{{rank['fine_grid']['best']['dice_mean']:.4f}}}",
    ]
    w("macros_ablation.tex", "\n".join(macros) + "\n")


def table_aggregation():
    d = pd.read_csv(f"{REPO}/results/ablation/tables/ablation_aggregation_dev.csv")
    es = {"Weighted mean (2:2:3)": "Promedio ponderado (2:2:3)",
          "Unweighted mean (1:1:1)": "Promedio no ponderado (1:1:1)",
          "Logit (log-odds) mean": "Promedio de \\emph{logits}",
          "Voxel-wise maximum": "Máximo por vóxel",
          "Voxel-wise minimum": "Mínimo por vóxel",
          "Majority vote (2/3 at 0.5)": "Voto por mayoría (2/3 a 0{,}5)",
          "Weighted mean + largest connected component":
              "Ponderado + mayor componente conexa"}
    lines = []
    for _, r in d.iterrows():
        thr = r["selected_threshold"]
        thr = "---" if pd.isna(thr) or thr == "n/a" else f"{float(thr):g}"
        lines.append(" & ".join([
            es.get(r["rule"], r["rule"]), thr, f"{r['dice_mean']:.3f}",
            f"{r['dice_sd']:.3f}", f"{r['precision_mean']:.3f}",
            f"{r['recall_mean']:.3f}", f"{int(r['n_zero_dice'])}"]) + " \\\\")
    w_table("aggregation.tex", "lcccccc",
            "Regla & Umbral & Dice & DE & Precisión & Recall & Ceros", lines)


def table_descriptive():
    """Tabla 2 revisada: descriptivos de la cohorte clinica de test."""
    d = pd.read_csv(f"{REPO}/results/statistics/tables/descriptive_test.csv")
    lines, prev = [], None
    for _, r in d.iterrows():
        m = MODEL_ES[r["method"]]
        if m != prev:
            if prev is not None:
                lines.append("\\addlinespace")
            prev = m
            first = m
        else:
            first = ""
        lines.append(" & ".join([
            first, r["metric"],
            f"{r['mean']:.3f} $\\pm$ {r['sd']:.3f}",
            f"{r['median']:.3f} [{r['q1']:.3f}--{r['q3']:.3f}]",
            f"[{r['ci95_lo']:.3f}, {r['ci95_hi']:.3f}]",
            f"{r['min']:.3f}--{r['max']:.3f}"]) + " \\\\")
    w_table("descriptive_test.tex", "llcccc",
            "Método & Métrica & Media $\\pm$ DE & Mediana [RIC] & IC 95\\,\\% & "
            "Rango", lines)


def fmt_p_tex(p):
    """p con tres decimales; nunca 'p = 0.000'. Se reformatea desde el valor
    numerico porque al releer el CSV pandas convierte '1.000' en float."""
    return "{<}\\,0.001" if p < 0.001 else f"{p:.3f}"


def table_wilcoxon():
    """Tabla 4: comparaciones pareadas."""
    d = pd.read_csv(f"{REPO}/results/statistics/tables/wilcoxon_pairwise.csv")
    f = pd.read_csv(f"{REPO}/results/statistics/tables/friedman_omnibus.csv")
    lines, prev = [], None
    for _, r in d.iterrows():
        if r["metric"] != prev:
            if prev is not None:
                lines.append("\\addlinespace")
            prev = r["metric"]
            fr = f[f.metric == r["metric"]].iloc[0]
            lines.append(f"\\multicolumn{{7}}{{l}}{{\\itshape {r['metric']} "
                         f"--- Friedman $\\chi^2 = {fr['chi2']:.2f}$, "
                         f"gl $= {int(fr['df'])}$, "
                         f"$p = {fmt_p_tex(fr['p_value'])}$}} \\\\")
        comp = r["comparison"].replace("Ensemble vs ", "")
        comp = MODEL_ES.get(comp, comp)
        star = "$^{*}$" if r["significant_holm_005"] else ""
        lines.append(" & ".join([
            f"Ensemble vs.\\ {comp}",
            f"{r['median_difference']:+.3f}",
            f"[{r['diff_ci95_lo']:+.3f}, {r['diff_ci95_hi']:+.3f}]",
            f"${fmt_p_tex(r['p_value'])}$",
            f"${fmt_p_tex(r['p_holm'])}$" + star,
            f"{r['rank_biserial']:+.2f}",
            f"{int(r['n_favouring_reference'])}/{int(r['n_pairs'])}"]) + " \\\\")
    w_table("wilcoxon.tex", "lccccc c",
            "Comparación & Dif.\\ mediana & IC 95\\,\\% & $p$ & $p$ (Holm) & $r$ & "
            "A favor", lines)


def table_synthetic():
    """Tabla 1 ampliada con dispersion e IC."""
    d = pd.read_csv(f"{REPO}/results/statistics/tables/synthetic_descriptive.csv")
    order_ds = ["large_tumor", "small_tumor", "Mixed_Size", "Hybrid"]
    order_m = ["nnU-Net", "3D U-Net", "Attention U-Net"]
    lines = []
    for ds in order_ds:
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\itshape {DATASET_ES[ds]}}} \\\\")
        for m in order_m:
            sub = d[(d.dataset == ds) & (d.model == m)]
            if sub.empty:
                continue
            cells = [m]
            for metric in ["Dice", "IoU", "Precision", "Recall"]:
                r = sub[sub.metric == metric].iloc[0]
                cells.append(f"{r['mean']:.3f} $\\pm$ {r['sd']:.3f}")
            r = sub[sub.metric == "Dice"].iloc[0]
            cells.append(f"[{r['ci95_lo']:.3f}, {r['ci95_hi']:.3f}]")
            cells.append(f"{int(r['n'])}")
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\addlinespace")
    w_table("synthetic_descriptive.tex", "lccccc c",
            "Modelo & Dice & IoU & Precisión & Recall & IC 95\\,\\% (Dice) & $n$",
            lines[:-1], size="\\footnotesize")


def table_failures():
    d = pd.read_csv(f"{REPO}/results/statistics/tables/failure_modes_test.csv")
    lines = []
    for _, r in d.iterrows():
        lines.append(" & ".join([
            MODEL_ES[r["method"]], f"{int(r['n_zero_dice'])}/10",
            f"{int(r['n_dice_below_0p10'])}/10",
            f"{int(r['n_dice_below_0p30'])}/10",
            f"{r['min_dice']:.3f}", f"{r['median_dice']:.3f}",
            f"{r['mean_dice']:.3f}"]) + " \\\\")
    w_table("failure_modes.tex", "lcccccc",
            "Método & Dice $=0$ & Dice $<0{,}10$ & Dice $<0{,}30$ & Mínimo & "
            "Mediana & Media", lines)


def table_reproduction():
    """Validacion de la re-inferencia frente a los valores publicados."""
    runs = [
        ("3D U-Net", "large_tumor", "results/outputs_clean/UNet_BCE_Dataset_large_tumor"),
        ("3D U-Net", "small_tumor", "results/outputs_clean/UNet_BCE_Dataset_small_tumor"),
        ("3D U-Net", "Mixed_Size", "results/outputs_clean/UNet_BCE_Dataset_Mixed_Size"),
        ("3D U-Net", "Hybrid", "results/outputs_improved/UNet_BCE_Dataset_Hybrid"),
        ("Attention U-Net", "large_tumor", "results/outputs_improved/Attention_UNet_Dataset_large_tumor"),
        ("Attention U-Net", "small_tumor", "results/outputs_improved/Attention_UNet_Dataset_small_tumor"),
        ("Attention U-Net", "Mixed_Size", "results/outputs_improved/Attention_UNet_Dataset_Mixed_Size"),
        ("Attention U-Net", "Hybrid", "results/outputs_improved/Attention_UNet_Dataset_Hybrid"),
    ]
    lines = []
    for model, ds, run in runs:
        c = json.load(open(f"{REPO}/{run}/logs/test_metrics_per_case.json"))["reproduction_check"]
        lines.append(" & ".join([
            model, DATASET_ES[ds],
            f"{c['published_dice']:.6f}", f"{c['recomputed_dice']:.6f}",
            f"{c['worst_overlap_abs_diff']:.1e}",
            f"{c['worst_distance_abs_diff']:.2f}",
            "\\checkmark" if c["passed"] else "---"]) + " \\\\")
    w_table("reproduction.tex", "llccccc",
            "Modelo & Dataset & Dice publicado & Dice recalculado & "
            "Solapam.\\ (abs.) & Dist.\\ (vóx.) & OK", lines)


def macros_stats():
    """Cifras sueltas que el texto cita, para no teclearlas."""
    comp = json.load(open(f"{REPO}/results/statistics/tables/complementarity_summary.json"))
    sel = json.load(open(f"{REPO}/results/ablation/tables/selected_config_test.json"))
    m = [
        f"\\newcommand{{\\meanRho}}{{{comp['mean_spearman_between_members']:.3f}}}",
        f"\\newcommand{{\\nBeatsBest}}{{{comp['n_ensemble_beats_best_single']}}}",
        f"\\newcommand{{\\nCases}}{{{comp['n_cases']}}}",
        f"\\newcommand{{\\testDice}}{{{sel['summary']['dice']['mean']:.4f}}}",
        f"\\newcommand{{\\testDiceMed}}{{{sel['summary']['dice']['median']:.4f}}}",
        f"\\newcommand{{\\testDiceCIlo}}{{{sel['summary']['dice']['ci95'][0]:.4f}}}",
        f"\\newcommand{{\\testDiceCIhi}}{{{sel['summary']['dice']['ci95'][1]:.4f}}}",
    ]
    w("macros_stats.tex", "\n".join(m) + "\n")


if __name__ == "__main__":
    table_ablation()
    table_stability()
    table_aggregation()
    table_descriptive()
    table_wilcoxon()
    table_synthetic()
    table_failures()
    table_reproduction()
    macros_stats()
