# Análisis estadístico de la cohorte clínica

Salidas de `ensemble/statistical_analysis.py` sobre los 10 casos clínicos de test.

| Archivo | Contenido |
|---|---|
| `descriptive_test.csv` | Media ± DE, mediana [IQR] e IC 95 % bootstrap por método y métrica |
| `friedman_omnibus.csv` | Friedman sobre los cuatro métodos, por métrica |
| `wilcoxon_pairwise.csv` | Wilcoxon pareado del ensemble contra cada red, con corrección Holm |
| `failure_modes_test.csv` | Recuento de fallos completos por método |
| `complementarity_spearman.csv` | Correlación de Spearman entre los Dice por caso de los tres miembros |
| `complementarity_summary.json` | Resumen de complementariedad — **leer la nota de abajo** |
| `synthetic_descriptive.csv` | Descriptivos por caso de las ocho ejecuciones sintéticas |
| `analysis_config.json` | Semilla, número de remuestreos, convención de DE, criterio de multiplicidad |

## Nota sobre `complementarity_summary.json`

El campo **`mean_best_single` (0.565)** es un **oráculo por caso**: para cada paciente
toma el Dice del mejor de los tres modelos individuales *en ese paciente concreto*, y
promedia. El campo `n_ensemble_beats_all_singles` (2 de 10) cuenta en cuántos pacientes el
ensemble supera simultáneamente a los tres miembros.

**No es una estrategia realizable y no debe compararse con `mean_ensemble` (0.518) como si
fueran dos métodos alternativos.** Elegir el mejor modelo para cada paciente requiere
conocer la anotación de referencia de ese paciente, que es precisamente lo que se quiere
predecir. Es una cota superior teórica, no un rendimiento alcanzable.

Su función en el análisis es otra: cuantificar **cuánta complementariedad hay entre los
miembros**. Que el oráculo esté por encima del ensemble indica que los tres modelos fallan
en pacientes distintos y que aún queda margen que una regla de fusión mejor podría
capturar. Esa misma complementariedad es la que mide, de forma más interpretable, la
correlación de Spearman media entre miembros (0.53) que sí se reporta en el manuscrito.

El manuscrito no reporta `mean_best_single`, y no debe reportarse sin esta aclaración.
