# Paper assets — ensemble sim→real (D004, test 100% real)

Artefactos generados para el manuscrito por [`ensemble/make_paper_assets.py`](../../ensemble/make_paper_assets.py).
Todo se calcula desde `data_probs/` + `data/real/` (sin GPU, sin reentrenar) y desde los
logs reales en `results/finetuning_logs/` y `results/outputs_*/`.

Regenerar:

```bash
python ensemble/make_paper_assets.py     # entorno con numpy + matplotlib + scipy
```

Configuración del ensemble (elegida en OOF dev, sin tocar test): pesos `nn=2, att=2, bce=3`,
umbral `0.3`. Modelos individuales evaluados a umbral `0.5`.

## `scores/`
Métricas Dice / IoU / Precision / Recall para nnU-Net, Attention, U-Net BCE y el **Ensemble**.

| Archivo | Contenido |
|---|---|
| `scores_per_case_{test,dev}.csv` | métrica por caso y por método (formato largo) |
| `scores_summary_{test,dev}.csv`  | media, std, mediana, min, max por método y métrica |
| `fig_scores_comparison_{test,dev}.png` | barras agrupadas (media ± std) métrica × método |
| `fig_per_case_dice_{test,dev}.png` | Dice por caso, ensemble vs miembros |

**Resumen test (10 reales held-out):**

| Método | Dice | IoU | Precision | Recall |
|---|:--:|:--:|:--:|:--:|
| nnU-Net   | 0.482 | 0.367 | 0.495 | 0.497 |
| Attention | 0.372 | 0.244 | 0.291 | 0.603 |
| U-Net BCE | 0.367 | 0.249 | 0.496 | 0.338 |
| **Ensemble** | **0.518** | **0.384** | 0.465 | **0.630** |

El ensemble es el mejor en Dice, IoU y Recall, y elimina los casos nulos (0/10).

## `examples/`
Resultados cualitativos sobre los 10 casos de test. Slice mostrada = la de mayor área de GT.
Codificación **TP (amarillo) / FP (rojo) / FN (verde)**.

- `example_real_dbt_XXX.png` — 1 por caso (imagen | GT | predicción vs GT).
- `examples_grid_all_test.png` — montaje 2×5 con etiqueta de calidad por caso.

Categorías por Dice: **Excelente** ≥0.65 · **Bueno** ≥0.45 · **Regular** <0.45.
Ejemplos representativos: excelente `real_dbt_005` (0.778), bueno `real_dbt_013` (0.590),
regular `real_dbt_019` (0.050, tumor de muy bajo contraste).

## `curves/`
| Archivo | Contenido |
|---|---|
| `nnunet_ft_curves.png` | **fine-tuning de nnU-Net** (miembro principal): train_loss, val_loss y val pseudo-Dice por época, media ± std de los 5 folds (Tversky+CE, sim→real) |
| `nnunet_ft_per_epoch.csv` | los mismos valores por época (media/std de folds) |
| `base_attention_curves.png` | entrenamiento **base sintético** de Attention U-Net ('Mixed_Size', warm-start): train_loss + val Dice/IoU/Precision/Recall |
| `base_unet_bce_curves.png`  | ídem para U-Net BCE |

> Nota: las curvas de validación **por época** solo existen para el fine-tuning de nnU-Net
> (registradas por nnU-Netv2) y para el entrenamiento base de las redes custom. El
> fine-tuning de las redes custom solo registró *train-loss* disperso (ver
> `results/finetuning_logs/custom_nets/`); el test se evaluó una sola vez (no hay curva de test).
