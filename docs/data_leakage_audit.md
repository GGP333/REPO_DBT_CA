# Auditoría de data leakage train/test (4 arquitecturas × 4 datasets)

**Fecha**: 2026-05-28
**Alcance**: experimentos en `outputs_improved/` (UNet_BCE, Attention_UNet, nnUnet_original) y `experiments/nnunet_official/` (nnUNet oficial).

## Resumen

Se detectó **data leakage 100% en UNet_BCE** para Dataset_large_tumor, Dataset_small_tumor y Dataset_Both. Las 8/3/11 cases del test set están todas dentro del train set, lo que invalida las métricas de test reportadas para esos tres runs. Las restantes 13 combinaciones (UNet_BCE Both_RealWorld + las 4 de Attention_UNet + las 4 de nnUnet_original + las 4 de nnUNet oficial) están limpias.

## Tabla de auditoría

| Arquitectura | Dataset | train | val | test | tr+va+te | universe | cov% | tr∩va | **te∩tr** | te∩va | te=imagesTs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **UNet_BCE** | Large | 72 | 18 | 8 | 90 | 90 | 100 | 0 | **8** ❌ | 0 | ✓ |
| **UNet_BCE** | Small | 24 | 6 | 3 | 30 | 30 | 100 | 0 | **3** ❌ | 0 | ✓ |
| **UNet_BCE** | Both | 96 | 24 | 11 | 120 | 120 | 100 | 0 | **11** ❌ | 0 | ✓ |
| UNet_BCE | Both_RealWorld | 104 | 26 | 10 | 140 | 140 | 100 | 0 | 0 ✓ | 0 | ✓ |
| Attention_UNet | Large | 65 | 16 | 8 | 89 | 90 | 98.9 | 0 | 0 ✓ | 0 | ✓ |
| Attention_UNet | Small | 22 | 5 | 3 | 30 | 30 | 100 | 0 | 0 ✓ | 0 | ✓ |
| Attention_UNet | Both | 87 | 22 | 11 | 120 | 120 | 100 | 0 | 0 ✓ | 0 | ✓ |
| Attention_UNet | Both_RealWorld | 104 | 26 | 10 | 140 | 140 | 100 | 0 | 0 ✓ | 0 | ✓ |
| nnUnet_original | Large | 65 | 16 | 8 | 89 | 90 | 98.9 | 0 | 0 ✓ | 0 | ✓ |
| nnUnet_original | Small | 22 | 5 | 3 | 30 | 30 | 100 | 0 | 0 ✓ | 0 | ✓ |
| nnUnet_original | Both | 87 | 22 | 11 | 120 | 120 | 100 | 0 | 0 ✓ | 0 | ✓ |
| nnUnet_original | Both_RealWorld | 104 | 26 | 10 | 140 | 140 | 100 | 0 | 0 ✓ | 0 | ✓ |
| nnUNet oficial | (los 4) | — | — | — | — | — | — | — | 0 ✓ | — | ✓ |

`universe` = total de estudios disponibles para el dataset (`imagesTr ∪ imagesTs` de la convención nnUNet); `te=imagesTs` indica si el set de test predicho coincide exactamente con `imagesTs` de nnUNet.

## Causa del leakage en UNet_BCE

Archivo: `src/models/unet_bce/src/dataset_dbt.py`, función `make_dataloaders`.

La rama `else` (cuando el dataset no contiene casos `real_dbt_*`, es decir Large/Small/Both) usa `test_ratio` como variable en la línea 239:

```python
n_test = max(1, int(round(n_total * test_ratio)))
```

Sin embargo, `test_ratio` **no está definido** en el scope: ni es atributo de `cfg`, ni parámetro, ni global del módulo. Al ejecutar este path se produce `NameError: name 'test_ratio' is not defined` (verificado replicando el código en aislamiento).

Esto implica que **el código actual no puede reproducir los resultados** que están guardados en `outputs_improved/UNet_BCE_Dataset_{large_tumor,small_tumor,Both}/`. La versión histórica del código (no commiteada) debe haber tenido un comportamiento distinto que llevó al leakage: probablemente cargaba el test desde una fuente independiente (por ejemplo, los `imagesTs` de nnUNet) sin excluir esos archivos del train.

Evidencia del leakage histórico:
- En UNet_BCE Small, `run_config.json` declara train=24, val=6, sin lista de test.
- En `test_preds_npy/` aparecen exactamente `dbt_001, dbt_004, dbt_021` (= `imagesTs` de nnUNet Dataset002_DBTSmall).
- `dbt_001, dbt_004, dbt_021` están listados explícitamente entre los 24 `train.study_ids` del run_config.

Mismo patrón en Large (8/8) y Both (11/11). En Both_RealWorld el test es 100% `real_dbt_*` y ningún `real_dbt_*` aparece en el train, por lo que el run quedó limpio "por accidente".

## Comportamiento correcto de las demás arquitecturas

- **Attention_UNet**: el split está en `train_unet_dbt.py`, no en `dataset_dbt.py`. Usa `--test-ratio 0.1` por defecto, `_normalize_ratios` y asigna ids excluyentemente a train/val/test (`train_unet_dbt.py:75-153`). `make_dataloaders` sólo recibe records ya separados.
- **nnUnet_original**: misma estructura que Attention_UNet — split externo en el script de entrada.
- **nnUNet oficial**: usa la convención `imagesTr` / `imagesTs` propia de nnU-Net v2; el test queda físicamente fuera del directorio de training.

## Impacto sobre los resultados publicables

Las siguientes métricas son **inválidas** (test contaminado con train):

| Run | Dice test reportado | ¿Usable? |
|---|---|---|
| UNet_BCE Large | 0.9138 | ❌ inflado |
| UNet_BCE Small | 0.8207 | ❌ inflado |
| UNet_BCE Both | 0.8543 | ❌ inflado |
| UNet_BCE Both+Real | 0.4116 | ✓ |

Todos los demás runs (12) son válidos.

## Benchmark consolidado (DICE val/test) descartando runs contaminados

Para nnU-Net oficial, val se reporta con `checkpoint_best.pth` (re-evaluado el 2026-05-28, consistente con el test).

| Arquitectura \ Dataset | Large (val/test) | Small (val/test) | Both (val/test) | Both+Real (val/test) | Epochs |
|---|---|---|---|---|---|
| UNet_BCE | 0.810 / ⚠️0.914 | 0.539 / ⚠️0.821 | 0.802 / ⚠️0.854 | 0.732 / 0.412 | 500 |
| Attention_UNet | 0.660 / 0.778 | 0.392 / 0.367 | 0.480 / 0.582 | 0.584 / 0.421 | 500 |
| nnUnet_original | **0.898** / **0.890** | 0.689 / 0.447 | 0.720 / **0.842** | **0.777** / **0.464** | 500 |
| nnUNet oficial | 0.892 / 0.864 | **0.736** / **0.560** | **0.772** / 0.841 | 0.731 / 0.353 | 1000 |

(⚠️ = test contaminado; negrita = mejor de la columna entre los runs válidos)

## Mejor época nnUNet oficial (selección por EMA pseudo-Dice)

| Dataset | Mejor época | Pseudo-Dice (best) | Pseudo-Dice (epoch 999) |
|---|---|---|---|
| Dataset001_DBTLarge | 414 | 0.9040 | 0.8897 |
| Dataset002_DBTSmall | 239 | 0.8894 | 0.8437 |
| Dataset003_DBTBoth | 131 | 0.8722 | 0.8455 |
| Dataset004_DBTBothRealWorld | 327 | 0.8555 | 0.8351 |

D003 convergió en ~130 épocas; entrenar 1000 fue redundante.

## Acciones pendientes (recomendadas)

1. **Re-entrenar UNet_BCE** en Large/Small/Both excluyendo `imagesTs` del split antes de invocar `make_dataloaders`, o reescribir `make_dataloaders` para que reciba el universo limpio. Mientras eso no se haga, no reportar los Dice de test para esos tres datasets.
2. **Reparar** `src/models/unet_bce/src/dataset_dbt.py:239`: definir `test_ratio` (atributo de `cfg` o parámetro) o adoptar la misma arquitectura que Attention_UNet (split externo al `make_dataloaders`).
3. Considerar mover el split a un módulo único compartido (`src/shared/splits.py`) para evitar tres implementaciones independientes con riesgo de divergencia.
4. Confirmar caso faltante en Attention_UNet Large / nnUnet_original Large (89 de 90 cubiertos). No es leak, pero hay un estudio fuera del experimento — verificar si quedó filtrado por máscara vacía o error de carga.

## Ubicación de evidencias

- Validation con `checkpoint_best.pth` (re-generadas): `experiments/nnunet_official/nnunet_root/results/Dataset00*/nnUNetTrainer_Seeded42__nnUNetPlans__3d_fullres/fold_0/validation_best/summary.json`
- Validation con `checkpoint_final.pth` (backup original): `…/fold_0/validation_final/summary.json`
- Test (siempre `checkpoint_best.pth`): `experiments/nnunet_official/eval/results_*__nnUNetTrainer_Seeded42.json`
- run_config.json de las otras arquitecturas: `outputs_improved/<Arch>_Dataset_*/run_config.json`
- Predicciones test crudas: `outputs_improved/<Arch>_Dataset_*/test_preds_npy/`
