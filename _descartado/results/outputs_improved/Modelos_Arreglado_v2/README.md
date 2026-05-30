# Segmentacion 3D de Volumenes DBT (Digital Breast Tomosynthesis)

## 1. Vision General del Proyecto

Este proyecto implementa y compara tres arquitecturas de redes neuronales convolucionales 3D para la **segmentacion automatica de tumores mamarios** en volumenes de Tomosintesis Digital de Mama (DBT). El objetivo es evaluar el rendimiento de distintas variantes de U-Net en la tarea de segmentacion binaria voxel a voxel, utilizando multiples configuraciones de datos que incluyen tumores de diferentes tamanos y datos reales vs sinteticos.

### Contribuciones principales

- Comparacion sistematica de 3 arquitecturas (nnUNet 3D auto-configurado, Attention U-Net 3D, U-Net BCE) sobre 4 configuraciones de datos.
- Implementacion de pipeline de auto-configuracion inspirado en nnU-Net (Isensee et al.): fingerprint del dataset, planificacion automatica de arquitectura con pooling anisotropo, entrenamiento basado en patches con foreground oversampling, sliding window inference con ponderacion Gaussiana, y post-procesamiento de connected components.
- Evaluacion con 12 metricas de segmentacion incluyendo Hausdorff Distance 95 (HD95), Average Precision (AP) y False Positives por volumen.
- Pipeline completo desde preprocesamiento TIFF hasta evaluacion cuantitativa y visualizacion de predicciones.
- Entrenamiento de 500 epocas con reproducibilidad garantizada (seed=42).

---

## 2. Datasets

Los datos provienen de volumenes 3D de DBT, originalmente almacenados como stacks de imagenes TIFF (una imagen por slice axial). El preprocesamiento los convierte en arreglos NumPy `.npy` con forma `Z x 256 x 512` (Z x H x W), donde **Z se conserva variable** segun cada estudio.

### 2.1 Preprocesamiento

El preprocesamiento utilizado para entrenar/evaluar los experimentos reportados corresponde a `scripts/preprocess_simple.py`. Este pipeline es deliberadamente minimalista y aplica exactamente los siguientes pasos:

1. **Carga de estudio y mascara GT**
   - Lee el stack TIFF de imagen (`img_*`) y reconstruye un volumen 3D en formato `(Z, Y, X)`.
   - Lee la mascara de lesion desde `mask_*` (o `--mask_dir`), verifica que tenga la misma forma que la imagen.
   - Si llega una dimension extra singleton (por ejemplo `(1, Z, Y, X)`), la compacta para mantener volumen 3D valido.

2. **Clipping de intensidades**
   - Aplica recorte por percentiles con `--clip_percentiles` (default `[0.5, 99.5]`).
   - Este paso reduce valores extremos antes de normalizar.

3. **Normalizacion de imagen**
   - `--normalize zscore` (default): estandariza con media y desviacion estandar del volumen completo.
   - `--normalize minmax`: escala a `[0, 1]`.
   - `--normalize none`: conserva intensidades tras clipping.

4. **Resize en plano XY**
   - Redimensiona solo XY a `--target_size_xy` (default `512 256`) y mantiene `Z` sin cambios.
   - Interpolacion lineal para imagen continua.
   - Interpolacion nearest-neighbor para la mascara, seguida de binarizacion (`> 0.5`) para preservar etiquetas discretas.

5. **Guardado de artefactos**
   - Guarda:
     - `{study_id}_img.npy` (`float32`)
     - `{study_id}_mask.npy` (`uint8`)
   - Registra `preprocess_params.json` con parametros usados (shape original, resize, clipping y modo de normalizacion).

### 2.2 Configuraciones de datos

Se definen 4 datasets a partir del conjunto preprocesado:

| Dataset | Descripcion | Estudios en directorio | Estudios usados | Train | Val | Test |
|---------|-------------|:---:|:---:|:---:|:---:|:---:|
| **Dataset_large_tumor** | Solo tumores grandes (alta relacion fg/bg) | 90 | 90 | 72 | 18 | 8 |
| **Dataset_small_tumor** | Solo tumores pequenos (baja relacion fg/bg) | 30 | 30 | 24 | 6 | 3 |
| **Dataset_Both** | Mezcla de tumores grandes y pequenos (90 large + 30 small) | 120 | 120 | 96 | 24 | 11 |
| **Dataset_Both_RealWorld** | `dbt_*` + `real_dbt_*` (datos clinicos reales) | 140 | 140 | ~104 | ~26 | 10 (real_dbt_*) |

**Nota sobre los splits Train/Val/Test**: los valores de la tabla corresponden al split usado por UNet_BCE y Attention U-Net (80/20 train/val). nnUNet usa un split ligeramente diferente (e.g. 65/16/8 en large_tumor, 87/22/11 en Both). En ambos casos el conjunto de test es el mismo (8, 3, 11 estudios respectivamente) y se evalua con el best checkpoint.

**Nota sobre Dataset_Both_RealWorld**: el directorio contiene 140 estudios (120 `dbt_*` + 20 `real_dbt_*`). El test set se construye con el **50% de los `real_dbt_*`** (10 estudios clinicos reales). El 50% restante de `real_dbt_*` se mezcla con los `dbt_*` para formar train/val. Esto permite que el modelo aprenda de datos reales y sinteticos, y se evalue exclusivamente con datos clinicos reales no vistos.

### 2.3 Split train/val/test

- **Seed fija**: 42 (reproducibilidad total).
- **Split por estudio**: la particion se hace a nivel de estudio (no por slice), evitando fuga de datos.
- **Caso general (sin `real_dbt_*`)**: se usan las proporciones configuradas de train/val/test sobre el total del dataset.
- **Caso especial con datos reales (`real_dbt_*`)**: el 50% de los estudios `real_dbt_*` se asigna exclusivamente al conjunto **test**. El 50% restante se mezcla con los `dbt_*` para formar train/val. Asi el modelo entrena con datos reales y sinteticos, y se evalua solo con datos clinicos reales no vistos.
- Esta logica esta implementada en `Models/nnUnet_original/train_unet_dbt.py`, `Models/Attention U_Net/train_unet_dbt.py` y `Models/U_Net BCE/src/dataset_dbt.py`.

---

## 3. Arquitecturas de Modelos

Se comparan tres variantes de U-Net 3D para segmentacion volumetrica. Conceptualmente, las tres comparten el patron encoder-decoder con skip connections, pero difieren en como modelan contexto, como fusionan informacion multi-escala, como se adaptan a la geometria del dato y que objetivo de optimizacion priorizan.

### 3.1 nnUNet 3D Auto-configurado (`nnUnet_original`)

**Parametros**: variables (auto-calculados por el planner)

Implementacion que sigue la filosofia central de nnU-Net de Isensee et al. (1809.10486): la arquitectura **no esta fija**, sino que se **auto-configura** a partir de un analisis automatico de las propiedades del dataset. El pipeline de auto-configuracion consta de tres etapas:

#### 3.1.1 Dataset Fingerprint (`src/fingerprint.py`)

Antes de entrenar, se escanean todos los volumenes del dataset y se extraen estadisticas:
- **Shape mediano, minimo y maximo** `(Z, H, W)` de todos los estudios.
- **Ratio foreground/background** mediano (porcentaje de voxeles que son tumor).
- **Estadisticas de intensidad** (media, std, percentiles globales).
- **Spacing** (parametrizable; por defecto `(1, 1, 1)` para datos ya resampleados).

El fingerprint se guarda como `fingerprint.json` en el directorio de salida.

#### 3.1.2 Architecture Planner (`src/planner.py`)

Recibe el fingerprint y la VRAM disponible de la GPU, y calcula automaticamente:

- **Patch size**: basado en el shape mediano del dataset, ajustado para caber en GPU. Cada dimension se redondea a multiplo de `2^(n_pools_en_ese_eje)`.
- **Strides anisotropos por nivel**: para cada eje se calcula `max_pools = floor(log2(dim / 4))`. Si Z es mucho menor que H/W (datos anisotropos, comun en DBT), Z recibe menos poolings. Ejemplo: `[(1,2,2), (2,2,2), (2,2,2), (1,2,2)]` en lugar del stride isotrópico `(2,2,2)` fijo.
- **Canales por nivel**: comienza en `base_ch=32`, duplica por nivel, con tope en 320.
- **Batch size**: estimado por heuristica de VRAM (`patch_voxels * max_channels * overhead`), maximizando lo que cabe en GPU.

El plan se guarda como `architecture_plan.json`.

#### 3.1.3 Arquitectura resultante

El modelo se construye con `nnUNet3D.from_plan(plan)` y conserva los principios arquitectonicos de nnU-Net:

- **Bloques residuales (`nnUNetResidualBlock`)**: convoluciones 3D con atajo residual, estabilizando el entrenamiento profundo.
- **Normalizacion por instancia** (`InstanceNorm3d`): util con batch size pequeño.
- **Activacion LeakyReLU**: reduce el riesgo de neuronas muertas.
- **Encoder con strides anisotropos**: los pooling se adaptan a la geometria del dato. Ejes cortos (Z en DBT) reciben menos downsampling para no perder informacion inter-slice.
- **Decoder con `ConvTranspose3d`** anisotropo: los kernels de upsample coinciden con los strides del encoder.
- **Gradient checkpointing**: intercambia computo por memoria.
- **Cabeza binaria**: `Conv3d 1x1 + sigmoid`.

#### 3.1.4 Entrenamiento basado en patches (`src/patch_utils.py`)

En lugar de procesar volumenes completos con padding, el entrenamiento usa **patches aleatorios** de tamaño fijo (determinado por el planner):

- Con probabilidad `oversample_foreground` (0.33 por defecto), el centro del patch se fuerza sobre un voxel de foreground, asegurando que la red vea suficiente tumor a pesar del desbalance extremo (fg < 1%).
- Las coordenadas de foreground se pre-calculan por volumen para extraccion rapida.
- El collate es trivial (todos los patches tienen el mismo tamaño), eliminando la necesidad de padding dinamico.

#### 3.1.5 Sliding Window Inference

En validacion y test, el volumen completo se recorre con **sliding window** con overlap configurable (50% por defecto):

- Cada patch se predice independientemente.
- Las predicciones se acumulan usando un **mapa de importancia Gaussiano** (voxeles centrales pesan mas que los bordes) para evitar artefactos en las costuras.
- El resultado es un mapa de probabilidad a resolucion completa.

#### 3.1.6 Post-procesamiento (`src/postprocess.py`)

Tras la prediccion, se aplica filtrado de **componentes conectados 3D**:

- Se prueban varios umbrales de tamaño minimo (`min_size`) sobre el conjunto de validacion.
- Se selecciona automaticamente el `min_size` que maximiza el Dice en validacion.
- Se aplica el mismo umbral a las predicciones de test.

### 3.2 Attention U-Net 3D (`Attention_UNet`)

**Parametros**: 35.575M

Arquitectura inspirada en Attention U-Net de Ozan Oktay et al. (arXiv:1804.03999), cuyo aporte central es introducir **Attention Gates (AGs)** en los skips para suprimir activaciones irrelevantes y enfatizar regiones candidatas al objetivo.

La idea clave de los AGs en este repo es:

- **Señal de compuerta (`gate`)**: viene del decoder (mas semantica, menor resolucion).
- **Señal de skip (`skip`)**: viene del encoder (mas detalle espacial, menos semantica).
- **Fusion aditiva + activacion**:
  - `Wg(gate) + Wx(skip) -> ReLU -> psi(1x1x1) -> sigmoid = alpha`
  - `skip_atendido = skip * alpha`

Asi, el decoder no concatena "todo" el skip, sino una version ponderada espacialmente por relevancia para la lesion.

Ademas de los AGs base de Oktay, tu implementacion introduce componentes extra orientados a DBT:

- **`DBTSpecificBlock3D` multi-escala** (`1x1x1`, `3x3x3`, `5x5x5`) para combinar patrones finos y gruesos de tejido.
- **CBAM 3D** (canal + espacial) en bloques clave, reforzando recalibracion de features.
- **`ConvBlock3D` con Dropout3D**: mejora regularizacion en un modelo de alta capacidad.
- **Decoder con upsample trilineal + AG + concatenacion + convolucion**.
- **Checkpointing** para hacer viable su mayor complejidad parametrica.

Comparada con una U-Net clasica, esta variante intenta mejorar sensibilidad en estructuras pequenas o ambiguas al filtrar mejor el ruido anatomico de fondo.

### 3.3 U-Net BCE (`UNet_BCE`)

**Parametros**: 23.535M

Basada en la U-Net original de Ronneberger et al. (MICCAI 2015; arXiv:1505.04597), extendida a 3D para segmentacion volumetrica.

La estructura sigue el esquema canonico:

- **Encoder (contraccion)**: bloques convolucionales 3D + `MaxPool3d(2)` para comprimir resolucion y aumentar contexto.
- **Bottleneck**: bloque de mayor ancho de canales (512) donde se integra informacion global del volumen.
- **Decoder (expansion)**: upsampling trilineal progresivo, fusion con skip connections del encoder y refinamiento convolucional.
- **Alineacion espacial de skips**: cuando hay diferencias por tamanos impares, se aplica center crop para concatenar sin errores.
- **Cabeza de salida**: `Conv3d 1x1 + sigmoid` para mapa probabilistico binario.

Aqui la diferencia principal no es solo arquitectonica, sino de objetivo:

- **Loss BCE**: penaliza error voxel-a-voxel de manera directa y estable.
- En contraste, nnUNet y Attention U-Net en este proyecto usan configuraciones tipo Dice+CE para enfatizar solape y balancear clase minoritaria.

Esta variante funciona como baseline fuerte: menos mecanismos avanzados que Attention U-Net, pero con inductive bias claro, implementacion estable y buena capacidad de generalizacion.

### 3.4 Comparacion de arquitecturas

| Caracteristica | nnUNet 3D Auto | Attention U-Net | U-Net BCE |
|---------------|:---------:|:---------------:|:---------:|
| Parametros (M) | 31.3 (auto) | 35.575 | 23.535 |
| Auto-configuracion | Si (fingerprint + planner) | No | No |
| Entrenamiento | Patch-based | Volumen completo | Volumen completo |
| Inferencia | Sliding window + Gaussian | Forward directo | Forward directo |
| Pooling | Anisotropo (adaptado a Z/H/W) | Isotropo (stride 2) | Isotropo (stride 2) |
| Post-procesamiento | Connected components auto | No | No |
| Normalizacion | InstanceNorm3D | InstanceNorm3D | InstanceNorm3D |
| Activacion | LeakyReLU | ReLU | ReLU |
| Bloques residuales | Si | No | No |
| Attention gates | No | Si | No |
| Upsampling | ConvTranspose3d (aniso) | Trilineal | Trilineal |
| Dropout | No | Si (0.1) | No |
| Funcion de perdida | Dice + CE | Dice + CE | BCE |
| Gradient checkpointing | Si | Si | No |

---

## 4. Configuracion de Entrenamiento

Todos los experimentos comparten la misma configuracion base, definida en `configs/config.yaml`:

| Parametro | Valor |
|-----------|-------|
| **Seed** | 42 |
| **Epocas** | 500 |
| **Optimizador** | AdamW (lr=0.001, weight_decay=1e-5) |
| **Scheduler** | Polynomial LR Decay: `lr * (1 - epoch/max_epochs)^0.9` |
| **Warmup** | 5 epocas (linear warmup) |
| **Batch size** | 1 (Att. U-Net, U-Net BCE) / auto-calculado (nnUNet) |
| **AMP** | Activado (Mixed Precision con GradScaler) |
| **Gradient clipping** | max_norm=1.0 |
| **Umbral de binarizacion** | 0.5 |
| **Foreground oversampling** | 0.5 (Att. U-Net, U-Net BCE) / 0.33 patch-based (nnUNet) |
| **Augmentacion** | Activada |
| **Early stopping** | Desactivado (se entrenan las 500 epocas completas) |

**Diferencias especificas de nnUNet auto-configurado:**

| Parametro | Valor |
|-----------|-------|
| **auto_plan** | `true` (activa fingerprint + planner) |
| **gpu_vram_gb** | 16.0 (RTX 4070 Ti SUPER) |
| **Entrenamiento** | Patch-based (tamaño auto-calculado) |
| **Validacion/Test** | Sliding window con overlap=0.5 y ponderacion Gaussiana |
| **Foreground oversampling** | 0.33 (prob. de forzar patch centrado en foreground) |
| **Post-procesamiento** | Connected components (min_size auto-optimizado en val) |

### 4.1 Funciones de perdida

- **Dice + CE** (nnUNet, Attention U-Net): `0.5 * DiceLoss + 0.5 * BCE`. Combina la optimizacion directa del coeficiente Dice con la estabilidad del BCE para manejar el desbalance de clases extremo.
- **BCE** (U-Net BCE): Solo Binary Cross-Entropy con logits (numericamente estable).

### 4.2 Guardado de checkpoints

- **best.pt**: Mejor modelo segun val_dice (se guarda cuando mejora con min_delta=0.001).
- **last.pt**: Ultimo checkpoint de cada epoca (para recuperacion).

### 4.3 Hardware

- **GPU**: NVIDIA GeForce RTX 4070 Ti SUPER
- **CUDA**: 12.4
- **PyTorch**: 2.5.1+cu124
- **Sistema**: Linux 6.17.9 (Arch)

### 4.4 Tiempo de entrenamiento

| Modelo | Dataset | Tiempo |
|--------|---------|--------|
| nnUNet | Both_RealWorld | 12.0h |
| nnUNet | Both | 12.1h |
| nnUNet | small_tumor | 9.9h |
| nnUNet | large_tumor | 11.2h |
| Attention_UNet | Both_RealWorld | 5.7h |
| Attention_UNet | Both | 5.7h |
| Attention_UNet | small_tumor | 1.4h |
| Attention_UNet | large_tumor | 4.3h |
| UNet_BCE | Both_RealWorld | 2.8h |
| UNet_BCE | Both | 2.8h |
| UNet_BCE | small_tumor | 0.7h |
| UNet_BCE | large_tumor | 2.1h |
| **Total** | | **~70.8h** |

---

## 5. Metricas de Evaluacion

Se calculan 12 metricas de segmentacion para cada experimento. Todas las metricas se evaluan sobre el mejor checkpoint (best.pt) en los conjuntos de validacion y test.

### 5.1 Metricas de solapamiento

| Metrica | Formula | Rango | Ideal |
|---------|---------|:-----:|:-----:|
| **Dice Coefficient** | 2\|P intersect G\| / (\|P\| + \|G\|) | [0, 1] | 1 |
| **IoU (Jaccard)** | \|P intersect G\| / \|P union G\| | [0, 1] | 1 |
| **Precision** | TP / (TP + FP) | [0, 1] | 1 |
| **Recall** | TP / (TP + FN) | [0, 1] | 1 |
| **F1 Score** | 2 * Precision * Recall / (Precision + Recall) | [0, 1] | 1 |
| **Accuracy** | (TP + TN) / Total voxels | [0, 1] | 1 |

### 5.2 Metricas de distancia

| Metrica | Descripcion | Rango | Ideal |
|---------|-------------|:-----:|:-----:|
| **Hausdorff Distance (HD)** | Maxima distancia bidireccional entre superficies de prediccion y GT. Calculada sobre puntos de superficie extraidos por erosion morfologica. | [0, inf) | 0 |
| **HD95** | Percentil 95 de las distancias de superficie combinadas. Mas robusto a outliers que HD. | [0, inf) | 0 |

### 5.3 Metricas de deteccion

| Metrica | Descripcion | Rango | Ideal |
|---------|-------------|:-----:|:-----:|
| **FP/Volume** | Proporcion de voxeles falsos positivos respecto al volumen total. | [0, 1] | 0 |
| **FP/Image** | Promedio de FP por slice axial (normalizado por area del slice). | [0, 1] | 0 |
| **AP (Average Precision)** | Area bajo la curva Precision-Recall variando el umbral de probabilidad (19 umbrales de 0.05 a 0.95). Usa interpolacion monotonica. | [0, 1] | 1 |
| **mAP** | Mean AP estilo COCO: promedio de AP sobre 10 umbrales de IoU (0.5 a 0.95 en pasos de 0.05). | [0, 1] | 1 |

---

## 6. Resultados

### 6.1 Metricas de validacion (best checkpoint)

| Modelo | Dataset | Dice | IoU | Precision | Recall | HD95 | AP | mAP |
|--------|---------|:----:|:---:|:---------:|:------:|:----:|:--:|:---:|
| nnUNet | Both_RealWorld | 0.7499 | 0.6485 | 0.7324 | 0.7989 | 32.06 | 0.6849 | 0.3778 |
| nnUNet | Both | 0.7201 | 0.6335 | 0.7115 | 0.7501 | 33.23 | 0.6781 | 0.4010 |
| nnUNet | small_tumor | 0.6890 | 0.5688 | 0.7368 | 0.7188 | 34.73 | 0.5798 | 0.2627 |
| **nnUNet** | **large_tumor** | **0.8976** | **0.8156** | **0.8778** | **0.9230** | **1.79** | **0.8598** | **0.5972** |
| Att. U-Net | Both_RealWorld | 0.4871 | 0.3737 | 0.4774 | 0.5558 | 68.36 | 0.4137 | 0.1006 |
| Att. U-Net | Both | 0.4799 | 0.3663 | 0.4916 | 0.5245 | 68.85 | 0.4061 | 0.0958 |
| Att. U-Net | small_tumor | 0.3924 | 0.2740 | 0.3585 | 0.6022 | 88.88 | 0.2827 | 0.0217 |
| Att. U-Net | large_tumor | 0.6597 | 0.5138 | 0.5865 | 0.7811 | 29.57 | 0.5614 | 0.1601 |
| U-Net BCE | Both_RealWorld | 0.7796 | 0.6505 | 0.7882 | 0.7801 | 14.25 | 0.8309 | 0.3374 |
| **U-Net BCE** | **Both** | **0.8020** | **0.6767** | **0.8289** | **0.7840** | **20.15** | **0.8483** | **0.3860** |
| U-Net BCE | small_tumor | 0.5394 | 0.4089 | 0.5600 | 0.5470 | 51.38 | 0.5785 | 0.0817 |
| U-Net BCE | large_tumor | 0.8100 | 0.6949 | 0.8626 | 0.7715 | 21.68 | 0.8525 | 0.4510 |

### 6.2 Metricas de test (best checkpoint)

| Modelo | Dataset | Dice | IoU | Precision | Recall | HD95 | AP | mAP |
|--------|---------|:----:|:---:|:---------:|:------:|:----:|:--:|:---:|
| nnUNet | Both_RealWorld | 0.8113 | 0.7113 | 0.8366 | 0.8234 | 16.34 | 0.7667 | 0.4624 |
| nnUNet | Both | 0.8417 | 0.7371 | 0.8126 | 0.8840 | 8.58 | 0.7878 | 0.4695 |
| nnUNet | small_tumor | 0.4470 | 0.3374 | 0.4364 | 0.5327 | 46.68 | 0.3347 | 0.0608 |
| nnUNet | large_tumor | 0.8901 | 0.8040 | 0.9085 | 0.8760 | 2.74 | 0.8439 | 0.5668 |
| Att. U-Net | Both_RealWorld | 0.5729 | 0.4280 | 0.5854 | 0.6298 | 49.98 | 0.4564 | 0.0971 |
| Att. U-Net | Both | 0.5823 | 0.4334 | 0.6036 | 0.6458 | 54.42 | 0.4564 | 0.0888 |
| Att. U-Net | small_tumor | 0.3670 | 0.2692 | 0.3047 | 0.5045 | 53.90 | 0.2795 | 0.0405 |
| Att. U-Net | large_tumor | 0.7776 | 0.6434 | 0.7661 | 0.8124 | 29.78 | 0.7028 | 0.2529 |
| **U-Net BCE** | **Both_RealWorld** | **0.8233** | **0.7096** | **0.7937** | **0.8623** | **6.92** | **0.8888** | **0.4381** |
| **U-Net BCE** | **Both** | **0.8543** | **0.7511** | **0.8385** | **0.8781** | **10.96** | **0.9271** | **0.5266** |
| **U-Net BCE** | **small_tumor** | **0.8207** | **0.6964** | **0.8103** | **0.8334** | **22.98** | **0.8996** | **0.3621** |
| **U-Net BCE** | **large_tumor** | **0.9138** | **0.8418** | **0.9240** | **0.9050** | **1.78** | **0.9669** | **0.7080** |

### 6.3 Mejor epoca por experimento

| Modelo | Dataset | Mejor epoca (de 500) | Val Dice en mejor epoca |
|--------|---------|:----:|:----:|
| nnUNet | Both_RealWorld | 150 | 0.7499 |
| nnUNet | Both | 136 | 0.7201 |
| nnUNet | small_tumor | 202 | 0.6890 |
| nnUNet | large_tumor | 45 | 0.8976 |
| Att. U-Net | Both_RealWorld | 253 | 0.4871 |
| Att. U-Net | Both | 394 | 0.4799 |
| Att. U-Net | small_tumor | 286 | 0.3924 |
| Att. U-Net | large_tumor | 462 | 0.6597 |
| U-Net BCE | Both_RealWorld | 199 | 0.7796 |
| U-Net BCE | Both | 200 | 0.8020 |
| U-Net BCE | small_tumor | 284 | 0.5394 |
| U-Net BCE | large_tumor | 300 | 0.8100 |

---

## 7. Estructura de Archivos

```
CRISTINA ALFARO FINALE/
|
|-- Dataset/                          # Datos originales (TIFF stacks)
|   |-- Dataset_Both_RealWorld/       # dbt_001..dbt_120 + real_dbt_*
|   |-- Dataset_Both/                 # dbt_* (tumores grandes + pequenos)
|   |-- Dataset_small_tumor/          # Solo tumores pequenos
|   |-- Dataset_large_tumor/          # Solo tumores grandes
|
|-- Dataset_Preprocessed/             # Datos preprocesados (.npy)
|   |-- Dataset_Both_RealWorld/
|   |   |-- dbt_001/
|   |   |   |-- dbt_001_img.npy       # Volumen normalizado (Z,H,W) float32
|   |   |   |-- dbt_001_mask.npy      # Mascara binaria (Z,H,W) float32
|   |   |-- ...
|   |-- Dataset_Both/
|   |-- Dataset_small_tumor/
|   |-- Dataset_large_tumor/
|
|-- Models/
|   |-- nnUnet_original/
|   |   |-- src/                      # Codigo fuente del modelo
|   |   |   |-- nnunet.py             # Arquitectura nnUNet3D (strides anisotropos, from_plan)
|   |   |   |-- fingerprint.py        # Analisis estadistico del dataset
|   |   |   |-- planner.py            # Auto-planificacion de arquitectura
|   |   |   |-- patch_utils.py        # Extraccion de patches + sliding window inference
|   |   |   |-- postprocess.py        # Filtrado de componentes conectados
|   |   |   |-- train_eval.py         # Entrenamiento/evaluacion (con sliding window)
|   |   |   |-- metrics.py            # 12 metricas de segmentacion
|   |   |   |-- dataset_dbt.py        # Dataset (volumenes completos + PatchDBTDataset)
|   |   |   |-- io_utils.py           # Descubrimiento de estudios
|   |   |   |-- config_loader.py      # Carga de config YAML
|   |   |-- configs/config.yaml       # Hiperparametros + auto-configuracion
|   |   |-- train_unet_dbt.py         # Script principal (fingerprint -> plan -> train)
|   |
|   |-- Attention U_Net/              # Attention U-Net 3D
|   |   |-- src/
|   |   |   |-- attention_unet3d.py   # AttentionUNet3D + AttentionGate3D
|   |   |   |-- ...
|   |   |-- train_unet_dbt.py
|   |
|   |-- U_Net BCE/                    # U-Net con BCE Loss
|   |   |-- src/
|   |   |   |-- unet3d.py             # UNet3D estandar
|   |   |   |-- ...
|   |   |-- train_unet_bce.py
|   |
|   |-- shared/                       # Modulos compartidos
|       |-- visualization.py          # Generacion de PNGs de muestras
|       |-- traceability.py           # run_config.json y run_summary.json
|       |-- augmentations.py          # Aumentaciones de datos
|
|-- outputs_improved/                 # Resultados de los 12 experimentos
|   |-- training_summary.json         # Resumen global con todas las metricas
|   |-- {Modelo}_{Dataset}/
|       |-- checkpoints/
|       |   |-- best.pt               # Mejor checkpoint (por val_dice; incluye plan si auto)
|       |   |-- last.pt               # Ultimo checkpoint
|       |-- logs/
|       |   |-- metrics.csv           # Metricas por epoca
|       |   |-- metrics.jsonl         # Metricas por epoca (JSON lines)
|       |   |-- val_metrics.json      # Metricas finales de validacion (12 metricas)
|       |   |-- test_metrics.json     # Metricas finales de test (12 metricas)
|       |   |-- paper_curves.png      # Graficas principales (2x4 panel)
|       |   |-- paper_curves_extra.png # Graficas HD, FP, AP
|       |   |-- lr_curve.png          # Curva de learning rate
|       |   |-- individual_plots/     # Graficas individuales por metrica
|       |-- preds_npy/                # Predicciones de validacion
|       |   |-- {study_id}_probs.npy  # Probabilidades (float32)
|       |   |-- {study_id}_bin.npy    # Prediccion binaria (0/1)
|       |-- test_preds_npy/           # Predicciones de test
|       |-- samples/                  # PNGs de visualizacion
|       |   |-- val_sample_dbt_XXX.png   # Muestra de validacion
|       |   |-- test_sample_dbt_XXX.png  # Muestra de test
|       |-- fingerprint.json          # [nnUNet] Estadisticas del dataset
|       |-- architecture_plan.json    # [nnUNet] Plan auto-generado (patch, strides, canales)
|       |-- postprocess_config.json   # [nnUNet] min_size optimo para connected components
|       |-- run_config.json           # Configuracion completa del experimento
|       |-- run_summary.json          # Resumen final del entrenamiento
|
|-- scripts/                          # Scripts de preprocesamiento
|   |-- preprocess_dbt.py            # Preprocesamiento principal
|   |-- run_preprocess_all.py        # Ejecucion de preprocesamiento en batch
|   |-- visualize_preprocessed_dataset.py
|   |-- ...
|
|-- run_test_trainings.py             # Orquestador de entrenamientos (12 experimentos)
|-- fix_outputs.py                    # Post-procesamiento de resultados
|-- generate_samples_and_summary.py   # Generacion de muestras y resumen
```



## 8. Referencias

- Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*, 18(2), 203-211.
- Isensee, F., Petersen, J., Klein, A., Zimmerer, D., et al. (2018). nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation. *arXiv preprint arXiv:1809.10486*.
- Oktay, O., et al. (2018). Attention U-Net: Learning Where to Look for the Pancreas. *arXiv preprint arXiv:1804.03999*.
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI 2015*, 234-241.
- Woo, S., et al. (2018). CBAM: Convolutional Block Attention Module. *ECCV 2018*.

