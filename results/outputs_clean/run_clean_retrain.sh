#!/usr/bin/env bash
# Re-entrenamiento limpio de UNet_BCE en los 3 datasets contaminados por leakage.
# Split forzado: test = imagesTs de nnUNet (excluido de train/val) vía --test-ids-file.
set -u
cd /home/gabriel/Escritorio/Paper_DBT
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mamai_2
export PYTHONPATH="src/models/unet_bce/src:src/shared"

DATASETS=(small_tumor large_tumor Mixed_Size)

for ds in "${DATASETS[@]}"; do
  out="outputs_clean/UNet_BCE_Dataset_${ds}"
  mkdir -p "$out"
  echo "==================================================================="
  echo "[$(date '+%F %T')] START $ds  -> $out"
  echo "==================================================================="
  python src/models/unet_bce/train_unet_bce.py \
    --config "outputs_clean/_configs/${ds}.yaml" \
    --dataset-root "Dataset_Preprocessed/Dataset_${ds}" \
    --test-ids-file "outputs_clean/_test_ids/${ds}.txt" \
    --out-dir "$out" \
    > "$out/train.log" 2>&1
  rc=$?
  echo "[$(date '+%F %T')] DONE  $ds  (exit=$rc)"
  if [ $rc -ne 0 ]; then
    echo "[$(date '+%F %T')] ABORT: $ds falló (exit=$rc). Revisa $out/train.log" >&2
    exit $rc
  fi
done
echo "[$(date '+%F %T')] TODOS LOS DATASETS COMPLETADOS"
