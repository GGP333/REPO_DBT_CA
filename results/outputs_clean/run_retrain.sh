#!/usr/bin/env bash
# Re-entrena UNet_BCE en Large/Small/Mixed_Size con splits limpios (test = imagesTs).
# Outputs en /home/gabriel/Escritorio/Paper_DBT/outputs_clean/UNet_BCE_Dataset_<short>/
set -euo pipefail

ROOT=/home/gabriel/Escritorio/Paper_DBT
OUT_BASE=$ROOT/outputs_clean
TRAIN=$ROOT/src/models/unet_bce/train_unet_bce.py
CFG=$ROOT/src/models/unet_bce/configs/config.yaml
DATA_BASE=$ROOT/Dataset_Preprocessed

export PYTHONPATH="$ROOT/src/models/unet_bce/src:$ROOT/src/shared:${PYTHONPATH:-}"
cd $ROOT/src/models/unet_bce/src

for spec in \
    "small_tumor:Dataset_small_tumor" \
    "large_tumor:Dataset_large_tumor" \
    "Mixed_Size:Dataset_Mixed_Size" ; do
    short="${spec%:*}"
    folder="${spec#*:}"
    out="$OUT_BASE/UNet_BCE_Dataset_${short}"
    tids="$OUT_BASE/_test_ids/${short}.txt"
    log="$OUT_BASE/UNet_BCE_Dataset_${short}.log"

    if [ -f "$out/run_summary.json" ]; then
        echo "[skip] $short ya tiene run_summary.json"
        continue
    fi

    echo "[run] $short -> $out (log: $log)"
    mkdir -p "$out"
    conda run --no-capture-output -n mamai_2 python "$TRAIN" \
        --config "$CFG" \
        --dataset-root "$DATA_BASE/$folder" \
        --test-ids-file "$tids" \
        --out-dir "$out" \
        2>&1 | tee "$log"
done

echo "=== Todos los runs completados ==="
