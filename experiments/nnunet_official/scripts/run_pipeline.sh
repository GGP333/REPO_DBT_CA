#!/usr/bin/env bash
# End-to-end pipeline for one dataset id (1..4).
#
# Usage:
#     ./run_pipeline.sh <DATASET_ID> [<TRAINER>] [<FOLD>]
#         DATASET_ID  : 1, 2, 3 or 4
#         TRAINER     : one of
#                         canonical_seeded       -> nnUNetTrainer_Seeded42 (Isensee canonical + seed 42)
#                                                   ** default **
#                         canonical_seeded_es    -> nnUNetTrainer_Seeded42_ES (canonical + seed 42 +
#                                                   early stopping; currently buggy, do not use)
#                         canonical              -> nnUNetTrainer (Isensee defaults, no explicit seed)
#                         paper_standardized     -> nnUNetTrainer_PaperStandardized (AdamW/500ep/BS=1)
#                         <N>epochs              -> nnUNetTrainer_<N>epochs (smoke tests)
#                       default: canonical_seeded
#         FOLD        : default 0
#
# Expects the conda env `nnunet_official` to exist and the custom trainers to
# be installed (`python install_trainers.py`).

set -euo pipefail

DSID="${1:?usage: run_pipeline.sh <DATASET_ID> [<TRAINER>] [<FOLD>]}"
TRAINER_ALIAS="${2:-canonical_seeded}"
FOLD="${3:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/env.sh"

CONDA_RUN=(conda run --no-capture-output -n nnunet_official)

# ---- Resolve trainer alias -> trainer class name ----
case "${TRAINER_ALIAS}" in
    canonical)             TRAINER="nnUNetTrainer" ;;
    canonical_seeded)      TRAINER="nnUNetTrainer_Seeded42" ;;
    canonical_seeded_es)   TRAINER="nnUNetTrainer_Seeded42_ES" ;;
    paper_standardized)    TRAINER="nnUNetTrainer_PaperStandardized" ;;
    *epochs)               TRAINER="nnUNetTrainer_${TRAINER_ALIAS}" ;;
    *)                     TRAINER="${TRAINER_ALIAS}" ;;  # accept raw class name
esac

DATASET_NAME=$("${CONDA_RUN[@]}" python -c "
import sys; sys.path.insert(0, '${SCRIPT_DIR}')
from datasets import dataset_folder
print(dataset_folder(${DSID}))
")

echo
echo "=================================================================="
echo " Dataset:  ${DATASET_NAME}"
echo " Trainer:  ${TRAINER}  (alias: ${TRAINER_ALIAS})"
echo " Fold:     ${FOLD}"
echo "=================================================================="

# ---- 0. Make sure custom trainers are installed ----
"${CONDA_RUN[@]}" python "${SCRIPT_DIR}/install_trainers.py" >/dev/null

# ---- 1. Convert .npy -> NIfTI ----
if [[ ! -d "${nnUNet_raw}/${DATASET_NAME}/imagesTr" ]]; then
    echo "[1/6] Converting .npy -> NIfTI"
    "${CONDA_RUN[@]}" python "${SCRIPT_DIR}/01_convert_to_nifti.py" \
        --nnunet-raw "${nnUNet_raw}" --datasets "${DSID}"
else
    echo "[1/6] NIfTI already converted -> skip"
fi

# ---- 2. plan_and_preprocess ----
if [[ ! -f "${nnUNet_preprocessed}/${DATASET_NAME}/nnUNetPlans.json" ]]; then
    echo "[2/6] nnUNetv2_plan_and_preprocess"
    "${CONDA_RUN[@]}" nnUNetv2_plan_and_preprocess -d "${DSID}" --verify_dataset_integrity
else
    echo "[2/6] Plan already exists -> skip"
fi

# ---- 3. splits_final.json (repo split for honest comparison) ----
echo "[3/6] Writing splits_final.json"
"${CONDA_RUN[@]}" python "${SCRIPT_DIR}/02_make_splits.py" \
    --nnunet-preprocessed "${nnUNet_preprocessed}" --datasets "${DSID}"

# ---- 4. Train (--c lets us resume from checkpoint_latest.pth if interrupted) ----
echo "[4/6] nnUNetv2_train (3d_fullres, ${TRAINER}, fold=${FOLD}, --c)"
if [[ "${TRAINER}" == "nnUNetTrainer" ]]; then
    "${CONDA_RUN[@]}" nnUNetv2_train "${DSID}" 3d_fullres "${FOLD}" --c
else
    "${CONDA_RUN[@]}" nnUNetv2_train "${DSID}" 3d_fullres "${FOLD}" -tr "${TRAINER}" --c
fi

# ---- 5. Predict on imagesTs ----
PRED_DIR="${EXP_ROOT}/eval/predictions/${DATASET_NAME}__${TRAINER}"
mkdir -p "${PRED_DIR}"
echo "[5/6] nnUNetv2_predict -> ${PRED_DIR}"
PRED_FLAGS=(-i "${nnUNet_raw}/${DATASET_NAME}/imagesTs" -o "${PRED_DIR}"
            -d "${DSID}" -c 3d_fullres -f "${FOLD}"
            -chk checkpoint_best.pth)   # paper P55: select best by val Dice
if [[ "${TRAINER}" != "nnUNetTrainer" ]]; then
    PRED_FLAGS+=(-tr "${TRAINER}")
fi
"${CONDA_RUN[@]}" nnUNetv2_predict "${PRED_FLAGS[@]}"

# ---- 6. Evaluate ----
RESULTS_JSON="${EXP_ROOT}/eval/results_${DATASET_NAME}__${TRAINER}.json"
echo "[6/6] Evaluating -> ${RESULTS_JSON}"
"${CONDA_RUN[@]}" python "${SCRIPT_DIR}/06_evaluate.py" \
    --pred-dir "${PRED_DIR}" \
    --gt-dir "${nnUNet_raw}/${DATASET_NAME}/labelsTs" \
    --out-json "${RESULTS_JSON}"

echo
echo "Done: ${DATASET_NAME} with ${TRAINER}"
