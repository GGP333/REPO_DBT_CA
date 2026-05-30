# Repo_DBT_CA — Contents & Provenance

Support repository for the manuscript *3D Segmentation of DBT Volumes (Digital Breast
Tomosynthesis)* (BMC v10). It contains the **code, configurations, evaluation metrics and
result figures** needed to inspect and reproduce the reported experiments. Large binary
artifacts (raw/preprocessed imaging data, model weights, voxel-level prediction arrays) are
**intentionally excluded** — they are reproducible from the code and are too large / contain
patient-derived data unsuitable for distribution.

## What is included

| Path | Contents |
|------|----------|
| `README.md` | Full project description (architectures, datasets, training, metrics, results). |
| `requirements.txt` | Python dependencies. |
| `src/` | All source code: preprocessing, the 3 in-house architectures (`unet_bce`, `attention_unet`, `nnunet`), shared utilities, evaluation. |
| `experiments/nnunet_official/scripts/` | Official nnU-Netv2 pipeline scripts and custom trainers (`nnUNetTrainer_Seeded42`, etc.). |
| `experiments/nnunet_official/eval/*.json` | Official nnU-Net test results per dataset. |
| `experiments/nnunet_official/nnunet_root/` | Lightweight nnU-Net configs only: `dataset.json`, plans, `splits_final.json`, fingerprints and `summary.json` (validation_best / validation_final / test). |
| `results/outputs_clean/` | **UNet_BCE re-trained without data leakage** for Large / Small / Both (metrics, curves, sample figures, run configs). |
| `results/outputs_improved/` | The other architectures (`Attention_UNet`, `nnUnet_original`) across the 4 datasets, plus the clean `UNet_BCE_Dataset_Both_RealWorld` run. Metrics, training curves, qualitative sample figures and run configs. |
| `docs/data_leakage_audit.md` | Audit report documenting the UNet_BCE leakage and its resolution. |

## What is excluded (and why)

- **`Dataset/`, `Dataset_Preprocessed/`, `data/`** — raw/preprocessed DBT volumes (~15 GB, patient-derived). Regenerate with `src/preprocessing/`.
- **Model weights** (`checkpoints/`, `*.pt`, `*.pth`) — reproducible by training.
- **Prediction arrays** (`preds_npy/`, `test_preds_npy/`, `*.npy`) — reproducible at inference; qualitative `samples/*.png` are kept instead.
- **`nnunet_root/{raw,preprocessed}`** large tensors and `eval/predictions/` NIfTIs.
- **`Experimento_Cristina_Alfaro_Enviar*`** and `*.zip` archives.

## Important caveat — UNet_BCE data leakage

The original UNet_BCE runs for **Large / Small / Both** (in the project's `outputs_improved/`)
suffered 100% train↔test leakage and their test metrics were **invalid**. They are **not**
included here. Instead, `results/outputs_clean/` contains the leakage-free re-trained runs for
those three datasets. The `UNet_BCE_Dataset_Both_RealWorld` run was already clean and is kept
as-is. See `docs/data_leakage_audit.md` for the full analysis.

## Architectures × datasets covered

- Architectures: **UNet_BCE**, **Attention_UNet**, **nnUnet_original** (in-house nnU-Net
  replica) and **official nnU-Netv2**.
- Datasets: **D001 Large**, **D002 Small**, **D003 Both** (synthetic/curated), **D004
  Both_RealWorld** (test set = 100% real clinical cases → out-of-distribution evaluation).
