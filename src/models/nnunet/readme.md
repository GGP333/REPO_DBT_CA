# Auto-configured 3D nnU-Net

Modular pipeline to train a 3D nnU-Net on DBT (Digital Breast Tomosynthesis) volumes
with automatic architecture planning based on dataset properties.

## Overview

This implementation follows the core philosophy of nnU-Net (Isensee et al., 2021):
the architecture is not fixed but **auto-configures** itself from an automatic
analysis of the dataset's statistical fingerprint.

Auto-configuration pipeline:

1. **Dataset fingerprint** (`src/fingerprint.py`): median/min/max shape,
   foreground/background ratio, intensity statistics.
2. **Architecture planner** (`src/planner.py`): computes patch size, anisotropic
   strides per level, channels per level, and batch size based on available GPU VRAM.
3. **Model construction** (`src/nnunet.py`): `nnUNet3D.from_plan(plan)` with
   residual blocks, instance normalization, LeakyReLU, anisotropic `ConvTranspose3d`,
   and gradient checkpointing.

Training is patch-based with foreground oversampling. Inference uses sliding
window with Gaussian weighting plus connected-components post-processing
(`src/postprocess.py`).

## Data layout

Preprocessed DBT volumes are expected as `.npy` files under:

```
data/preprocessed/<Dataset>/<study_id>/
    <study_id>_img.npy    # float32, shape (Z, H, W)
    <study_id>_mask.npy   # uint8,   shape (Z, H, W)
```

See `data/preprocessed/README.md` for details.

## Training

```bash
python src/models/nnunet/train_unet_dbt.py \
    --config src/models/nnunet/configs/config.yaml
```

Configuration is loaded from `configs/config.yaml`. Key parameters:

- `auto_plan: true` enables fingerprint + planner.
- `gpu_vram_gb: 16.0` is the assumed GPU memory budget.
- `oversample_foreground: 0.33` is the probability of forcing patch centers on
  foreground voxels during training.

## Outputs

- `checkpoints/best.pt`: best model by validation Dice (includes the plan
  if auto-configured).
- `checkpoints/last.pt`: last epoch checkpoint.
- `logs/metrics.csv`, `logs/metrics.jsonl`: per-epoch metrics.
- `logs/val_metrics.json`, `logs/test_metrics.json`: final 12 metrics.
- `preds_npy/`: validation predictions (`*_probs.npy`, `*_bin.npy`).
- `test_preds_npy/`: test predictions.
- `fingerprint.json`, `architecture_plan.json`, `postprocess_config.json`:
  auto-configuration artifacts.

## See also

The top-level `README.md` contains the full project documentation, including
architecture comparison, training details, and results tables.
