# U-Net 3D with BCE Loss

Modular pipeline to train a 3D U-Net on DBT (Digital Breast Tomosynthesis)
volumes using Binary Cross-Entropy as the optimization objective. Based on the
original U-Net by Ronneberger et al. (MICCAI 2015; arXiv:1505.04597), extended
to 3D.

## Overview

Canonical encoder-decoder with skip connections:

- **Encoder**: 3D convolutional blocks + `MaxPool3d(2)`.
- **Bottleneck**: widest-channel block (512) integrating global volume
  information.
- **Decoder**: trilinear upsampling, fusion with encoder skip connections,
  convolutional refinement. Center cropping is applied on skips when odd-size
  mismatches appear.
- **Output head**: `Conv3d 1x1 + sigmoid`.

**Key objective difference**: this model optimizes **Binary Cross-Entropy with
logits** instead of Dice + CE. BCE penalizes voxel-wise error directly and is
numerically stable under AMP.

~23.5M parameters.

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
python src/models/unet_bce/train_unet_bce.py \
    --config src/models/unet_bce/configs/config.yaml
```

Default training settings:
- 500 epochs, AdamW (lr=1e-3, wd=1e-5), polynomial LR decay, 5-epoch warmup.
- BCE-with-logits loss (stable under AMP).
- AMP enabled, gradient clipping max_norm=1.0, binarization threshold 0.5.
- Foreground oversampling = 0.5.

## Outputs

- `checkpoints/best.pt`, `checkpoints/last.pt`
- `logs/metrics.csv`, `logs/metrics.jsonl`
- `logs/val_metrics.json`, `logs/test_metrics.json`
- `preds_npy/`, `test_preds_npy/`
- `samples/`: validation / test sample PNGs

## See also

The top-level `README.md` contains the full project documentation, including
architecture comparison, training details, and results tables.
