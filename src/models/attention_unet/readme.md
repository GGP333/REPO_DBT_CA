# Attention U-Net 3D

Modular pipeline to train a 3D Attention U-Net on DBT (Digital Breast
Tomosynthesis) volumes. Inspired by Oktay et al. (arXiv:1804.03999).

## Overview

This architecture extends the classic 3D U-Net with **Attention Gates (AGs)**
on the skip connections, which spatially weight encoder features by their
relevance to the target lesion before they are concatenated in the decoder.

Additional DBT-oriented components:

- **`DBTSpecificBlock3D`** multi-scale block (`1x1x1`, `3x3x3`, `5x5x5`) that
  combines fine and coarse tissue patterns.
- **3D CBAM** (channel + spatial attention) in key blocks for feature
  recalibration.
- **`ConvBlock3D` with Dropout3D** for regularization.
- **Trilinear upsampling + attention gate + concatenation + convolution** in
  the decoder.
- **Gradient checkpointing** to fit the larger parametric footprint (~35.6M
  parameters).

## Attention gate formulation

For each skip connection:

```
alpha        = sigmoid( psi( ReLU( Wg(gate) + Wx(skip) ) ) )
attended     = skip * alpha
decoder_in   = concat(upsampled_decoder, attended)
```

Where `gate` comes from the lower-resolution decoder path (more semantic) and
`skip` comes from the encoder path (more spatial detail).

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
python src/models/attention_unet/train_unet_dbt.py \
    --config src/models/attention_unet/configs/config.yaml
```

Default training settings match the other models:
- 500 epochs, AdamW (lr=1e-3, wd=1e-5), polynomial LR decay, 5-epoch warmup.
- Dice + BCE loss (0.5 / 0.5).
- AMP enabled, gradient clipping max_norm=1.0, binarization threshold 0.5.

## Outputs

- `checkpoints/best.pt`, `checkpoints/last.pt`
- `logs/metrics.csv`, `logs/metrics.jsonl`
- `logs/val_metrics.json`, `logs/test_metrics.json`
- `preds_npy/`, `test_preds_npy/`
- `samples/`: validation / test sample PNGs
- Attention map visualizations can be produced via `src/attention_viz.py`.

## See also

The top-level `README.md` contains the full project documentation, including
architecture comparison, training details, and results tables.
