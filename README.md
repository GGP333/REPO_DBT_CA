# 3D Segmentation of DBT Volumes (Digital Breast Tomosynthesis)

## 1. Project Overview

This project implements and compares three 3D convolutional neural network architectures for **automatic breast tumor segmentation** in Digital Breast Tomosynthesis (DBT) volumes. The goal is to evaluate the performance of different U-Net variants on the task of voxel-wise binary segmentation, using multiple data configurations that include tumors of different sizes and real vs. synthetic data.

### Main contributions

- Systematic comparison of 3 architectures across 4 data configurations: **nnU-Net** (official
  nnU-Netv2), **3D U-Net (baseline)** (U-Net BCE), and **Attention U-Net 3D**.
- Use of the official **nnU-Netv2** framework (Isensee et al.) as the auto-configuring
  reference model — dataset fingerprint, automatic architecture planning, patch-based training,
  sliding window inference. The official pipeline lives in `experiments/nnunet_official/`.
- Evaluation with 12 segmentation metrics including Hausdorff Distance 95 (HD95), Average Precision (AP), and False Positives per volume.
- Complete pipeline from TIFF preprocessing to quantitative evaluation and prediction visualization.
- 500-epoch training with guaranteed reproducibility (seed=42).

---

## 2. Datasets

The data comes from 3D DBT volumes, originally stored as TIFF image stacks (one image per axial slice). Preprocessing converts them into NumPy `.npy` arrays with shape `Z x 256 x 512` (Z x H x W), where **Z is kept variable** for each study.

### 2.1 Preprocessing

The preprocessing used for training/evaluating the reported experiments corresponds to `src/preprocessing/preprocess_simple.py`. This pipeline is deliberately minimalist and applies exactly the following steps:

1. **Load study and GT mask**
   - Reads the image TIFF stack (`img_*`) and reconstructs a 3D volume in `(Z, Y, X)` format.
   - Reads the lesion mask from `mask_*` (or `--mask_dir`) and verifies that it has the same shape as the image.
   - If an extra singleton dimension arrives (e.g., `(1, Z, Y, X)`), it is squeezed to maintain a valid 3D volume.

2. **Intensity clipping**
   - Applies percentile-based clipping with `--clip_percentiles` (default `[0.5, 99.5]`).
   - This step reduces extreme values before normalization.

3. **Image normalization**
   - `--normalize zscore` (default): standardizes using the mean and standard deviation of the full volume.
   - `--normalize minmax`: scales to `[0, 1]`.
   - `--normalize none`: preserves intensities after clipping.

4. **XY-plane resize**
   - Resizes only XY to `--target_size_xy` (default `512 256`) and leaves `Z` unchanged.
   - Linear interpolation for the continuous image.
   - Nearest-neighbor interpolation for the mask, followed by binarization (`> 0.5`) to preserve discrete labels.

5. **Artifact saving**
   - Saves:
     - `{study_id}_img.npy` (`float32`)
     - `{study_id}_mask.npy` (`uint8`)
   - Records `preprocess_params.json` with the parameters used (original shape, resize, clipping, and normalization mode).

### 2.2 Data configurations

Four datasets are defined from the preprocessed set:

| Dataset | Description | Studies in directory | Studies used | Train | Val | Test |
|---------|-------------|:---:|:---:|:---:|:---:|:---:|
| **Dataset_large_tumor** | Only large tumors (high fg/bg ratio) | 90 | 90 | 72 | 18 | 8 |
| **Dataset_small_tumor** | Only small tumors (low fg/bg ratio) | 30 | 30 | 24 | 6 | 3 |
| **Dataset_Mixed_Size** | Mixture of large and small tumors (90 large + 30 small) | 120 | 120 | 96 | 24 | 11 |
| **Dataset_Hybrid** | `dbt_*` + `real_dbt_*` (real clinical data) | 140 | 140 | ~104 | ~26 | 10 (real_dbt_*) |

**Note on Train/Val/Test splits**: the values in the table correspond to the split used by UNet_BCE and Attention U-Net (80/20 train/val). nnUNet uses a slightly different split (e.g., 65/16/8 on large_tumor, 87/22/11 on Mixed_Size). In both cases, the test set is the same (8, 3, 11 studies respectively) and is evaluated with the best checkpoint.

**Note on Dataset_Hybrid**: the directory contains 140 studies (120 `dbt_*` + 20 `real_dbt_*`). The test set is built with **50% of the `real_dbt_*`** (10 real clinical studies). The remaining 50% of `real_dbt_*` is mixed with the `dbt_*` to form train/val. This allows the model to learn from real and synthetic data, and to be evaluated exclusively on unseen real clinical data.

### 2.3 Train/val/test split

- **Fixed seed**: 42 (full reproducibility).
- **Study-level split**: partitioning is done at the study level (not by slice), preventing data leakage.
- **General case (without `real_dbt_*`)**: uses the configured train/val/test proportions over the total dataset.
- **Special case with real data (`real_dbt_*`)**: 50% of the `real_dbt_*` studies are assigned exclusively to the **test** set. The remaining 50% is mixed with the `dbt_*` to form train/val. This way the model trains with real and synthetic data and is evaluated only on unseen real clinical data.
- This logic is implemented in `src/models/attention_unet/train_unet_dbt.py` and
  `src/models/unet_bce/src/dataset_dbt.py`. The official nnU-Net reproduces the same split via
  `experiments/nnunet_official/scripts/02_make_splits.py` (see `splits_final.json`).

---

## 3. Model Architectures

Three 3D U-Net variants are compared for volumetric segmentation. Conceptually, all three share the encoder-decoder pattern with skip connections, but differ in how they model context, how they fuse multi-scale information, how they adapt to the data geometry, and which optimization objective they prioritize.

### 3.1 nnU-Net (official nnU-Netv2)

**Reference auto-configuring model.** The manuscript uses the official **nnU-Netv2** framework
by Isensee et al. (Nature Methods 2021) as the strong, self-configuring baseline. nnU-Netv2
analyses the dataset (fingerprint), automatically plans the 3D full-resolution architecture
(patch size, anisotropic pooling adapted to the DBT Z/H/W geometry, channel widths and batch
size), trains with patch-based foreground oversampling, and runs sliding-window inference with
Gaussian weighting.

The complete, reproducible pipeline (NIfTI conversion, splits, custom seeded trainer, training
and evaluation) is in **`experiments/nnunet_official/`**:

- `scripts/` — `01_convert_to_nifti.py`, `02_make_splits.py`, `06_evaluate.py`, and the custom
  trainer `nnUNetTrainer_Seeded42` (fixed seed = 42).
- `nnunet_root/` — `dataset.json`, plans, `splits_final.json`, fingerprints and the
  `validation_best` / `validation_final` summaries per dataset.
- `eval/results_Dataset00*__nnUNetTrainer_Seeded42.json` — test metrics, computed with the same
  `compute_metrics` used for the other models so the numbers are 1:1 comparable.

> An earlier *in-house* nnU-Net replica (`nnUnet_original`) was used during development but was
> **superseded by the official nnU-Netv2** and is not part of the manuscript. Its code and
> outputs are kept under `_descartado/` for traceability only.

### 3.2 Attention U-Net 3D (`src/models/attention_unet`)

**Parameters**: 35.575M

Architecture inspired by the Attention U-Net of Ozan Oktay et al. (arXiv:1804.03999), whose central contribution is the introduction of **Attention Gates (AGs)** in the skip connections to suppress irrelevant activations and emphasize candidate regions for the target.

The key idea of the AGs in this repository is:

- **Gate signal (`gate`)**: comes from the decoder (more semantic, lower resolution).
- **Skip signal (`skip`)**: comes from the encoder (more spatial detail, less semantic).
- **Additive fusion + activation**:
  - `Wg(gate) + Wx(skip) -> ReLU -> psi(1x1x1) -> sigmoid = alpha`
  - `attended_skip = skip * alpha`

Thus, the decoder does not concatenate "the entire" skip, but a version spatially weighted by relevance to the lesion.

In addition to the base AGs from Oktay, this implementation introduces extra DBT-oriented components:

- **Multi-scale `DBTSpecificBlock3D`** (`1x1x1`, `3x3x3`, `5x5x5`) to combine fine and coarse tissue patterns.
- **3D CBAM** (channel + spatial) in key blocks, reinforcing feature recalibration.
- **`ConvBlock3D` with Dropout3D**: improves regularization in a high-capacity model.
- **Decoder with trilinear upsample + AG + concatenation + convolution**.
- **Checkpointing** to make its higher parametric complexity feasible.

Compared with a classical U-Net, this variant aims to improve sensitivity on small or ambiguous structures by better filtering background anatomical noise.

### 3.3 U-Net BCE (`src/models/unet_bce`)

**Parameters**: 23.535M

Based on the original U-Net by Ronneberger et al. (MICCAI 2015; arXiv:1505.04597), extended to 3D for volumetric segmentation.

The structure follows the canonical scheme:

- **Encoder (contraction)**: 3D convolutional blocks + `MaxPool3d(2)` to compress resolution and increase context.
- **Bottleneck**: block with the largest channel width (512) where global volume information is integrated.
- **Decoder (expansion)**: progressive trilinear upsampling, fusion with encoder skip connections, and convolutional refinement.
- **Spatial skip alignment**: when differences appear due to odd sizes, center cropping is applied to concatenate without errors.
- **Output head**: `Conv3d 1x1 + sigmoid` for a binary probability map.

Here the main difference is not only architectural, but objective-wise:

- **BCE Loss**: penalizes voxel-to-voxel error directly and stably.
- In contrast, nnUNet and Attention U-Net in this project use Dice+CE-like configurations to emphasize overlap and balance the minority class.

This variant serves as a strong baseline: fewer advanced mechanisms than Attention U-Net, but with clear inductive bias, stable implementation, and good generalization capacity.

### 3.4 Architecture comparison

| Feature | nnU-Net (v2) | Attention U-Net | U-Net BCE |
|---------------|:---------:|:---------------:|:---------:|
| Parameters (M) | 31.3 (auto) | 35.575 | 23.535 |
| Auto-configuration | Yes (fingerprint + planner) | No | No |
| Training | Patch-based | Full volume | Full volume |
| Inference | Sliding window + Gaussian | Direct forward | Direct forward |
| Pooling | Anisotropic (adapted to Z/H/W) | Isotropic (stride 2) | Isotropic (stride 2) |
| Post-processing | Auto connected components | No | No |
| Normalization | InstanceNorm3D | InstanceNorm3D | InstanceNorm3D |
| Activation | LeakyReLU | ReLU | ReLU |
| Residual blocks | Yes | No | No |
| Attention gates | No | Yes | No |
| Upsampling | ConvTranspose3d (aniso) | Trilinear | Trilinear |
| Dropout | No | Yes (0.1) | No |
| Loss function | Dice + CE | Dice + CE | BCE |
| Gradient checkpointing | Yes | Yes | No |

---

## 4. Training Configuration

The two in-house-trained models (**3D U-Net baseline** and **Attention U-Net**) share the base
configuration below, defined in `configs/config.yaml`. The **official nnU-Net** follows
nnU-Netv2's own self-configured defaults (notably **1000 epochs**); see
`experiments/nnunet_official/` for its exact plans and trainer.

| Parameter | Value |
|-----------|-------|
| **Seed** | 42 |
| **Epochs** | 500 |
| **Optimizer** | AdamW (lr=0.001, weight_decay=1e-5) |
| **Scheduler** | Polynomial LR Decay: `lr * (1 - epoch/max_epochs)^0.9` |
| **Warmup** | 5 epochs (linear warmup) |
| **Batch size** | 1 |
| **AMP** | Enabled (Mixed Precision with GradScaler) |
| **Gradient clipping** | max_norm=1.0 |
| **Binarization threshold** | 0.5 |
| **Foreground oversampling** | 0.5 |
| **Augmentation** | Enabled |
| **Early stopping** | Disabled (full 500 epochs are trained) |

**Official nnU-Net:** trained with the **nnU-Netv2** framework using its self-configured 3D
full-resolution plans (auto patch size, anisotropic pooling, patch-based foreground
oversampling, sliding-window Gaussian inference) and a custom seeded trainer
(`nnUNetTrainer_Seeded42`). See `experiments/nnunet_official/nnunet_root/.../plans.json`.

### 4.1 Loss functions

- **Dice + CE** (nnU-Net, Attention U-Net): `0.5 * DiceLoss + 0.5 * BCE`. Combines direct optimization of the Dice coefficient with BCE stability to handle extreme class imbalance.
- **BCE** (U-Net BCE): Only Binary Cross-Entropy with logits (numerically stable).

### 4.2 Checkpoint saving

- **best.pt**: Best model by val_dice (saved when it improves with min_delta=0.001).
- **last.pt**: Last checkpoint of each epoch (for recovery).

### 4.3 Hardware

- **GPU**: NVIDIA GeForce RTX 4070 Ti SUPER
- **CUDA**: 12.4
- **PyTorch**: 2.5.1+cu124
- **System**: Linux 6.17.9 (Arch)

### 4.4 Training time

Indicative wall-clock from the development machine (RTX 4070 Ti SUPER). The official nnU-Net was
trained separately with nnU-Netv2 (1000 epochs) and is not included in this table.

| Model | Dataset | Time |
|--------|---------|--------|
| Attention_UNet | Hybrid | 5.7h |
| Attention_UNet | Mixed_Size | 5.7h |
| Attention_UNet | small_tumor | 1.4h |
| Attention_UNet | large_tumor | 4.3h |
| UNet_BCE | Hybrid | 2.8h |
| UNet_BCE | Mixed_Size | 2.8h |
| UNet_BCE | small_tumor | 0.7h |
| UNet_BCE | large_tumor | 2.1h |
| **Total (in-house models)** | | **~25.5h** |

---

## 5. Evaluation Metrics

Twelve segmentation metrics are computed for each experiment. All metrics are evaluated on the best checkpoint (best.pt) on the validation and test sets.

### 5.1 Overlap metrics

| Metric | Formula | Range | Ideal |
|---------|---------|:-----:|:-----:|
| **Dice Coefficient** | 2\|P intersect G\| / (\|P\| + \|G\|) | [0, 1] | 1 |
| **IoU (Jaccard)** | \|P intersect G\| / \|P union G\| | [0, 1] | 1 |
| **Precision** | TP / (TP + FP) | [0, 1] | 1 |
| **Recall** | TP / (TP + FN) | [0, 1] | 1 |
| **F1 Score** | 2 * Precision * Recall / (Precision + Recall) | [0, 1] | 1 |
| **Accuracy** | (TP + TN) / Total voxels | [0, 1] | 1 |

### 5.2 Distance metrics

| Metric | Description | Range | Ideal |
|---------|-------------|:-----:|:-----:|
| **Hausdorff Distance (HD)** | Maximum bidirectional distance between prediction and GT surfaces. Computed on surface points extracted by morphological erosion. | [0, inf) | 0 |
| **HD95** | 95th percentile of combined surface distances. More robust to outliers than HD. | [0, inf) | 0 |

### 5.3 Detection metrics

| Metric | Description | Range | Ideal |
|---------|-------------|:-----:|:-----:|
| **FP/Volume** | Proportion of false-positive voxels relative to the total volume. | [0, 1] | 0 |
| **FP/Image** | Mean FP per axial slice (normalized by slice area). | [0, 1] | 0 |
| **AP (Average Precision)** | Area under the Precision-Recall curve while varying the probability threshold (19 thresholds from 0.05 to 0.95). Uses monotonic interpolation. | [0, 1] | 1 |
| **mAP** | COCO-style Mean AP: average AP over 10 IoU thresholds (0.5 to 0.95 in steps of 0.05). | [0, 1] | 1 |

---

## 6. Results

These are the numbers reported in the manuscript. **nnU-Net** = official nnU-Netv2
(`nnUNetTrainer_Seeded42`); **3D U-Net (baseline)** = U-Net BCE; **Attention U-Net** =
Attention U-Net 3D. For U-Net BCE on Large / Small / Mixed_Size these are the **leakage-free
re-trained** runs (`results/outputs_clean/`); see `docs/data_leakage_audit.md`. All metrics
use the best checkpoint.

### 6.1 Validation metrics (best checkpoint)

Dice and IoU are reported for all three models (the official nnU-Netv2 validation summary
stores only these two natively).

| Model | Dataset | Dice | IoU |
|--------|---------|:----:|:---:|
| nnU-Net | large_tumor | 0.8922 | 0.8081 |
| nnU-Net | small_tumor | 0.7358 | 0.6208 |
| nnU-Net | Mixed_Size | 0.7716 | 0.6746 |
| nnU-Net | Hybrid | 0.7308 | 0.6428 |
| 3D U-Net (baseline) | large_tumor | 0.7873 | 0.6591 |
| 3D U-Net (baseline) | small_tumor | 0.5361 | 0.3931 |
| 3D U-Net (baseline) | Mixed_Size | 0.7003 | 0.5895 |
| 3D U-Net (baseline) | Hybrid | 0.7317 | 0.6325 |
| Attention U-Net | large_tumor | 0.6597 | 0.5138 |
| Attention U-Net | small_tumor | 0.3924 | 0.2740 |
| Attention U-Net | Mixed_Size | 0.4799 | 0.3663 |
| Attention U-Net | Hybrid | 0.5838 | 0.4579 |

### 6.2 Test metrics (best checkpoint)

| Model | Dataset | Dice | IoU | Precision | Recall | HD95 | AP | mAP |
|--------|---------|:----:|:---:|:---------:|:------:|:----:|:--:|:---:|
| nnU-Net | large_tumor | 0.8641 | 0.7661 | 0.8828 | 0.8612 | 48.50 | 0.7569 | 0.4639 |
| nnU-Net | small_tumor | 0.5602 | 0.4758 | 0.6739 | 0.5739 | 34.37 | 0.4592 | 0.2133 |
| nnU-Net | Mixed_Size | 0.8414 | 0.7372 | 0.8942 | 0.8082 | 2.17 | 0.7241 | 0.4151 |
| nnU-Net | Hybrid | 0.3525 | 0.2786 | 0.5862 | 0.3647 | 141.22 | 0.2700 | 0.0761 |
| 3D U-Net (baseline) | large_tumor | 0.8295 | 0.7167 | 0.8410 | 0.8462 | 3.46 | 0.8769 | 0.4689 |
| 3D U-Net (baseline) | small_tumor | 0.2635 | 0.1979 | 0.2337 | 0.3319 | 40.00 | 0.2655 | 0.0945 |
| 3D U-Net (baseline) | Mixed_Size | 0.7264 | 0.6084 | 0.7282 | 0.7383 | 18.07 | 0.7536 | 0.3308 |
| 3D U-Net (baseline) | Hybrid | 0.4116 | 0.2982 | 0.5416 | 0.4731 | 81.15 | 0.3959 | 0.0638 |
| Attention U-Net | large_tumor | 0.7776 | 0.6434 | 0.7661 | 0.8124 | 29.78 | 0.7028 | 0.2529 |
| Attention U-Net | small_tumor | 0.3670 | 0.2692 | 0.3047 | 0.5045 | 53.90 | 0.2795 | 0.0405 |
| Attention U-Net | Mixed_Size | 0.5823 | 0.4334 | 0.6036 | 0.6458 | 54.42 | 0.4564 | 0.0888 |
| Attention U-Net | Hybrid | 0.4207 | 0.2941 | 0.4098 | 0.4879 | 68.66 | 0.2965 | 0.0174 |

Per-case test results for the official nnU-Net live in
`experiments/nnunet_official/eval/results_Dataset00*__nnUNetTrainer_Seeded42.json`; for the
other two models in `results/outputs_*/<run>/logs/test_metrics.json`.

---

## 7. Repository Structure

```
Paper_DBT/
|
|-- data/                             # Data (excluded from git)
|   |-- raw/                          # Original TIFF data
|   |   |-- README.md                 # Instructions to obtain the data
|   |-- preprocessed/                 # Preprocessed .npy data
|       |-- README.md                 # Preprocessing instructions
|
|-- src/                              # Source code (in-house models)
|   |-- models/
|   |   |-- attention_unet/           # Attention U-Net 3D
|   |   |   |-- src/                  # attention_unet3d.py, metrics.py, ...
|   |   |   |-- configs/config.yaml
|   |   |   |-- train_unet_dbt.py
|   |   |
|   |   |-- unet_bce/                 # 3D U-Net baseline (BCE loss)
|   |       |-- src/                  # unet3d.py, dataset_dbt.py, metrics.py, ...
|   |       |-- configs/config.yaml
|   |       |-- train_unet_bce.py
|   |
|   |-- shared/                       # Modules shared between models
|   |   |-- visualization.py          # Sample PNG generation
|   |   |-- traceability.py           # run_config.json and run_summary.json
|   |   |-- augmentations.py          # Data augmentations
|   |
|   |-- preprocessing/                # Preprocessing pipeline (preprocess_simple.py, ...)
|   |
|   |-- run_test_trainings.py         # Orchestrator (in-house models)
|   |-- generate_samples_and_summary.py  # Sample and summary generation
|   |-- plot_training_curves.py       # Training-loss figure (all 3 models, per dataset)
|
|-- experiments/
|   |-- nnunet_official/              # Official nnU-Netv2 pipeline (the manuscript's "nnU-Net")
|       |-- scripts/                  # 01_convert_to_nifti, 02_make_splits, 06_evaluate,
|       |   |                         #   metrics.py, custom trainers (nnUNetTrainer_Seeded42)
|       |-- nnunet_root/              # dataset.json, plans, splits_final, fingerprints, summaries
|       |-- eval/                     # results_Dataset00*__nnUNetTrainer_Seeded42.json (test)
|
|-- results/
|   |-- outputs_clean/                # UNet_BCE leakage-free re-trains (Large/Small/Mixed_Size)
|   |-- outputs_improved/             # Attention_UNet (4 datasets) + UNet_BCE Hybrid
|   |-- figures/                      # training_curves_<dataset>.png (per-dataset loss curves)
|
|-- docs/
|   |-- data_leakage_audit.md         # Leakage audit and resolution
|
|-- _descartado/                      # Not used in the manuscript (in-house nnU-Net replica, etc.)
|
|-- REPO_CONTENTS.md                  # Contents & provenance
|-- requirements.txt                  # Python dependencies
|-- README.md                         # This file
```

### 7.1 Workflow

```
1. Obtain data         -> data/raw/
2. Preprocess          -> python src/preprocessing/preprocess_simple.py
3. Train models        -> in-house: src/run_test_trainings.py
                          nnU-Net:  experiments/nnunet_official/scripts/run_pipeline.sh
4. Inspect results     -> results/outputs_clean/, results/outputs_improved/, results/figures/
```

### 7.2 Files excluded from the repository

Due to their size, the following items are NOT included in the repository (see `.gitignore`):

- `data/raw/` and `data/preprocessed/`: medical data (privacy + size)
- `results/checkpoints/`: trained models (`.pt` files)
- `results/predictions/`: predictions in `.npy` format
- `results/logs/*.png`: generated plots

To reproduce the results, follow the instructions in Section 8.

---

## 8. Installation and Usage

### 8.1 Requirements

- Python >= 3.10
- CUDA 12.1+ (recommended for GPU)
- NVIDIA GPU with at least 8GB VRAM (16GB recommended for auto-configured nnUNet)

### 8.2 Installation

```bash
git clone https://github.com/<user>/<repository>.git
cd <repository>

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 8.3 Prepare the data

1. Place the original DBT data in `data/raw/` following the structure described in `data/raw/README.md`.
2. Run preprocessing:

```bash
python src/preprocessing/preprocess_simple.py \
    --data_dir data/raw/Dataset_large_tumor \
    --output_dir data/preprocessed/Dataset_large_tumor \
    --target_size_xy 512 256 \
    --normalize zscore
```

Or process all datasets in batch:

```bash
python src/preprocessing/run_preprocess_all.py
```

### 8.4 Train models

Train the two in-house models (3D U-Net baseline and Attention U-Net):

```bash
# Attention U-Net
python src/models/attention_unet/train_unet_dbt.py \
    --config src/models/attention_unet/configs/config.yaml

# U-Net BCE (3D U-Net baseline)
python src/models/unet_bce/train_unet_bce.py \
    --config src/models/unet_bce/configs/config.yaml
```

Train / evaluate the official **nnU-Net** (nnU-Netv2) via its dedicated pipeline:

```bash
# See experiments/nnunet_official/scripts/run_pipeline.sh for the full sequence:
#   01_convert_to_nifti.py -> 02_make_splits.py -> nnUNetv2_train (nnUNetTrainer_Seeded42)
#   -> 06_evaluate.py
bash experiments/nnunet_official/scripts/run_pipeline.sh
```

### 8.5 Evaluate and generate visualizations

```bash
python src/generate_samples_and_summary.py
```

Training-loss curves (one figure per dataset, three panels: 3D U-Net / nnU-Net / Attention
U-Net) are written to `results/figures/`:

```bash
python src/plot_training_curves.py --datasets large_tumor small_tumor Mixed_Size Hybrid
```

Curve sources match the reported runs: the official **nnU-Net** loss is read from its
nnU-Netv2 `training_log_*.txt` (compound DC+CE loss, can be negative; 1000 epochs by default —
the Large run's persisted log resumes at epoch 350), while the 3D U-Net baseline and Attention
U-Net read `train_loss` from their `logs/metrics.csv` (500 epochs).

---

## 9. References

- Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*, 18(2), 203-211.
- Isensee, F., Petersen, J., Klein, A., Zimmerer, D., et al. (2018). nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation. *arXiv preprint arXiv:1809.10486*.
- Oktay, O., et al. (2018). Attention U-Net: Learning Where to Look for the Pancreas. *arXiv preprint arXiv:1804.03999*.
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI 2015*, 234-241.
- Woo, S., et al. (2018). CBAM: Convolutional Block Attention Module. *ECCV 2018*.
# Repo_DBT
