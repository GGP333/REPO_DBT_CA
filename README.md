# 3D Segmentation of DBT Volumes (Digital Breast Tomosynthesis)

## 1. Project Overview

This project implements and compares three 3D convolutional neural network architectures for **automatic breast tumor segmentation** in Digital Breast Tomosynthesis (DBT) volumes. The goal is to evaluate the performance of different U-Net variants on the task of voxel-wise binary segmentation, using multiple data configurations that include tumors of different sizes and real vs. synthetic data.

### Main contributions

- Systematic comparison of 3 architectures (auto-configured nnUNet 3D, Attention U-Net 3D, U-Net BCE) across 4 data configurations.
- Implementation of an auto-configuration pipeline inspired by nnU-Net (Isensee et al.): dataset fingerprint, automatic architecture planning with anisotropic pooling, patch-based training with foreground oversampling, sliding window inference with Gaussian weighting, and connected-components post-processing.
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
| **Dataset_Both** | Mixture of large and small tumors (90 large + 30 small) | 120 | 120 | 96 | 24 | 11 |
| **Dataset_Both_RealWorld** | `dbt_*` + `real_dbt_*` (real clinical data) | 140 | 140 | ~104 | ~26 | 10 (real_dbt_*) |

**Note on Train/Val/Test splits**: the values in the table correspond to the split used by UNet_BCE and Attention U-Net (80/20 train/val). nnUNet uses a slightly different split (e.g., 65/16/8 on large_tumor, 87/22/11 on Both). In both cases, the test set is the same (8, 3, 11 studies respectively) and is evaluated with the best checkpoint.

**Note on Dataset_Both_RealWorld**: the directory contains 140 studies (120 `dbt_*` + 20 `real_dbt_*`). The test set is built with **50% of the `real_dbt_*`** (10 real clinical studies). The remaining 50% of `real_dbt_*` is mixed with the `dbt_*` to form train/val. This allows the model to learn from real and synthetic data, and to be evaluated exclusively on unseen real clinical data.

### 2.3 Train/val/test split

- **Fixed seed**: 42 (full reproducibility).
- **Study-level split**: partitioning is done at the study level (not by slice), preventing data leakage.
- **General case (without `real_dbt_*`)**: uses the configured train/val/test proportions over the total dataset.
- **Special case with real data (`real_dbt_*`)**: 50% of the `real_dbt_*` studies are assigned exclusively to the **test** set. The remaining 50% is mixed with the `dbt_*` to form train/val. This way the model trains with real and synthetic data and is evaluated only on unseen real clinical data.
- This logic is implemented in `src/models/nnunet/train_unet_dbt.py`, `src/models/attention_unet/train_unet_dbt.py`, and `src/models/unet_bce/src/dataset_dbt.py`.

---

## 3. Model Architectures

Three 3D U-Net variants are compared for volumetric segmentation. Conceptually, all three share the encoder-decoder pattern with skip connections, but differ in how they model context, how they fuse multi-scale information, how they adapt to the data geometry, and which optimization objective they prioritize.

### 3.1 Auto-configured 3D nnUNet (`src/models/nnunet`)

**Parameters**: variable (auto-computed by the planner)

Implementation that follows the core philosophy of nnU-Net by Isensee et al. (1809.10486): the architecture is **not fixed**, but **auto-configures** itself based on an automatic analysis of the dataset properties. The auto-configuration pipeline consists of three stages:

#### 3.1.1 Dataset Fingerprint (`src/fingerprint.py`)

Before training, all volumes in the dataset are scanned and statistics are extracted:
- **Median, minimum, and maximum shape** `(Z, H, W)` across all studies.
- **Median foreground/background ratio** (percentage of voxels that are tumor).
- **Intensity statistics** (mean, std, global percentiles).
- **Spacing** (parameterizable; default `(1, 1, 1)` for already-resampled data).

The fingerprint is saved as `fingerprint.json` in the output directory.

#### 3.1.2 Architecture Planner (`src/planner.py`)

Receives the fingerprint and the available GPU VRAM, and automatically computes:

- **Patch size**: based on the dataset's median shape, adjusted to fit in GPU. Each dimension is rounded to a multiple of `2^(n_pools_in_that_axis)`.
- **Anisotropic strides per level**: for each axis, `max_pools = floor(log2(dim / 4))` is computed. If Z is much smaller than H/W (anisotropic data, common in DBT), Z receives fewer poolings. Example: `[(1,2,2), (2,2,2), (2,2,2), (1,2,2)]` instead of the fixed isotropic stride `(2,2,2)`.
- **Channels per level**: starts at `base_ch=32`, doubles per level, capped at 320.
- **Batch size**: estimated by a VRAM heuristic (`patch_voxels * max_channels * overhead`), maximizing what fits in GPU.

The plan is saved as `architecture_plan.json`.

#### 3.1.3 Resulting architecture

The model is built with `nnUNet3D.from_plan(plan)` and preserves the architectural principles of nnU-Net:

- **Residual blocks (`nnUNetResidualBlock`)**: 3D convolutions with residual shortcut, stabilizing deep training.
- **Instance normalization** (`InstanceNorm3d`): useful with small batch size.
- **LeakyReLU activation**: reduces the risk of dead neurons.
- **Encoder with anisotropic strides**: pooling adapts to the data geometry. Short axes (Z in DBT) receive less downsampling to avoid losing inter-slice information.
- **Decoder with anisotropic `ConvTranspose3d`**: upsample kernels match the encoder strides.
- **Gradient checkpointing**: trades compute for memory.
- **Binary head**: `Conv3d 1x1 + sigmoid`.

#### 3.1.4 Patch-based training (`src/patch_utils.py`)

Instead of processing full volumes with padding, training uses **random patches** of fixed size (determined by the planner):

- With probability `oversample_foreground` (0.33 by default), the patch center is forced onto a foreground voxel, ensuring the network sees enough tumor despite extreme class imbalance (fg < 1%).
- Foreground coordinates are pre-computed per volume for fast extraction.
- Collation is trivial (all patches have the same size), eliminating the need for dynamic padding.

#### 3.1.5 Sliding Window Inference

At validation and test time, the full volume is traversed with a **sliding window** with configurable overlap (50% by default):

- Each patch is predicted independently.
- Predictions are accumulated using a **Gaussian importance map** (central voxels weigh more than edges) to avoid seam artifacts.
- The result is a full-resolution probability map.

#### 3.1.6 Post-processing (`src/postprocess.py`)

After prediction, **3D connected-components** filtering is applied:

- Several minimum-size thresholds (`min_size`) are tested on the validation set.
- The `min_size` that maximizes validation Dice is automatically selected.
- The same threshold is applied to test predictions.

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

| Feature | nnUNet 3D Auto | Attention U-Net | U-Net BCE |
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

All experiments share the same base configuration, defined in `configs/config.yaml`:

| Parameter | Value |
|-----------|-------|
| **Seed** | 42 |
| **Epochs** | 500 |
| **Optimizer** | AdamW (lr=0.001, weight_decay=1e-5) |
| **Scheduler** | Polynomial LR Decay: `lr * (1 - epoch/max_epochs)^0.9` |
| **Warmup** | 5 epochs (linear warmup) |
| **Batch size** | 1 (Att. U-Net, U-Net BCE) / auto-computed (nnUNet) |
| **AMP** | Enabled (Mixed Precision with GradScaler) |
| **Gradient clipping** | max_norm=1.0 |
| **Binarization threshold** | 0.5 |
| **Foreground oversampling** | 0.5 (Att. U-Net, U-Net BCE) / 0.33 patch-based (nnUNet) |
| **Augmentation** | Enabled |
| **Early stopping** | Disabled (full 500 epochs are trained) |

**Auto-configured nnUNet specifics:**

| Parameter | Value |
|-----------|-------|
| **auto_plan** | `true` (enables fingerprint + planner) |
| **gpu_vram_gb** | 16.0 (RTX 4070 Ti SUPER) |
| **Training** | Patch-based (auto-computed size) |
| **Validation/Test** | Sliding window with overlap=0.5 and Gaussian weighting |
| **Foreground oversampling** | 0.33 (prob. of forcing foreground-centered patch) |
| **Post-processing** | Connected components (min_size auto-optimized on val) |

### 4.1 Loss functions

- **Dice + CE** (nnUNet, Attention U-Net): `0.5 * DiceLoss + 0.5 * BCE`. Combines direct optimization of the Dice coefficient with BCE stability to handle extreme class imbalance.
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

| Model | Dataset | Time |
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

### 6.1 Validation metrics (best checkpoint)

| Model | Dataset | Dice | IoU | Precision | Recall | HD95 | AP | mAP |
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

### 6.2 Test metrics (best checkpoint)

| Model | Dataset | Dice | IoU | Precision | Recall | HD95 | AP | mAP |
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

### 6.3 Best epoch per experiment

| Model | Dataset | Best epoch (of 500) | Val Dice at best epoch |
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
|-- src/                              # Source code
|   |-- models/                       # Implementations of the 3 models
|   |   |-- nnunet/                   # Auto-configured nnU-Net 3D
|   |   |   |-- src/                  # Model modules
|   |   |   |   |-- nnunet.py         # nnUNet3D architecture (anisotropic strides)
|   |   |   |   |-- fingerprint.py    # Dataset statistical analysis
|   |   |   |   |-- planner.py        # Architecture auto-planning
|   |   |   |   |-- patch_utils.py    # Patches + sliding window inference
|   |   |   |   |-- postprocess.py    # Connected-components filtering
|   |   |   |   |-- train_eval.py     # Training/evaluation
|   |   |   |   |-- metrics.py        # 12 segmentation metrics
|   |   |   |   |-- dataset_dbt.py    # Dataset (volumes + patches)
|   |   |   |   |-- io_utils.py       # Study discovery
|   |   |   |   |-- config_loader.py  # YAML config loading
|   |   |   |-- configs/config.yaml   # Hyperparameters
|   |   |   |-- train_unet_dbt.py     # Main script
|   |   |
|   |   |-- attention_unet/           # Attention U-Net 3D
|   |   |   |-- src/
|   |   |   |   |-- attention_unet3d.py
|   |   |   |   |-- ...
|   |   |   |-- train_unet_dbt.py
|   |   |
|   |   |-- unet_bce/                 # U-Net with BCE Loss
|   |       |-- src/
|   |       |   |-- unet3d.py
|   |       |   |-- ...
|   |       |-- train_unet_bce.py
|   |
|   |-- shared/                       # Modules shared between models
|   |   |-- visualization.py          # Sample PNG generation
|   |   |-- traceability.py           # run_config.json and run_summary.json
|   |   |-- augmentations.py          # Data augmentations
|   |
|   |-- preprocessing/                # Preprocessing pipeline
|   |   |-- preprocess_simple.py      # Main preprocessing
|   |   |-- preprocess_dbt.py         # Alternative preprocessing
|   |   |-- run_preprocess_all.py     # Batch execution
|   |   |-- run_preprocess_simple.py
|   |   |-- visualize_preprocessed_dataset.py
|   |   |-- visualize_volumes.py
|   |   |-- build_global_hist_ref.py
|   |   |-- prune_slices_preprocessed_dataset.py
|   |   |-- summarize_dataset_dimensions.py
|   |
|   |-- evaluation/                   # Evaluation scripts (reserved)
|   |
|   |-- run_test_trainings.py         # Orchestrator for the 12 experiments
|   |-- train_all_models.py           # Master training script
|   |-- fix_outputs.py                # Result post-processing
|   |-- generate_samples_and_summary.py  # Sample and summary generation
|
|-- results/                          # Results (excluded from git)
|   |-- checkpoints/                  # Trained models (.pt)
|   |-- logs/                         # Metrics and plots
|   |-- predictions/                  # Predictions (.npy)
|   |-- README.md                     # Results documentation
|
|-- docs/                             # Additional documentation
|
|-- requirements.txt                  # Python dependencies
|-- .gitignore                        # Files excluded from the repo
|-- README.md                         # This file
```

### 7.1 Workflow

```
1. Obtain data         -> data/raw/
2. Preprocess          -> python src/preprocessing/preprocess_simple.py
3. Train models        -> python src/run_test_trainings.py
4. Evaluate results    -> results/logs/
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

To run all 12 experiments (3 models x 4 datasets):

```bash
python src/run_test_trainings.py
```

To train an individual model:

```bash
# nnUNet
python src/models/nnunet/train_unet_dbt.py \
    --config src/models/nnunet/configs/config.yaml

# Attention U-Net
python src/models/attention_unet/train_unet_dbt.py \
    --config src/models/attention_unet/configs/config.yaml

# U-Net BCE
python src/models/unet_bce/train_unet_bce.py \
    --config src/models/unet_bce/configs/config.yaml
```

### 8.5 Evaluate and generate visualizations

```bash
python src/generate_samples_and_summary.py
python src/fix_outputs.py
```

---

## 9. References

- Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*, 18(2), 203-211.
- Isensee, F., Petersen, J., Klein, A., Zimmerer, D., et al. (2018). nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation. *arXiv preprint arXiv:1809.10486*.
- Oktay, O., et al. (2018). Attention U-Net: Learning Where to Look for the Pancreas. *arXiv preprint arXiv:1804.03999*.
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI 2015*, 234-241.
- Woo, S., et al. (2018). CBAM: Convolutional Block Attention Module. *ECCV 2018*.
