# MarsDEMNet

A comparative study of classical and deep learning approaches for single-image DEM prediction from Mars CTX orbital imagery.

### Overview

The Context Camera (CTX) aboard the Mars Reconnaissance Orbiter has imaged **~99.5%** of the Martian surface at 5–6 m/pixel, but high-resolution stereo Digital Elevation Models (DEMs), the geometric data needed to characterize terrain slope and roughness at landing-relevant scales, exist for only **~0.5–1%** of the surface. MarsDEMNet attacks this coverage asymmetry by training models to predict dense elevation maps from a **single CTX optical image**, with no stereo pair required at inference time.

The project evaluates four progressively complex approaches on a shared, geography-aware split of the [MCTED dataset](https://huggingface.co/datasets/ESA-Datalabs/MCTED) (80,898 paired 518×518 CTX/DEM patches), under identical preprocessing and validity masking for a fair comparison.

![Qualitative DEM predictions: CTX input, predicted DEM, and ground-truth DEM](outputs/samples.png)
*Qualitative results on held-out validation patches (5×7 panel from `scripts/visualize_qualitative.py`): CTX input, predicted DEM, ground truth.*

### Results

All metrics are computed on the held-out validation split over valid DEM pixels only. Elevation errors are in meters of per-patch relative topography.

| # | Implementation | Val RMSE (m) | Val MAE (m) | δ₁ | Notes |
|---|----------------|:------------:|:-----------:|:----:|-------|
| 1 | Classical RF baseline | — | ~41.3 (elev. std)* | — | 63-D hand-crafted features; near-random slope direction (~43.6°); ~2× train/val gap |
| 2 | Single-output U-Net | 74.38 | 52.86 | 0.418 | Dense DEM; still declining at epoch 50 |
| 3 | Multi-output U-Net (4-block) | **74.29** | — | — | Elev + slope + roughness; uniform (1:1:1) loss beats elev-emphasized (2:1:1) at 74.50 |
| 4 | Multi-output U-Net + depth ablation | **59.88** | — | — | Implementation 3 deepened to 5 blocks (31.4M params); see capacity curve below |

\* The Random Forest predicts patch-level statistics (mean relative elevation, elevation std, dominant slope direction) rather than a dense map, so its numbers are not directly comparable to the dense models.

**Encoder depth ablation (Implementation 4):**

| Encoder depth | Val RMSE (m) |
|---------------|:------------:|
| 3-block | 82.80 |
| 4-block | 74.29 |
| 5-block | **59.88** |

Implementation 4 is the **same multi-output U-Net as Implementation 3** with encoder depth varied — the 4-block row here is exactly Implementation 3. Capacity scales monotonically with no overfitting at 5 blocks (31.4M params), a **19% RMSE improvement** over the 4-block baseline that slightly edges past the MCTED authors' published U-Net baseline of 61.90 m.

#### Key findings

- **Classical ceiling.** Patch-level hand-crafted features (LBP, Gabor, HOG, intensity moments) cannot capture the spatial structure needed for dense DEM prediction, slope-direction prediction is near-random and the large train/val gap indicates memorization.
- **Multi-task regularization.** Uniform loss weighting outperforms elevation-emphasized weighting, suggesting the auxiliary slope/roughness tasks act as a regularizer rather than a distraction when weighted evenly.
- **Depth helps, no overfitting.** A deeper 5-block encoder gives the best result with no train/val divergence, validating the capacity-scaling hypothesis for this dataset size.

### Repository structure
```
MarsDEMNet/
├── data/
│   ├── dataset.py          # MCTED Dataset: masking, normalization, geo-split, 3 backends
│   └── download_data.sh    # pulls MCTED from HuggingFace
├── scripts/
│   ├── classical/
│   │   ├── features.py      # 63-D hand-crafted features (LBP, Gabor, HOG, moments)
│   │   └── train.py         # Implementation 1 — Random Forest
│   ├── deeplearning/
│   │   ├── unet.py          # U-Net architecture (shared)
│   │   ├── trainer.py       # training loop (shared)
│   │   ├── impl2.py         # Implementation 2 — single-output U-Net
│   │   ├── impl3.py         # Implementation 3 — multi-output U-Net (4-block)
│   │   └── impl4.py         # Implementation 4 — depth ablation (5-block, best)
│   └── samples.py           # qualitative 5×7 panel -> outputs/samples.png
├── notebooks/plots.ipynb    # results & figures
├── outputs/                 # results.json + training/ablation curves per implementation
└── requirements.txt
```

### Dataset

MarsDEMNet uses **MCTED** (Mars CTX Terrain-Elevation Dataset), hosted at [`ESA-Datalabs/MCTED`](https://huggingface.co/datasets/ESA-Datalabs/MCTED). Each sample is four 518×518 files:

- `*.optical.png` — CTX orthoimage (uint8, monochrome stored as RGB)
- `*.elevation.tif` — DEM patch (float32, meters above the areoid)
- `*.initial_nan_mask.png` — originally-missing stereo pixels
- `*.deviation_mask.png` — elevation outliers removed during artifact processing

The official geography-aware split (`.artifacts/clustering/dataset_samples_split.yaml`) partitions at the source-scene level, so patches from a single stereo scene never straddle the train/val boundary, eliminating spatial-autocorrelation leakage.

**Normalization** (must match the pipeline exactly):
- Optical: per-patch 2nd–98th percentile clip, then per-patch z-score.
- DEM: per-patch **mean subtraction only** (no std division), preserving meter scale so RMSE/MAE stay physically interpretable.
- Classical pipeline: raw uint8 — **do not** apply the above normalization.

### Installation

```bash
git clone https://github.com/harshithkethavath/MarsDEMNet.git
cd MarsDEMNet
pip install -r requirements.txt
```

Set a HuggingFace token to download the dataset:

```bash
export HF_TOKEN=your_token_here
bash data/download_data.sh
```

### Usage

#### Dataset backends

`dataset.py` auto-detects three backends. The RAM-cache backend is strongly recommended on clustered/Lustre filesystems:

```python
from data.dataset import MCTEDDataset

# Directory of files
ds = MCTEDDataset(root="data/mcted", split="train")

# Packed HDF5
ds = MCTEDDataset(root="data/mcted.h5", split="train")

# RAM cache — loads all arrays into memory at init (≈122 GB train, 30 GB val)
ds = MCTEDDataset(root="data/mcted", split="train", cache=True)
```

#### Training

Each model has its own entry script; `unet.py` and `trainer.py` are shared modules.

```bash
python scripts/deeplearning/impl2.py   # single-output U-Net
python scripts/deeplearning/impl3.py   # multi-output U-Net (4-block)
python scripts/deeplearning/impl4.py   # multi-output U-Net + depth ablation (5-block, best)
python scripts/classical/train.py      # Random Forest baseline
```

Config/flags live at the top of each script.

#### Qualitative panel

```bash
python scripts/samples.py   # writes outputs/samples.png
```

### HPC notes

The project was developed on UGA's Sapelo2 cluster (A100/H100, 256 GB RAM nodes, SLURM).

The single most impactful infrastructure decision was **RAM-cache preloading**. The Sapelo2 `/scratch` Lustre filesystem could not sustain the ~260,000 small-file opens per epoch; loading all arrays into RAM at dataset init cut epoch time from **~3,800 s to ~570 s** (≈6.6×). Use `cache=True` and a 256 GB node for any full-dataset run.

Other choices worth noting:
- **Masked MAE loss** (not MSE) — handles invalid pixels and avoids gradient spikes from hundred-meter elevation discontinuities at crater walls and scarps.
- **Bilinear upsampling** (not transposed convolution) — avoids checkerboard artifacts on the non-power-of-two 518×518 inputs.
- Joint augmentation (flips, 90° rotations) applied identically to image, DEM, and mask to keep supervision aligned.

### Pretrained models

Checkpoints for all four implementations are published on HuggingFace: [`harshithkethavath/MarsDEMNet`](https://huggingface.co/harshithkethavath/MarsDEMNet)

### Citation

If you use MarsDEMNet, please cite:

```bibtex
@misc{marsdemnet2026,
  title     = {MarsDEMNet: Classical and Deep Learning Approaches for Single-Image Digital Elevation Model Prediction from Mars CTX Imagery},
  author    = {Harshith Kethavath},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/harshithkethavath/MarsDEMNet}
}
```
