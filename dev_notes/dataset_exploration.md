# Dataset Exploration — Developer Notes

This document explains how to get the repo set up, download the MCTED dataset, 
and explore it. Follow the steps in order.

---

## Repo Structure

```
MarsDEMNet/
├── data/
│   ├── download_data.sh       # script to download MCTED from HuggingFace
│   └── sample/                # downloaded patches land here (not committed to git)
├── dev_notes/
│   └── dataset_exploration.md # you are here
├── notebooks/
│   └── dataset_exploration.ipynb  # dataset exploration notebook
├── requirements.txt           # all project dependencies
└── venv/                      # your local virtual environment (not committed to git)
```

---

## Setup

**1. Create and activate a virtual environment using Python 3.10:**
```bash
python3.10 -m venv venv
source venv/bin/activate
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## Downloading the Dataset

The MCTED dataset is hosted on HuggingFace at `ESA-Datalabs/MCTED`. 
You need a HuggingFace account and a read token from 
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

**Set your token as an environment variable — never hardcode it:**
```bash
export HF_TOKEN=your_token_here
```

The download script lives in `data/download_data.sh` and has two modes:

**Sample mode — downloads first N patches only (recommended to start):**
```bash
cd data
bash download_data.sh --sample 10
```
Patches are extracted into `data/sample/`. Each patch consists of 4 files sharing 
the same base name:

| File | Description |
|------|-------------|
| `<base>.optical.png` | CTX grayscale image (uint8, saved as RGB) — model input |
| `<base>.elevation.tif` | Ground truth DEM (float32, meters) — prediction target |
| `<base>.initial_nan_mask.png` | Marks pixels where stereo matching failed (0=valid, 255=invalid) |
| `<base>.deviation_mask.png` | Marks elevation outlier pixels (0=valid, 255=invalid) |

**Full mode — downloads the entire dataset (87 train + 22 val tar files):**
```bash
cd data
bash download_data.sh --full
```
This will take a while depending on your connection. Full dataset is several hundred GB.

---

## Exploring the Dataset

Once you have downloaded a sample, open the exploration notebook:

```bash
cd ..   # back to repo root
jupyter notebook notebooks/dataset_exploration.ipynb
```

The notebook walks through the following in order:

1. **Schema inspection** — shapes, dtypes, what each of the 4 files contains
2. **Single patch visualization** — all 4 channels side by side
3. **Validity mask** — how the two masks combine, what percentage of pixels are usable
4. **Multi-patch grid** — visual variety across patches
5. **Profile slice** — optical intensity vs elevation along a single row, showing why brightness alone cannot predict elevation
6. **Scatter plot** — per-pixel intensity vs elevation across all valid pixels, Pearson correlation = -0.314
7. **Slope and roughness** — how they are derived analytically from the elevation TIF
8. **Normalization** — before vs after for both optical and DEM, why each transformation is necessary
9. **TIF explanation** — what float32 pixel values mean physically and why TIF is used over PNG for elevation data

---

## Notes

- `data/sample/` is in `.gitignore` — do not commit downloaded patches
- Always activate the venv before running anything: `source venv/bin/activate`
- If the download fails, make sure `HF_TOKEN` is set and has read access