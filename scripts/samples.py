import os
import sys
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
import tifffile
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from data.dataset import MCTEDDataset
from scripts.deeplearning.unet import UNet

# ── Control panel ─────────────────────────────────────────
SEEDS      = [42, 7, 123, 69, 143]   # one seed per row — change these
VAL_DIR    = "data/val"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINTS = {
    "Impl2":          "outputs/deeplearning/impl2/best_model.pt",
    "Impl3 Uniform":  "outputs/deeplearning/impl3/best_model_uniform.pt",
    "Impl3 Elev-Emph":"outputs/deeplearning/impl3/best_model_elev_emph.pt",
    "Impl4 3-block":  "outputs/deeplearning/impl4/best_model_3block.pt",
    "Impl4 5-block":  "outputs/deeplearning/impl4/best_model_5block.pt",
}

MODEL_CONFIGS = {
    "Impl2":           {"num_blocks": 4, "out_channels": 1},
    "Impl3 Uniform":   {"num_blocks": 4, "out_channels": 3},
    "Impl3 Elev-Emph": {"num_blocks": 4, "out_channels": 3},
    "Impl4 3-block":   {"num_blocks": 3, "out_channels": 3},
    "Impl4 5-block":   {"num_blocks": 5, "out_channels": 3},
}

# ── Helpers ───────────────────────────────────────────────
def load_model(name):
    cfg   = MODEL_CONFIGS[name]
    model = UNet(in_channels=1,
                 out_channels=cfg["out_channels"],
                 num_blocks=cfg["num_blocks"],
                 base_ch=32).to(DEVICE)
    ckpt  = torch.load(CHECKPOINTS[name], map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def pick_patch(seed):
    """Pick a random patch_id from val dir using the given seed."""
    all_ids = sorted(
        f[:-len(".optical.png")]
        for f in os.listdir(VAL_DIR)
        if f.endswith(".optical.png")
    )
    rng = random.Random(seed)
    return rng.choice(all_ids)


def load_patch(patch_id):
    base    = os.path.join(VAL_DIR, patch_id)
    optical = np.array(Image.open(f"{base}.optical.png").convert("L"), dtype=np.float32)
    elev    = tifffile.imread(f"{base}.elevation.tif").astype(np.float32)
    nan_mask = np.array(Image.open(f"{base}.initial_nan_mask.png").convert("L"), dtype=np.uint8)
    dev_mask = np.array(Image.open(f"{base}.deviation_mask.png").convert("L"),  dtype=np.uint8)
    valid   = (nan_mask == 0) & (dev_mask == 0)

    # normalize optical
    p_low, p_high = np.percentile(optical, 2), np.percentile(optical, 98)
    optical_norm  = np.clip(optical, p_low, p_high)
    mu, sigma     = optical_norm.mean(), optical_norm.std()
    sigma         = sigma if sigma > 1e-6 else 1.0
    optical_norm  = (optical_norm - mu) / sigma

    # normalize dem
    dem_mean = float(elev.mean())
    elev_rel = elev - dem_mean

    return optical, optical_norm, elev_rel, valid, dem_mean


def run_inference(model, optical_norm, out_channels):
    tensor = torch.from_numpy(optical_norm).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(tensor)
    pred_np = pred[0, 0].cpu().numpy()   # always take elevation channel (index 0)
    return pred_np


def normalize_dem(dem):
    """Normalize DEM to [0,1] for heatmap display."""
    d_min, d_max = dem.min(), dem.max()
    if d_max - d_min < 1e-6:
        return np.zeros_like(dem)
    return (dem - d_min) / (d_max - d_min)


# ── Main ──────────────────────────────────────────────────
def main():
    print("Loading models...")
    models = {name: load_model(name) for name in CHECKPOINTS}

    col_labels = ["CTX Image", "Ground Truth"] + list(CHECKPOINTS.keys())
    n_rows = len(SEEDS)
    n_cols = len(col_labels)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.5 * n_cols, 3.5 * n_rows))

    for row_idx, seed in enumerate(SEEDS):
        patch_id = pick_patch(seed)
        print(f"Row {row_idx+1} | seed={seed} | patch={patch_id}")

        optical_raw, optical_norm, elev_rel, valid, dem_mean = load_patch(patch_id)

        # build list of images to plot
        gt_hs  = normalize_dem(elev_rel)
        images  = [optical_raw, gt_hs]
        cmaps   = ['gray', 'terrain']

        for name, model in models.items():
            pred = run_inference(model, optical_norm,
                                 MODEL_CONFIGS[name]["out_channels"])
            images.append(normalize_dem(pred))
            cmaps.append('terrain')

        for col_idx, (img, cmap) in enumerate(zip(images, cmaps)):
            ax = axes[row_idx][col_idx]
            ax.imshow(img, cmap=cmap, interpolation='nearest')
            ax.axis('off')
            if row_idx == 0:
                ax.set_title(col_labels[col_idx], fontsize=22, fontweight="bold")
        

    plt.tight_layout()
    out_path = "outputs/samples.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()