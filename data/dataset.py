"""
data/dataset.py

MCTEDDataset — PyTorch Dataset for the MCTED patch dataset.

Reads raw patch files from disk on demand. Produces nothing permanent.
Each __getitem__ call loads four raw files, applies validity masking,
normalization, optional auxiliary label generation (slope + roughness),
and optional augmentation, then returns a tensor dict.

Expected directory layout (flat, no subdirectories):
    <root_dir>/
        <scene_pair>_<patch_idx>.optical.png
        <scene_pair>_<patch_idx>.elevation.tif
        <scene_pair>_<patch_idx>.initial_nan_mask.png
        <scene_pair>_<patch_idx>.deviation_mask.png
        ...

Usage:
    from data.dataset import MCTEDDataset
    from torch.utils.data import DataLoader

    # Training (Implementation 2 — single output)
    train_ds = MCTEDDataset("data/train/", augment=True)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)

    # Validation
    val_ds = MCTEDDataset("data/val/", augment=False)

    # Training (Implementation 3 — multi-output, needs slope + roughness)
    train_ds = MCTEDDataset("data/train/", augment=True, compute_auxiliary=True)
"""

import os
import warnings
import numpy as np
from PIL import Image
import tifffile
import torch
from torch.utils.data import Dataset
from scipy.ndimage import sobel, gaussian_filter


class MCTEDDataset(Dataset):
    """
    Live reader for MCTED patch quads.

    Args:
        root_dir (str):
            Path to extracted patch directory — either data/train/ or data/val/.
            The caller decides which split to use; this class is split-agnostic.

        augment (bool):
            If True, applies random horizontal flip, vertical flip, and 90-degree
            rotation. All transforms are applied identically to the optical image,
            DEM patch, and validity mask in lock-step. Set True for training only.

        compute_auxiliary (bool):
            If True, computes slope and roughness from the DEM and includes them
            in the returned dict as "slope" and "roughness" tensors. Required for
            Implementations 3 and 4. Skip for Implementations 1 and 2 to avoid
            unnecessary computation.

        pixel_size_m (float):
            Ground sampling distance in meters per pixel. CTX is 6.0 m/pixel.
            Used when computing slope gradient — affects the physical scale of
            the slope output (degrees) but not elevation or roughness.

        roughness_window (int):
            Side length of the local smoothing window used to compute roughness.
            Default 5 pixels = 30 meters at CTX resolution. Roughness is defined
            as the absolute deviation of each pixel from its local mean elevation.
    """

    def __init__(
        self,
        root_dir: str,
        augment: bool = False,
        compute_auxiliary: bool = False,
        pixel_size_m: float = 6.0,
        roughness_window: int = 5,
    ):
        self.root_dir = root_dir
        self.augment = augment
        self.compute_auxiliary = compute_auxiliary
        self.pixel_size_m = pixel_size_m
        self.roughness_window = roughness_window

        # ------------------------------------------------------------------
        # Build patch index by scanning for .optical.png files.
        # .optical.png is the canonical anchor: every valid patch has one.
        # Sorting guarantees identical index order across runs and OS/FS types.
        # ------------------------------------------------------------------
        all_files = os.listdir(root_dir)
        self.patch_ids = sorted(
            f[: -len(".optical.png")]
            for f in all_files
            if f.endswith(".optical.png")
        )

        if len(self.patch_ids) == 0:
            raise ValueError(
                f"No .optical.png files found in {root_dir}. "
                "Check that the dataset has been extracted correctly."
            )

    def __len__(self) -> int:
        return len(self.patch_ids)

    def __getitem__(self, idx: int) -> dict:
        patch_id = self.patch_ids[idx]
        base = os.path.join(self.root_dir, patch_id)

        # ------------------------------------------------------------------
        # 1. Load raw files
        # ------------------------------------------------------------------

        # Optical: saved as RGB but all channels identical — convert to L
        # immediately to discard the two redundant channels and halve memory.
        optical = np.array(
            Image.open(f"{base}.optical.png").convert("L"),
            dtype=np.float32,
        )  # (518, 518)

        # Elevation: float32 TIF — must use tifffile, not PIL.
        # PIL silently converts float32 to uint8 and destroys the data.
        elevation = tifffile.imread(f"{base}.elevation.tif").astype(np.float32)
        # (518, 518), values in meters above Martian areoid

        # Validity masks: uint8, 0 = valid, 255 = invalid
        nan_mask = np.array(
            Image.open(f"{base}.initial_nan_mask.png").convert("L"),
            dtype=np.uint8,
        )
        dev_mask = np.array(
            Image.open(f"{base}.deviation_mask.png").convert("L"),
            dtype=np.uint8,
        )

        # ------------------------------------------------------------------
        # 2. Combined validity mask
        # A pixel is valid only if it passes BOTH quality gates.
        # Shape: (518, 518), dtype bool, True = valid.
        # Used for masked loss during training and masked metrics at eval.
        # ------------------------------------------------------------------
        valid_mask = (nan_mask == 0) & (dev_mask == 0)

        # ------------------------------------------------------------------
        # 3. Optical normalization
        #    Step 1 — percentile clip: removes hot pixels and cosmic ray
        #             artifacts before statistics are computed. Clipping at
        #             2nd/98th rather than min/max prevents a single bright
        #             speck from compressing the entire dynamic range.
        #    Step 2 — z-score per patch: handles the large variation in
        #             illumination angle and surface albedo across Mars.
        #             Global statistics would be wrong here.
        # ------------------------------------------------------------------
        p_low = np.percentile(optical, 2)
        p_high = np.percentile(optical, 98)
        optical = np.clip(optical, p_low, p_high)

        mu = optical.mean()
        sigma = optical.std()
        if sigma < 1e-6:
            # Fully uniform patch (fill region or sensor artifact).
            # Warn once — these patches carry no texture signal.
            warnings.warn(
                f"Patch {patch_id} has near-zero optical std ({sigma:.2e}). "
                "Normalization will produce a zero tensor.",
                RuntimeWarning,
            )
            sigma = 1.0
        optical = (optical - mu) / sigma

        # ------------------------------------------------------------------
        # 4. DEM normalization
        #    Subtract per-patch mean → relative topography (terrain shape).
        #    Do NOT divide by std — keeps units in meters so MAE and RMSE
        #    remain physically interpretable.
        #    Store the mean so absolute elevation can be reconstructed at
        #    inference: predicted_absolute = predicted_relative + dem_mean.
        # ------------------------------------------------------------------
        dem_mean = float(elevation.mean())
        elevation = elevation - dem_mean  # now in meters, relative to patch mean

        # ------------------------------------------------------------------
        # 5. Auxiliary label generation (slope + roughness)
        #    Computed from the normalized DEM — so outputs are also relative.
        #    Only runs when compute_auxiliary=True (Implementations 3 and 4).
        #    On-the-fly computation avoids storing ~100GB of precomputed files.
        #    np.gradient on 518x518 float32 takes < 1ms — negligible vs I/O.
        # ------------------------------------------------------------------
        slope = None
        roughness = None
        if self.compute_auxiliary:
            # Sobel computes gradients over a 3x3 neighborhood before differencing,
            # making slope estimates robust to pixel-scale stereo reconstruction noise.
            # The /8 factor normalizes the Sobel kernel weight so output is in
            # meters/meter, consistent with pixel_size_m scaling.
            gx = sobel(elevation, axis=1) / (8 * self.pixel_size_m)
            gy = sobel(elevation, axis=0) / (8 * self.pixel_size_m)
            slope = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2))).astype(np.float32)
            # slope: (518, 518), degrees [0, 90)

            # Gaussian filter weights nearby pixels more than distant ones when
            # computing local mean, producing smoother roughness maps with fewer
            # blocky artifacts than a uniform box filter.
            # sigma = roughness_window / 2.0, so default window=5 gives sigma=2.5,
            # an effective neighborhood of ~15 meters at CTX resolution.
            local_mean = gaussian_filter(elevation, sigma=self.roughness_window / 2.0)
            roughness = np.abs(elevation - local_mean).astype(np.float32)
            # roughness: (518, 518), meters (absolute deviation from Gaussian local mean)

        # ------------------------------------------------------------------
        # 6. Augmentation (training only, augment=False for val/inference)
        #    All transforms applied identically to optical, elevation, and
        #    valid_mask using the same random state. Applying a flip to the
        #    image but not the DEM would misalign the supervision signal.
        #    Only geometric transforms — brightness jitter on the DEM would
        #    corrupt the physical elevation values.
        # ------------------------------------------------------------------
        if self.augment:
            optical, elevation, valid_mask, slope, roughness = _augment(
                optical, elevation, valid_mask, slope, roughness
            )

        # ------------------------------------------------------------------
        # 7. Convert to tensors
        #    unsqueeze(0) adds the channel dimension: (H,W) -> (1,H,W).
        #    Conv2d expects (C,H,W); single-channel inputs have C=1.
        #    valid_mask stays (H,W) bool — used for boolean indexing in loss.
        # ------------------------------------------------------------------
        sample = {
            "optical":  torch.from_numpy(optical).float().unsqueeze(0),    # (1,518,518)
            "dem":      torch.from_numpy(elevation).float().unsqueeze(0),  # (1,518,518)
            "valid":    torch.from_numpy(valid_mask.astype(np.uint8)).bool(),  # (518,518)
            "dem_mean": dem_mean,   # float, meters — for absolute reconstruction
            "patch_id": patch_id,  # str — for debugging and error logging
        }

        if self.compute_auxiliary:
            sample["slope"]     = torch.from_numpy(slope).float().unsqueeze(0)     # (1,518,518)
            sample["roughness"] = torch.from_numpy(roughness).float().unsqueeze(0) # (1,518,518)

        return sample


# ------------------------------------------------------------------------------
# Augmentation helper (module-level, not a method — keeps __getitem__ readable)
# ------------------------------------------------------------------------------

def _augment(optical, elevation, valid_mask, slope, roughness):
    """
    Apply random geometric augmentations identically across all arrays.

    Operations:
        - Random horizontal flip (p=0.5)
        - Random vertical flip (p=0.5)
        - Random 90-degree rotation: 0, 90, 180, or 270 degrees (uniform)

    All arrays are numpy, all 2D (H, W). Returns same types and shapes.
    slope and roughness may be None if compute_auxiliary=False.
    """
    # One shared RNG call sequence — same decisions applied to all arrays
    if np.random.rand() > 0.5:
        optical    = np.fliplr(optical)
        elevation  = np.fliplr(elevation)
        valid_mask = np.fliplr(valid_mask)
        if slope is not None:
            slope     = np.fliplr(slope)
            roughness = np.fliplr(roughness)

    if np.random.rand() > 0.5:
        optical    = np.flipud(optical)
        elevation  = np.flipud(elevation)
        valid_mask = np.flipud(valid_mask)
        if slope is not None:
            slope     = np.flipud(slope)
            roughness = np.flipud(roughness)

    k = np.random.randint(0, 4)  # 0=0°, 1=90°, 2=180°, 3=270°
    if k > 0:
        optical    = np.rot90(optical,    k)
        elevation  = np.rot90(elevation,  k)
        valid_mask = np.rot90(valid_mask, k)
        if slope is not None:
            slope     = np.rot90(slope,     k)
            roughness = np.rot90(roughness, k)

    # np.flip and np.rot90 return views with negative strides.
    # np.ascontiguousarray forces a copy with normal memory layout,
    # which torch.from_numpy requires.
    optical    = np.ascontiguousarray(optical)
    elevation  = np.ascontiguousarray(elevation)
    valid_mask = np.ascontiguousarray(valid_mask)
    if slope is not None:
        slope     = np.ascontiguousarray(slope)
        roughness = np.ascontiguousarray(roughness)

    return optical, elevation, valid_mask, slope, roughness