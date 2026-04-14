"""
data/dataset.py

MCTEDDataset — PyTorch Dataset for the MCTED patch dataset.
Supports three backends:

    files  — reads raw patch files from disk on demand (slow on Lustre)
    hdf5   — reads from a pre-built .h5 file
    ram    — loads entire dataset into RAM at init, zero disk I/O per epoch

Backend is auto-detected:
    root_dir ends with .h5  → hdf5
    cache=True              → ram (loads from directory into RAM at init)
    otherwise               → files

RAM backend is strongly recommended on cluster nodes with >256GB RAM.
Pays a one-time load cost (~10-30 min from Lustre) then runs at memory
speed for all subsequent epochs with zero filesystem overhead.
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
    Args:
        root_dir (str):
            Path to an HDF5 file (data/train.h5) or a directory (data/train/).

        augment (bool):
            Random horizontal flip, vertical flip, 90-degree rotation.
            Applied identically to optical, DEM, and validity mask.

        compute_auxiliary (bool):
            If True, computes slope and roughness from the DEM.
            Required for Implementations 3 and 4.

        cache (bool):
            If True and root_dir is a directory, loads all raw arrays into RAM
            at __init__ time. After loading, __getitem__ never touches disk.
            Recommended on nodes with >= 256GB RAM.

        pixel_size_m (float): CTX ground sampling distance in m/pixel.
        roughness_window (int): Local smoothing window for roughness (pixels).
    """

    def __init__(
        self,
        root_dir: str,
        augment: bool = False,
        compute_auxiliary: bool = False,
        cache: bool = False,
        pixel_size_m: float = 6.0,
        roughness_window: int = 5,
    ):
        self.root_dir          = root_dir
        self.augment           = augment
        self.compute_auxiliary = compute_auxiliary
        self.pixel_size_m      = pixel_size_m
        self.roughness_window  = roughness_window

        if root_dir.endswith(".h5"):
            self._backend = "hdf5"
            self._init_hdf5(root_dir)
        elif cache:
            self._backend = "ram"
            self._init_files(root_dir)
            self._load_into_ram()
        else:
            self._backend = "files"
            self._init_files(root_dir)

    # ------------------------------------------------------------------
    # Backend initialisation
    # ------------------------------------------------------------------

    def _init_hdf5(self, path):
        import h5py
        with h5py.File(path, "r") as f:
            self.patch_ids = [
                pid.decode() if isinstance(pid, bytes) else pid
                for pid in f["patch_ids"][:]
            ]
        self._h5_path = path
        self._h5_file = None  # opened lazily per worker

    def _init_files(self, root_dir):
        all_files      = os.listdir(root_dir)
        self.patch_ids = sorted(
            f[:-len(".optical.png")]
            for f in all_files if f.endswith(".optical.png")
        )
        if not self.patch_ids:
            raise ValueError(f"No .optical.png files found in {root_dir}.")

    def _load_into_ram(self):
        """
        One-time load of all raw arrays into RAM numpy arrays.
        Progress printed to stdout so SLURM logs show it.
        """
        N = len(self.patch_ids)
        print(f"[MCTEDDataset] Loading {N:,} patches into RAM from {self.root_dir} ...")

        self._cache_optical   = np.empty((N, 518, 518), dtype=np.uint8)
        self._cache_elevation = np.empty((N, 518, 518), dtype=np.float32)
        self._cache_nan_mask  = np.empty((N, 518, 518), dtype=np.uint8)
        self._cache_dev_mask  = np.empty((N, 518, 518), dtype=np.uint8)

        for i, patch_id in enumerate(self.patch_ids):
            base = os.path.join(self.root_dir, patch_id)
            self._cache_optical[i]   = np.array(Image.open(f"{base}.optical.png").convert("L"), dtype=np.uint8)
            self._cache_elevation[i] = tifffile.imread(f"{base}.elevation.tif").astype(np.float32)
            self._cache_nan_mask[i]  = np.array(Image.open(f"{base}.initial_nan_mask.png").convert("L"), dtype=np.uint8)
            self._cache_dev_mask[i]  = np.array(Image.open(f"{base}.deviation_mask.png").convert("L"), dtype=np.uint8)

            if (i + 1) % 5000 == 0:
                print(f"[MCTEDDataset]   {i+1:,}/{N:,} patches loaded")

        mem_gb = (self._cache_optical.nbytes + self._cache_elevation.nbytes +
                  self._cache_nan_mask.nbytes + self._cache_dev_mask.nbytes) / 1e9
        print(f"[MCTEDDataset] Cache ready. RAM used: {mem_gb:.1f} GB")

    # ------------------------------------------------------------------
    # Raw loading — one method per backend
    # ------------------------------------------------------------------

    def _load_hdf5(self, idx):
        if self._h5_file is None:
            import h5py
            self._h5_file = h5py.File(self._h5_path, "r")
        return (
            self._h5_file["optical"][idx].astype(np.float32),
            self._h5_file["elevation"][idx].astype(np.float32),
            self._h5_file["nan_mask"][idx],
            self._h5_file["dev_mask"][idx],
        )

    def _load_ram(self, idx):
        return (
            self._cache_optical[idx].astype(np.float32),
            self._cache_elevation[idx].copy(),
            self._cache_nan_mask[idx],
            self._cache_dev_mask[idx],
        )

    def _load_files(self, idx):
        patch_id = self.patch_ids[idx]
        base     = os.path.join(self.root_dir, patch_id)
        return (
            np.array(Image.open(f"{base}.optical.png").convert("L"), dtype=np.float32),
            tifffile.imread(f"{base}.elevation.tif").astype(np.float32),
            np.array(Image.open(f"{base}.initial_nan_mask.png").convert("L"), dtype=np.uint8),
            np.array(Image.open(f"{base}.deviation_mask.png").convert("L"),   dtype=np.uint8),
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, idx):
        patch_id = self.patch_ids[idx]

        if self._backend == "hdf5":
            optical, elevation, nan_mask, dev_mask = self._load_hdf5(idx)
        elif self._backend == "ram":
            optical, elevation, nan_mask, dev_mask = self._load_ram(idx)
        else:
            optical, elevation, nan_mask, dev_mask = self._load_files(idx)

        # Validity mask
        valid_mask = (nan_mask == 0) & (dev_mask == 0)

        # Optical normalisation
        p_low, p_high = np.percentile(optical, 2), np.percentile(optical, 98)
        optical = np.clip(optical, p_low, p_high)
        mu, sigma = optical.mean(), optical.std()
        if sigma < 1e-6:
            warnings.warn(f"Patch {patch_id} near-zero optical std.", RuntimeWarning)
            sigma = 1.0
        optical = (optical - mu) / sigma

        # DEM normalisation
        dem_mean  = float(elevation.mean())
        elevation = elevation - dem_mean

        # Auxiliary labels
        slope, roughness = None, None
        if self.compute_auxiliary:
            gx = sobel(elevation, axis=1) / (8 * self.pixel_size_m)
            gy = sobel(elevation, axis=0) / (8 * self.pixel_size_m)
            slope      = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2))).astype(np.float32)
            local_mean = gaussian_filter(elevation, sigma=self.roughness_window / 2.0)
            roughness  = np.abs(elevation - local_mean).astype(np.float32)

        # Augmentation
        if self.augment:
            optical, elevation, valid_mask, slope, roughness = _augment(
                optical, elevation, valid_mask, slope, roughness
            )

        # To tensors
        sample = {
            "optical":  torch.from_numpy(optical).float().unsqueeze(0),
            "dem":      torch.from_numpy(elevation).float().unsqueeze(0),
            "valid":    torch.from_numpy(valid_mask.astype(np.uint8)).bool(),
            "dem_mean": dem_mean,
            "patch_id": patch_id,
        }
        if self.compute_auxiliary:
            sample["slope"]     = torch.from_numpy(slope).float().unsqueeze(0)
            sample["roughness"] = torch.from_numpy(roughness).float().unsqueeze(0)

        return sample


# ------------------------------------------------------------------------------
# Augmentation helper
# ------------------------------------------------------------------------------

def _augment(optical, elevation, valid_mask, slope, roughness):
    if np.random.rand() > 0.5:
        optical, elevation, valid_mask = np.fliplr(optical), np.fliplr(elevation), np.fliplr(valid_mask)
        if slope is not None:
            slope, roughness = np.fliplr(slope), np.fliplr(roughness)

    if np.random.rand() > 0.5:
        optical, elevation, valid_mask = np.flipud(optical), np.flipud(elevation), np.flipud(valid_mask)
        if slope is not None:
            slope, roughness = np.flipud(slope), np.flipud(roughness)

    k = np.random.randint(0, 4)
    if k > 0:
        optical, elevation, valid_mask = np.rot90(optical, k), np.rot90(elevation, k), np.rot90(valid_mask, k)
        if slope is not None:
            slope, roughness = np.rot90(slope, k), np.rot90(roughness, k)

    optical    = np.ascontiguousarray(optical)
    elevation  = np.ascontiguousarray(elevation)
    valid_mask = np.ascontiguousarray(valid_mask)
    if slope is not None:
        slope     = np.ascontiguousarray(slope)
        roughness = np.ascontiguousarray(roughness)

    return optical, elevation, valid_mask, slope, roughness