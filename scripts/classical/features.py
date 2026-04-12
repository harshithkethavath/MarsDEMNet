"""
scripts/classical/features.py

Feature extraction pipeline for Implementation 1 — Classical RF Baseline.

Loads raw MCTED patch files directly (bypasses MCTEDDataset — no tensors,
no z-score normalization). Produces a fixed-length feature vector per patch
from the optical image, and three target scalars from the DEM.

Feature vector layout (concatenated in this order):
    [0  : 26 ] LBP histogram          — 26 bins (radius=3, 24 neighbors, uniform)
    [26 : 38 ] Gabor mean responses   — 12 filters × 1 mean = 12 values
    [38 : 50 ] Gabor var  responses   — 12 filters × 1 var  = 12 values
    [50 : 59 ] HOG gradient histogram — 9 orientation bins
    [59 : 63 ] Intensity moments      — mean, std, skewness, kurtosis
    Total: 63 features

Target vector layout (3 scalars per patch):
    y[0] — elevation std  (meters)  : terrain roughness / undulation magnitude
    y[1] — elevation mean (meters)  : always ~0 after mean subtraction, kept for completeness
    y[2] — dominant slope direction (degrees, 0–180): orientation of steepest descent

Usage:
    from scripts.classical.features import build_feature_matrix

    X_train, y_train, ids_train = build_feature_matrix("data/train/", max_patches=None)
    X_val,   y_val,   ids_val   = build_feature_matrix("data/val/",   max_patches=None)
"""

import os
import warnings
import numpy as np
from PIL import Image
import tifffile
import cv2
from scipy.ndimage import sobel, gaussian_filter
from skimage.feature import local_binary_pattern
from skimage.filters import gabor_kernel
from scipy.stats import skew, kurtosis
from multiprocessing import Pool, cpu_count
from typing import Optional, Tuple, List


# ─────────────────────────────────────────────────────────────────────────────
# Constants matching the proposal spec exactly
# ─────────────────────────────────────────────────────────────────────────────

LBP_RADIUS       = 3
LBP_N_POINTS     = 24          # 8 * radius
LBP_METHOD       = "uniform"   # rotation-invariant uniform LBP → 26 bins

GABOR_WAVELENGTHS    = [4, 8, 16]   # pixels
GABOR_ORIENTATIONS   = [0, 45, 90, 135]  # degrees
# 4 orientations × 3 scales = 12 filters → 24 features (mean + var each)

HOG_N_BINS       = 9
HOG_CELL_SIZE    = 16          # non-overlapping cells, pixels

MIN_VALID_FRAC   = 0.30        # discard patches with fewer valid DEM pixels than this
PIXEL_SIZE_M     = 6.0         # CTX ground sampling distance, meters/pixel

# Precompute Gabor kernels once at module load — building them inside
# _gabor_features() on every patch wastes ~20ms per call unnecessarily.
_GABOR_KERNELS = [
    (
        np.real(gabor_kernel(frequency=1.0 / wl, theta=np.deg2rad(th))).astype(np.float32),
        np.imag(gabor_kernel(frequency=1.0 / wl, theta=np.deg2rad(th))).astype(np.float32),
    )
    for wl in GABOR_WAVELENGTHS
    for th in GABOR_ORIENTATIONS
]
FEATURE_IMG_SIZE = 256         # downsample optical to this before feature extraction.
                               # LBP and Gabor are global texture summaries — histogram
                               # aggregation makes full 518×518 resolution unnecessary.
                               # (256/518)² ≈ 4× fewer pixels → ~4× faster LBP + Gabor.


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(
    root_dir: str,
    max_patches: Optional[int] = None,
    verbose: bool = True,
    n_workers: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Scan root_dir for MCTED patch quads, extract features and targets.

    Args:
        root_dir:    Path to data/train/ or data/val/.
        max_patches: If set, stop after this many patches.
        verbose:     Print progress.
        n_workers:   Number of parallel worker processes.
                     0 = use all available CPUs.
                     1 = single-process (easier to debug).

    Returns:
        X     : (N, 63) float32 feature matrix.
        y     : (N, 3)  float32 target matrix — [elev_std, elev_mean, slope_dir].
        ids   : list of N patch_id strings.
    """
    all_files = os.listdir(root_dir)
    patch_ids = sorted(
        f[: -len(".optical.png")]
        for f in all_files
        if f.endswith(".optical.png")
    )

    if len(patch_ids) == 0:
        raise ValueError(f"No .optical.png files found in {root_dir}.")

    if max_patches is not None:
        patch_ids = patch_ids[:max_patches]

    # Build argument list for workers: (root_dir, patch_id)
    args = [(root_dir, pid) for pid in patch_ids]

    n_workers = n_workers if n_workers > 0 else cpu_count()
    # For small jobs, single-process is faster (no fork overhead)
    if len(patch_ids) <= 32:
        n_workers = 1

    if verbose:
        print(f"  Extracting {len(patch_ids)} patches with {n_workers} workers ...")

    if n_workers == 1:
        results = [_process_one(a) for a in args]
    else:
        with Pool(processes=n_workers) as pool:
            results = pool.map(_process_one, args)

    # Filter skipped patches (returned as None)
    X_rows, y_rows, valid_ids = [], [], []
    skipped = 0
    for res in results:
        if res is None:
            skipped += 1
            continue
        feat, targ, pid = res
        X_rows.append(feat)
        y_rows.append(targ)
        valid_ids.append(pid)

    if verbose:
        print(f"  Done. {len(valid_ids)} patches kept, {skipped} skipped.")

    X = np.stack(X_rows, axis=0).astype(np.float32)
    y = np.stack(y_rows, axis=0).astype(np.float32)

    return X, y, valid_ids


def _process_one(args: Tuple[str, str]) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    """
    Worker function: load one patch, extract features and targets.
    Returns None if the patch should be skipped.
    Must be module-level (not a lambda or closure) for multiprocessing pickle.
    """
    root_dir, patch_id = args
    base = os.path.join(root_dir, patch_id)

    try:
        optical, elevation, valid_mask = _load_patch(base)
    except Exception as e:
        warnings.warn(f"Failed to load {patch_id}: {e}", RuntimeWarning)
        return None

    if valid_mask.mean() < MIN_VALID_FRAC:
        return None

    feat = extract_features(optical)
    targ = extract_targets(elevation, valid_mask)
    return feat, targ, patch_id


# ─────────────────────────────────────────────────────────────────────────────
# File loading (raw, no z-score — intentional)
# ─────────────────────────────────────────────────────────────────────────────

def _load_patch(
    base: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one patch quad from disk.

    Returns:
        optical    : (518, 518) uint8  — percentile-clipped, rescaled to [0, 255].
                     NOT z-score normalized. LBP and Gabor need stable uint8-like
                     values; z-score floats produce correct but less discriminative
                     texture histograms.
        elevation  : (518, 518) float32 — mean-subtracted (relative topography, meters).
        valid_mask : (518, 518) bool    — True = valid DEM pixel.
    """
    # Optical: RGB saved, all channels identical → convert to L
    optical_raw = np.array(
        Image.open(f"{base}.optical.png").convert("L"),
        dtype=np.float32,
    )

    # Percentile clip — same as dataset.py — then rescale to [0, 255] uint8.
    # This removes hot pixels and preserves the relative intensity structure
    # that LBP and Gabor need, without compressing dynamic range.
    p_low  = np.percentile(optical_raw, 2)
    p_high = np.percentile(optical_raw, 98)
    optical_clipped = np.clip(optical_raw, p_low, p_high)

    # Rescale to [0, 255]
    rng = p_high - p_low
    if rng < 1e-6:
        optical_uint8 = np.zeros_like(optical_clipped, dtype=np.uint8)
    else:
        optical_uint8 = ((optical_clipped - p_low) / rng * 255).astype(np.uint8)

    # Downsample for feature extraction — texture statistics are spatially
    # aggregated anyway (histograms, means), so full 518×518 resolution adds
    # computation without meaningful feature quality gain.
    if FEATURE_IMG_SIZE < optical_uint8.shape[0]:
        from PIL import Image as _Image
        optical_uint8 = np.array(
            _Image.fromarray(optical_uint8).resize(
                (FEATURE_IMG_SIZE, FEATURE_IMG_SIZE), _Image.BILINEAR
            ),
            dtype=np.uint8,
        )

    # Elevation: float32 TIF
    elevation = tifffile.imread(f"{base}.elevation.tif").astype(np.float32)
    dem_mean  = float(elevation.mean())
    elevation = elevation - dem_mean   # relative topography

    # Validity masks: 0 = valid, 255 = invalid
    nan_mask = np.array(Image.open(f"{base}.initial_nan_mask.png").convert("L"), dtype=np.uint8)
    dev_mask = np.array(Image.open(f"{base}.deviation_mask.png").convert("L"),   dtype=np.uint8)
    valid_mask = (nan_mask == 0) & (dev_mask == 0)

    return optical_uint8, elevation, valid_mask


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction (optical image → 87-dim vector)
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(optical: np.ndarray) -> np.ndarray:
    """
    Extract 87-dimensional feature vector from a single optical patch.

    Args:
        optical: (518, 518) uint8 array, percentile-clipped.

    Returns:
        (87,) float32 feature vector.
    """
    f_lbp    = _lbp_features(optical)      # 26
    f_gabor  = _gabor_features(optical)    # 48  (24 mean + 24 var)
    f_hog    = _hog_features(optical)      # 9
    f_stats  = _intensity_moments(optical) # 4

    return np.concatenate([f_lbp, f_gabor, f_hog, f_stats]).astype(np.float32)


def _lbp_features(optical: np.ndarray) -> np.ndarray:
    """
    Local Binary Pattern histogram.

    uniform LBP with radius=3, 24 neighbors → 26 uniform pattern bins
    (24 rotation-invariant uniform patterns + 1 non-uniform bin).
    Histogram normalized to sum to 1.

    LBP encodes micro-texture by comparing each pixel to its circular
    neighborhood. Boulder fields, dune fields, and smooth plains have
    characteristically different LBP distributions.
    """
    lbp = local_binary_pattern(optical, LBP_N_POINTS, LBP_RADIUS, method=LBP_METHOD)
    n_bins = LBP_N_POINTS + 2   # uniform: P+1 uniform patterns + 1 non-uniform
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)   # (26,)


def _gabor_features(optical: np.ndarray) -> np.ndarray:
    """
    Gabor filter bank: 4 orientations × 3 wavelengths = 12 filters.
    For each filter: mean and variance of response magnitude.
    → 12 mean values + 12 variance values = 24 features.

    Kernels are precomputed once at module load (_GABOR_KERNELS) so
    gabor_kernel() is never called per patch.
    """
    optical_f = optical.astype(np.float32)
    means     = np.empty(12, dtype=np.float32)
    variances = np.empty(12, dtype=np.float32)

    for idx, (k_real, k_imag) in enumerate(_GABOR_KERNELS):
        real_resp = cv2.filter2D(optical_f, -1, k_real)
        imag_resp = cv2.filter2D(optical_f, -1, k_imag)
        magnitude = np.sqrt(real_resp**2 + imag_resp**2)
        means[idx]     = magnitude.mean()
        variances[idx] = magnitude.var()

    return np.concatenate([means, variances])   # (24,)


def _hog_features(optical: np.ndarray) -> np.ndarray:
    """
    HOG-inspired gradient orientation histogram.

    Computes gradient magnitude and orientation at each pixel, then
    accumulates a single global orientation histogram weighted by magnitude,
    using HOG_N_BINS=9 orientation bins over [0°, 180°) (unsigned gradients).

    High-gradient pixels (crater walls, scarps) dominate the histogram.
    Flat featureless terrain produces a low-magnitude, nearly uniform histogram.
    → 9 features.
    """
    optical_f = optical.astype(np.float32)

    # Sobel gradients
    gx = sobel(optical_f, axis=1)
    gy = sobel(optical_f, axis=0)

    magnitude   = np.sqrt(gx**2 + gy**2)
    orientation = np.degrees(np.arctan2(gy, gx)) % 180   # unsigned [0, 180)

    # Magnitude-weighted histogram over orientation bins
    hist, _ = np.histogram(
        orientation.ravel(),
        bins=HOG_N_BINS,
        range=(0, 180),
        weights=magnitude.ravel(),
        density=True,
    )
    return hist.astype(np.float32)   # (9,)


def _intensity_moments(optical: np.ndarray) -> np.ndarray:
    """
    Four statistical moments of the pixel intensity distribution.

    mean     — average brightness (illumination / albedo)
    std      — contrast / dynamic range within patch
    skewness — asymmetry of brightness distribution (shadowed vs. sunlit bias)
    kurtosis — peakedness (many pixels at one brightness vs. broad spread)

    → 4 features.
    """
    flat = optical.ravel().astype(np.float64)
    return np.array([
        flat.mean(),
        flat.std(),
        skew(flat),
        kurtosis(flat),
    ], dtype=np.float32)   # (4,)


# ─────────────────────────────────────────────────────────────────────────────
# Target extraction (DEM → 3 scalars over valid pixels only)
# ─────────────────────────────────────────────────────────────────────────────

def extract_targets(
    elevation: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """
    Compute three patch-level DEM statistics over valid pixels only.

    Args:
        elevation  : (518, 518) float32 — mean-subtracted elevation in meters.
        valid_mask : (518, 518) bool    — True = valid pixel.

    Returns:
        (3,) float32 array: [elev_std, elev_mean, dominant_slope_dir]

        elev_std          (meters)  — elevation standard deviation over valid pixels.
                                      Primary target. High = rough/undulating terrain.
                                      Low = flat plains.

        elev_mean         (meters)  — mean elevation over valid pixels.
                                      Nearly always ~0 (mean was subtracted globally),
                                      but valid-pixel mean can differ slightly.
                                      Kept for completeness.

        dominant_slope_dir (degrees) — orientation (0–180°) of the dominant gradient
                                       direction in the DEM. Computed from the circular
                                       mean of per-pixel gradient orientations, weighted
                                       by gradient magnitude. Captures overall terrain tilt.
    """
    valid_elev = elevation[valid_mask]

    elev_mean = float(valid_elev.mean()) if len(valid_elev) > 0 else 0.0
    elev_std  = float(valid_elev.std())  if len(valid_elev) > 0 else 0.0

    # Dominant slope direction: gradient orientation of DEM weighted by magnitude
    gx = sobel(elevation, axis=1) / (8 * PIXEL_SIZE_M)
    gy = sobel(elevation, axis=0) / (8 * PIXEL_SIZE_M)

    magnitude   = np.sqrt(gx**2 + gy**2)
    orientation = np.degrees(np.arctan2(gy, gx)) % 180   # unsigned, [0, 180)

    # Restrict to valid pixels for the weighted orientation mean
    mag_valid   = magnitude[valid_mask]
    orient_valid = orientation[valid_mask]

    if mag_valid.sum() < 1e-9:
        dominant_dir = 0.0
    else:
        # Circular mean using doubled angles to handle the [0, 180) periodicity
        theta2  = np.deg2rad(2 * orient_valid)
        sin_sum = (mag_valid * np.sin(theta2)).sum()
        cos_sum = (mag_valid * np.cos(theta2)).sum()
        dominant_dir = float(np.degrees(np.arctan2(sin_sum, cos_sum)) / 2 % 180)

    return np.array([elev_std, elev_mean, dominant_dir], dtype=np.float32)
