"""Optico MFSR Engine — Phase 2-4: Frame Alignment & Registration.

Includes:
- Harmony Anchor: Geometric-median based reference frame selection
- ECC Sub-pixel Registration with adaptive scale selection
- N_eff Entropy Dither Quality (replaces Rayleigh-based metric)

Dither quality metric history
------------------------------
The original calculate_dither_quality_2d() applied a bias correction
(null_floor = π / (4N)) to the Rayleigh resultant R that removed nearly
all signal: in sandbox testing across 8 representative sub-pixel
distributions, 6 out of 8 returned Q = 1.0 regardless of actual coverage.
When Q = 1.0, blur_limit = decay * sqrt(1 / (1-1)) = ∞, effectively
disabling the pre-flight dither branch entirely.

Replacement: calculate_dither_quality_neff() uses Shannon entropy on a
4×4 sub-pixel histogram to compute N_eff = 2^H (effective independent
sub-pixel positions). N_eff is returned as the dither quality figure and
used directly in preflight.py as blur_limit = decay * sqrt(N_eff).

This is both physically interpretable (N_eff uniform positions support
sqrt(N_eff) × super-resolution) and numerically stable across all
tested distributions.
"""
import logging
import math
from typing import Optional

import cv2
import numpy as np

from .constants import OpticoConfig

logger = logging.getLogger(__name__)

# Number of bins per sub-pixel axis for the dither histogram.
# 4 bins = 4×4 = 16 cells, supporting N_eff in [1, 16].
# Matches typical handheld burst diversity; change to 8 for finer resolution.
_DITHER_GRID_N: int = 4


def calculate_dither_quality_neff(
    M_list: list[Optional[np.ndarray]],
) -> float:
    """Estimate sub-pixel dither quality via Shannon entropy N_eff.

    Splits the [0, 1) × [0, 1) sub-pixel offset space into a
    _DITHER_GRID_N × _DITHER_GRID_N histogram, computes Shannon entropy
    H (bits), and returns N_eff = 2^H as a count of effective independent
    sub-pixel positions. Maximum N_eff = _DITHER_GRID_N² = 16 (perfectly
    uniform). Minimum N_eff = 1.0 (all frames at the same sub-pixel
    position).

    This replaces the Rayleigh-statistic approach, which applied an
    over-aggressive small-sample bias correction that saturated Q to 1.0
    for almost all realistic burst distributions, disabling the pre-flight
    blur_limit branch.

    Parameters
    ----------
    M_list : list of Optional[np.ndarray]
        List of 2×3 affine matrices (or None for rejected frames).

    Returns
    -------
    float
        N_eff in [1.0, _DITHER_GRID_N²]. Higher = better sub-pixel coverage.
        The caller (preflight.py) is expected to use sqrt(N_eff) as the
        dither contribution to the blur limit.
    """
    fxs, fys = [], []
    for M in M_list:
        if M is not None:
            fxs.append(float(M[0, 2]) % 1.0)
            fys.append(float(M[1, 2]) % 1.0)

    n = len(fxs)
    if n <= 1:
        return 1.0

    fx = np.array(fxs)
    fy = np.array(fys)

    hist, _, _ = np.histogram2d(
        fx, fy,
        bins=_DITHER_GRID_N,
        range=[[0.0, 1.0], [0.0, 1.0]],
    )
    hist = hist / max(hist.sum(), 1e-9)
    nonzero = hist[hist > 0]
    H_bits = float(-np.sum(nonzero * np.log2(nonzero)))  # 0 .. log2(grid_n^2)
    N_eff = 2.0 ** H_bits                                # 1.0 .. grid_n^2

    logger.debug(
        "Dither quality: n=%d, H=%.3f bits, N_eff=%.2f (max %.0f)",
        n, H_bits, N_eff, _DITHER_GRID_N ** 2,
    )
    return float(N_eff)


def select_reference_frame(
    images: list[np.ndarray],
    M_list: list[Optional[np.ndarray]],
) -> int:
    """Harmony Anchor: Select the optimal reference frame.

    Computes the geometric median of all translation vectors (Weiszfeld's
    algorithm) to find the most 'harmonious' structural baseline, then
    selects the sharpest frame near that center.

    Parameters
    ----------
    images : list of np.ndarray
        Input burst images (BGR, uint8).
    M_list : list of Optional[np.ndarray]
        Affine matrices from an initial coarse alignment.

    Returns
    -------
    int
        Index of the selected reference frame.
    """
    n = len(images)
    if n <= 1:
        return 0

    translations = []
    valid_indices = []
    for i, M in enumerate(M_list):
        if M is not None:
            translations.append([float(M[0, 2]), float(M[1, 2])])
            valid_indices.append(i)

    if len(translations) < 2:
        return 0

    points = np.array(translations, dtype=np.float64)

    # Weiszfeld geometric median
    median = np.mean(points, axis=0)
    for _ in range(50):
        dists = np.maximum(np.linalg.norm(points - median, axis=1), 1e-8)
        weights = 1.0 / dists
        new_median = np.average(points, axis=0, weights=weights)
        if np.linalg.norm(new_median - median) < 1e-6:
            break
        median = new_median

    dists_to_median = np.linalg.norm(points - median, axis=1)
    near_threshold = float(np.percentile(dists_to_median, 50))

    best_idx = 0
    best_sharpness = -1.0
    for j, idx in enumerate(valid_indices):
        if dists_to_median[j] <= near_threshold + 1e-6:
            gray = cv2.cvtColor(images[idx], cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if sharpness > best_sharpness:
                best_sharpness = sharpness
                best_idx = idx

    logger.info(
        "Harmony Anchor selected frame %d (sharpness=%.1f)",
        best_idx, best_sharpness,
    )
    return best_idx


def _adaptive_align_scale(img_h: int, img_w: int) -> float:
    """Choose ECC downscale factor based on image resolution.

    Too small a scale loses sub-pixel information needed for accurate
    ECC convergence; too large wastes compute and memory.
    Benchmarks show scale=0.50 is optimal for <1024px (largest dim)
    while scale=0.75 is better for >1024px (sub-pixel detail preserved).
    Very large images (>2048px) use 0.50 again because the raw resolution
    provides sufficient ECC signal even after halving.

    Parameters
    ----------
    img_h, img_w : int
        Input image dimensions.

    Returns
    -------
    float
        ECC downscale factor.
    """
    max_dim = max(img_h, img_w)
    if max_dim <= 512:
        return 0.75   # small: keep as much detail as possible
    elif max_dim <= 1024:
        return 0.50   # medium: good balance
    elif max_dim <= 2048:
        return 0.75   # high-res: 0.75 outperforms 0.5 on large shifts
    else:
        return 0.50   # very high-res: 0.5 sufficient, saves memory


def align_images_ecc(
    images: list[np.ndarray],
    ref_idx: int = 0,
    config: Optional[OpticoConfig] = None,
) -> tuple[list[Optional[np.ndarray]], list[float]]:
    """Phase 2 & 4: ECC Sub-pixel Registration.

    Calculates sub-pixel alignment matrices using Enhanced Correlation
    Coefficient maximization. Uses MOTION_TRANSLATION to avoid
    overfitting to noise.

    The downscale factor is chosen adaptively by _adaptive_align_scale()
    unless config.align_scale is explicitly overridden (non-default value).

    Parameters
    ----------
    images : list of np.ndarray
        Input burst images (BGR, uint8).
    ref_idx : int
        Index of the reference frame.
    config : OpticoConfig, optional
        Configuration parameters. Uses defaults if None.

    Returns
    -------
    M_list : list of Optional[np.ndarray]
        2x3 affine matrices (None for rejected frames).
    cc_list : list of float
        ECC correlation coefficients per frame.
    """
    if config is None:
        config = OpticoConfig()

    max_offset = config.max_offset
    n = len(images)

    if n == 0:
        raise ValueError("Empty image list")

    ref_img = images[ref_idx]
    if ref_img.ndim != 3 or ref_img.shape[2] != 3:
        raise ValueError(f"Expected BGR images, got shape {ref_img.shape}")

    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    h, w = ref_gray.shape

    # Adaptive scale: use config value only if user explicitly overrode it;
    # otherwise derive from image dimensions.
    from .constants import DEFAULT_ALIGN_SCALE
    if config.align_scale != DEFAULT_ALIGN_SCALE:
        align_sc = config.align_scale
    else:
        align_sc = _adaptive_align_scale(h, w)

    h_s = max(8, int(h * align_sc))
    w_s = max(8, int(w * align_sc))

    logger.info(
        "Aligning %d images using ECC (ref=%d, scale=%.2f, img=%dx%d)",
        n, ref_idx, align_sc, w, h,
    )

    ref_gray_sm = cv2.resize(ref_gray, (w_s, h_s), interpolation=cv2.INTER_AREA)

    M_list: list[Optional[np.ndarray]] = []
    cc_list: list[float] = []

    warp_mode = cv2.MOTION_TRANSLATION
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        config.ecc_iterations,
        config.ecc_epsilon,
    )

    for i in range(n):
        if i == ref_idx:
            M_list.append(np.eye(2, 3, dtype=np.float32))
            cc_list.append(1.0)
            continue

        img_gray = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        img_gray_sm = cv2.resize(img_gray, (w_s, h_s), interpolation=cv2.INTER_AREA)

        M = np.eye(2, 3, dtype=np.float32)
        try:
            cc, M = cv2.findTransformECC(
                ref_gray_sm, img_gray_sm, M, warp_mode, criteria,
                None, config.ecc_gauss_filt_size,
            )
            scale_factor = 1.0 / align_sc
            M[0, 2] *= scale_factor
            M[1, 2] *= scale_factor

            if abs(M[0, 2]) > max_offset or abs(M[1, 2]) > max_offset:
                logger.warning(
                    "Frame %d offset (%.2f, %.2f) exceeds max %.1f, rejecting",
                    i, M[0, 2], M[1, 2], max_offset,
                )
                M_list.append(None)
                cc_list.append(0.0)
            else:
                M_list.append(M)
                cc_list.append(float(cc))
                logger.debug(
                    "Frame %d: cc=%.4f, offset=(%.3f, %.3f)",
                    i, cc, M[0, 2], M[1, 2],
                )
        except cv2.error as e:
            logger.warning("ECC failed for frame %d: %s", i, e)
            M_list.append(None)
            cc_list.append(0.0)

    valid_count = sum(1 for m in M_list if m is not None)
    logger.info("Alignment complete: %d/%d frames valid", valid_count, n)
    return M_list, cc_list
