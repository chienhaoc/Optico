"""Optico MFSR Engine — Phase 6: Dynamic Foreground Masking.

Detects and masks regions with non-rigid motion to prevent
alignment drift blur during Drizzle stacking.

Kernel size adaptive sizing
---------------------------
Fixed pixel kernel sizes (BG=7, SUBJ=11, BLUR=5) are effective for
small to medium images (≤ ~2 MP) but become negligibly small for
high-resolution input (e.g. 6000×4000 = 24 MP).  At that resolution
7 px = 0.17% of min_dim, which means morphological dilation barely
expands motion masks at all.

Fix: compute adaptive kernel sizes proportional to image min_dim,
clamped to the fixed constants as a lower bound and forced to odd:
  BG_kern   = max(BG_KERNEL_SIZE,   odd(round(min_dim * BG_KERNEL_MIN_DIM_FRAC)))
  SUBJ_kern = max(SUBJ_KERNEL_SIZE, odd(round(min_dim * SUBJ_KERNEL_MIN_DIM_FRAC)))
  BLUR_kern = max(MASK_BLUR_KSIZE,  odd(round(min_dim * MASK_BLUR_MIN_DIM_FRAC)))
"""
import logging
from typing import Optional

import cv2
import numpy as np

from .constants import (
    OpticoConfig,
    BG_KERNEL_SIZE, SUBJ_KERNEL_SIZE, SUBJ_DILATE_ITERATIONS,
    MASK_BLUR_KSIZE,
    BG_KERNEL_MIN_DIM_FRAC, SUBJ_KERNEL_MIN_DIM_FRAC, MASK_BLUR_MIN_DIM_FRAC,
)

logger = logging.getLogger(__name__)


def _adaptive_odd(base: int, min_dim: int, frac: float) -> int:
    """Compute adaptive odd kernel size: max(base, round(min_dim*frac)), forced odd."""
    sz = max(base, round(min_dim * frac))
    if sz % 2 == 0:
        sz += 1
    return sz


def calculate_dynamic_mask(
    images: list[np.ndarray],
    M_list: list[Optional[np.ndarray]],
    ref_idx: int = 0,
    config: Optional[OpticoConfig] = None,
) -> list[np.ndarray]:
    """Phase 6: Dynamic Foreground Masking (Dual-Threshold).

    For each frame, computes a soft weight map [0.0, 1.0] indicating
    how much each pixel should contribute to the Drizzle stack.
    Regions with detected motion are suppressed.

    Parameters
    ----------
    images : list of np.ndarray
        Input burst images (BGR, uint8).
    M_list : list of Optional[np.ndarray]
        Affine matrices from alignment (None = rejected frame).
    ref_idx : int
        Index of the reference frame.
    config : OpticoConfig, optional
        Configuration. Uses defaults if None.

    Returns
    -------
    list of np.ndarray
        Per-frame weight maps as float32 arrays in [0.0, 1.0].
    """
    if config is None:
        config = OpticoConfig()

    ref_img = images[ref_idx]
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = ref_gray.shape
    min_dim = min(h, w)

    # Adaptive kernel sizes: proportional to resolution, lower-bounded by constants
    bg_kern   = _adaptive_odd(BG_KERNEL_SIZE,   min_dim, BG_KERNEL_MIN_DIM_FRAC)
    subj_kern = _adaptive_odd(SUBJ_KERNEL_SIZE, min_dim, SUBJ_KERNEL_MIN_DIM_FRAC)
    blur_kern = _adaptive_odd(MASK_BLUR_KSIZE,  min_dim, MASK_BLUR_MIN_DIM_FRAC)

    logger.info(
        "Calculating dynamic masks (Dual-Threshold), "
        "img=%dx%d, kernels: BG=%d SUBJ=%d BLUR=%d",
        w, h, bg_kern, subj_kern, blur_kern,
    )

    # Gradient magnitude for edge-aware normalization
    grad_x = cv2.Sobel(ref_gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(ref_gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(grad_x, grad_y)

    # Poisson-Gaussian noise model: sigma = sqrt(aI + b)
    noise_std = np.sqrt(
        config.noise_gain * np.maximum(ref_gray, 0.0) + config.noise_offset
    )
    denom = noise_std + config.gradient_weight * grad + 1.0

    kernel_bg = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (bg_kern, bg_kern)
    )
    kernel_subj = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (subj_kern, subj_kern)
    )

    weight_maps: list[np.ndarray] = []

    for i, img in enumerate(images):
        if i == ref_idx or M_list[i] is None:
            weight_maps.append(np.ones((h, w), dtype=np.float32))
            continue

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        warped = cv2.warpAffine(
            img_gray, M_list[i], (w, h),
            flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
        )

        diff = np.abs(warped.astype(np.float32) - ref_gray)
        norm_diff = diff / denom

        bg_motion = (norm_diff > config.bg_threshold).astype(np.uint8)
        bg_motion = cv2.dilate(bg_motion, kernel_bg, iterations=1)

        subj_motion = (norm_diff > config.subj_threshold).astype(np.uint8)
        subj_motion = cv2.dilate(
            subj_motion, kernel_subj,
            iterations=SUBJ_DILATE_ITERATIONS,
        )

        total_motion = np.maximum(bg_motion, subj_motion)

        mask = np.ones((h, w), dtype=np.float32)
        mask[total_motion > 0] = 0.0
        mask = cv2.GaussianBlur(mask, (blur_kern, blur_kern), 0)

        weight_maps.append(mask)

        motion_pct = 100.0 * float(np.mean(total_motion > 0))
        logger.debug("Frame %d: %.1f%% motion pixels masked", i, motion_pct)

    return weight_maps


def calculate_retained_ratio(
    weight_maps: list[np.ndarray],
) -> float:
    """Calculate the Global Retained Pixel Ratio from weight maps."""
    if not weight_maps:
        return 0.0
    ratios = [float(np.mean(w)) for w in weight_maps]
    global_ratio = float(np.mean(ratios))
    logger.info("Global Retained Pixel Ratio: %.4f", global_ratio)
    return global_ratio
