"""Optico MFSR Engine — Phase 2-4: Frame Alignment & Registration.

Includes:
- Harmony Anchor: Geometric-median based reference frame selection
- ECC Sub-pixel Registration with configurable parameters
- 2D Circular Statistics dither quality estimation
"""
import logging
from typing import Optional

import cv2
import numpy as np

from .constants import OpticoConfig

logger = logging.getLogger(__name__)


def calculate_dither_quality_2d(
    M_list: list[Optional[np.ndarray]],
) -> float:
    """Estimate sub-pixel dither quality using 2D circular statistics.

    Maps the fractional parts of translation vectors onto a unit torus
    and computes the 2D resultant vector length. A low resultant length
    indicates well-distributed sub-pixel shifts (ideal for SR).

    Parameters
    ----------
    M_list : list of Optional[np.ndarray]
        List of 2x3 affine matrices (or None for rejected frames).

    Returns
    -------
    float
        Dither quality in [0, 1]. Higher = better sub-pixel coverage.
    """
    valid_tx = []
    valid_ty = []
    for M in M_list:
        if M is not None:
            valid_tx.append(float(M[0, 2]))
            valid_ty.append(float(M[1, 2]))

    n = len(valid_tx)
    if n <= 1:
        return 0.0

    # Extract fractional parts (sub-pixel offsets)
    fx = np.array(valid_tx) - np.floor(valid_tx)
    fy = np.array(valid_ty) - np.floor(valid_ty)

    # Map to angles on the unit circle
    theta_x = 2.0 * np.pi * fx
    theta_y = 2.0 * np.pi * fy

    # Resultant vector length per axis
    mean_cos_x = float(np.mean(np.cos(theta_x)))
    mean_sin_x = float(np.mean(np.sin(theta_x)))
    mean_cos_y = float(np.mean(np.cos(theta_y)))
    mean_sin_y = float(np.mean(np.sin(theta_y)))

    Rx = np.sqrt(mean_cos_x**2 + mean_sin_x**2)
    Ry = np.sqrt(mean_cos_y**2 + mean_sin_y**2)

    # Joint 2D resultant: geometric mean preserves X-Y correlation
    R_2d = np.sqrt(Rx * Ry)

    # Quality = 1 - concentration (uniform = high quality)
    quality = float(1.0 - R_2d)
    return float(np.clip(quality, 0.0, 1.0))


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

    # Collect valid translation vectors
    translations = []
    valid_indices = []
    for i, M in enumerate(M_list):
        if M is not None:
            translations.append([float(M[0, 2]), float(M[1, 2])])
            valid_indices.append(i)

    if len(translations) < 2:
        return 0

    points = np.array(translations, dtype=np.float64)

    # Weiszfeld's algorithm for geometric median
    median = np.mean(points, axis=0)
    for _ in range(50):
        dists = np.linalg.norm(points - median, axis=1)
        dists = np.maximum(dists, 1e-8)
        weights = 1.0 / dists
        new_median = np.average(points, axis=0, weights=weights)
        if np.linalg.norm(new_median - median) < 1e-6:
            break
        median = new_median

    # Find frames near the geometric median (closest 50%)
    dists_to_median = np.linalg.norm(points - median, axis=1)
    near_threshold = float(np.percentile(dists_to_median, 50))

    # Among near-center frames, pick the sharpest
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


def align_images_ecc(
    images: list[np.ndarray],
    ref_idx: int = 0,
    config: Optional[OpticoConfig] = None,
) -> tuple[list[Optional[np.ndarray]], list[float]]:
    """Phase 2 & 4: ECC Sub-pixel Registration.

    Calculates sub-pixel alignment matrices using Enhanced Correlation
    Coefficient maximization. Uses MOTION_TRANSLATION to avoid
    overfitting to noise.

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

    align_sc = config.align_scale
    max_offset = config.max_offset
    n = len(images)

    if n == 0:
        raise ValueError("Empty image list")

    logger.info(
        "Aligning %d images using ECC (ref=%d, scale=%.2f)",
        n, ref_idx, align_sc,
    )

    ref_img = images[ref_idx]
    if ref_img.ndim != 3 or ref_img.shape[2] != 3:
        raise ValueError(f"Expected BGR images, got shape {ref_img.shape}")

    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    h, w = ref_gray.shape
    h_s, w_s = int(h * align_sc), int(w * align_sc)
    ref_gray_sm = cv2.resize(
        ref_gray, (w_s, h_s), interpolation=cv2.INTER_AREA
    )

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
        img_gray_sm = cv2.resize(
            img_gray, (w_s, h_s), interpolation=cv2.INTER_AREA
        )

        M = np.eye(2, 3, dtype=np.float32)
        try:
            cc, M = cv2.findTransformECC(
                ref_gray_sm, img_gray_sm, M, warp_mode, criteria,
                None, config.ecc_gauss_filt_size,
            )
            # Scale translation back to original resolution
            scale_factor = 1.0 / align_sc
            M[0, 2] *= scale_factor
            M[1, 2] *= scale_factor

            if abs(M[0, 2]) > max_offset or abs(M[1, 2]) > max_offset:
                logger.warning(
                    "Frame %d offset (%.2f, %.2f) exceeds max %.1f, "
                    "rejecting",
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
