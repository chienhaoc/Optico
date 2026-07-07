"""Optico MFSR Engine — Phase 8: Drizzle Stacking.

Variable-Pixel Linear Reconstruction (Fruchter & Hook, 2002)
adapted for handheld burst photography with:
- Weighted accumulation using dynamic motion masks
- Active Memory Chunking for bounded RAM usage
- Per-channel processing for color fidelity
"""
import gc
import logging
from typing import Optional

import cv2
import numpy as np

from .constants import OpticoConfig, DRIZZLE_WEIGHT_FLOOR

logger = logging.getLogger(__name__)


def _drizzle_chunk_vectorized(
    images: list[np.ndarray],
    M_list: list[Optional[np.ndarray]],
    weight_maps: list[np.ndarray],
    scale: float,
    pixfrac: float,
    y_start_hr: int,
    y_end_hr: int,
    hr_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized Drizzle for a single HR chunk.

    Uses warpAffine to map each input frame onto the HR grid,
    then accumulates with mask weighting.

    Parameters
    ----------
    images : list of np.ndarray
        Input burst (BGR, uint8).
    M_list : list of Optional[np.ndarray]
        Translation matrices.
    weight_maps : list of np.ndarray
        Per-frame weight maps (float32, [0, 1]).
    scale : float
        Output upscale factor.
    pixfrac : float
        Pixel fraction (droplet size, 0-1).
    y_start_hr, y_end_hr : int
        Vertical range of this chunk in HR coordinates.
    hr_w : int
        Width of the HR canvas.

    Returns
    -------
    numerator : np.ndarray, shape (chunk_h, hr_w, 3), float64
    denominator : np.ndarray, shape (chunk_h, hr_w), float64
    """
    chunk_h = y_end_hr - y_start_hr
    numerator = np.zeros((chunk_h, hr_w, 3), dtype=np.float64)
    denominator = np.zeros((chunk_h, hr_w), dtype=np.float64)

    # Pre-create coordinate grid for the HR chunk
    u_vals = np.arange(hr_w, dtype=np.float64)
    v_local_vals = np.arange(chunk_h, dtype=np.float64)
    u_grid, v_local_grid = np.meshgrid(u_vals, v_local_vals)

    # Absolute vertical coordinates on HR canvas
    v_grid = v_local_grid + y_start_hr

    # HR pixel bounding boxes: [x_h1, x_h2] x [y_h1, y_h2]
    # Local coordinates within the chunk (centered around integer pixel coordinates)
    x_h1 = u_grid - 0.5
    x_h2 = u_grid + 0.5
    y_h1 = v_local_grid - 0.5
    y_h2 = v_local_grid + 0.5

    # Half width of droplet on HR canvas
    r_droplet = 0.5 * pixfrac * scale

    for img, M, wmap in zip(images, M_list, weight_maps):
        if M is None:
            continue

        lr_h, lr_w = img.shape[:2]

        tx = float(M[0, 2])
        ty = float(M[1, 2])

        # Map HR coordinates back to fractional LR space:
        # LR_coord = HR_coord / scale + translation
        x_LR_prime = u_grid / scale + tx
        y_LR_prime = v_grid / scale + ty

        # Find the 4 nearest integer LR pixels surrounding the mapped point
        x0 = np.floor(x_LR_prime).astype(np.int32)
        y0 = np.floor(y_LR_prime).astype(np.int32)

        wmap_2d = wmap if wmap.ndim == 2 else wmap[:, :, 0]

        # Evaluate the 4 nearest neighbors: (x0, y0), (x0+1, y0), (x0, y0+1), (x0+1, y0+1)
        for dx_offset in (0, 1):
            for dy_offset in (0, 1):
                xn = x0 + dx_offset
                yn = y0 + dy_offset

                # In-bounds mask for this neighbor
                in_bounds = (xn >= 0) & (xn < lr_w) & (yn >= 0) & (yn < lr_h)

                # Clamp coordinates to safely index array
                xn_clamped = np.clip(xn, 0, lr_w - 1)
                yn_clamped = np.clip(yn, 0, lr_h - 1)

                # Gather raw values and motion mask weights
                val = img[yn_clamped, xn_clamped].astype(np.float64)
                w_motion = wmap_2d[yn_clamped, xn_clamped].astype(np.float64)
                w_motion = w_motion * in_bounds.astype(np.float64)

                # Map neighbor's center to HR canvas:
                # HR_coord = scale * (LR_coord - translation)
                # Local y offset is adjusted by subtracting y_start_hr
                x_c = scale * (xn.astype(np.float64) - tx)
                y_c = scale * (yn.astype(np.float64) - ty) - y_start_hr

                # Droplet boundaries on HR canvas: [x_d1, x_d2] x [y_d1, y_d2]
                x_d1 = x_c - r_droplet
                x_d2 = x_c + r_droplet
                y_d1 = y_c - r_droplet
                y_d2 = y_c + r_droplet

                # Calculate overlap interval in x and y
                overlap_x = np.maximum(0.0, np.minimum(x_h2, x_d2) - np.maximum(x_h1, x_d1))
                overlap_y = np.maximum(0.0, np.minimum(y_h2, y_d2) - np.maximum(y_h1, y_d1))

                # Overlap area: shape (chunk_h, hr_w)
                overlap_area = overlap_x * overlap_y

                # Combined weight for this neighbor
                combined_w = w_motion * overlap_area

                # Accumulate
                for c in range(3):
                    numerator[:, :, c] += combined_w * val[:, :, c]
                denominator += combined_w

    return numerator, denominator


def drizzle_stack(
    images: list[np.ndarray],
    M_list: list[Optional[np.ndarray]],
    weight_maps: list[np.ndarray],
    scale: float,
    ref_idx: int = 0,
    config: Optional[OpticoConfig] = None,
) -> np.ndarray:
    """Phase 8: Drizzle Multi-Frame Stacking with Memory Chunking.

    Implements Variable-Pixel Linear Reconstruction adapted for
    handheld burst photography. The HR canvas is split into horizontal
    chunks to bound peak memory usage.

    Parameters
    ----------
    images : list of np.ndarray
        Input burst (BGR, uint8).
    M_list : list of Optional[np.ndarray]
        Translation matrices from alignment.
    weight_maps : list of np.ndarray
        Per-frame weight maps from dynamic masking.
    scale : float
        Output upscale factor (from Pre-flight).
    ref_idx : int
        Reference frame index.
    config : OpticoConfig, optional
        Configuration.

    Returns
    -------
    np.ndarray
        High-resolution output image (BGR, float32, [0, 255]).
    """
    if config is None:
        config = OpticoConfig()

    pixfrac = config.pixfrac
    num_chunks = config.num_chunks

    if not images:
        raise ValueError("Empty image list")

    lr_h, lr_w = images[0].shape[:2]
    hr_h = int(round(lr_h * scale))
    hr_w = int(round(lr_w * scale))

    logger.info(
        "Drizzle: %d frames, scale=%.2f, pixfrac=%.2f, "
        "LR=%dx%d -> HR=%dx%d, chunks=%d",
        len(images), scale, pixfrac, lr_w, lr_h, hr_w, hr_h, num_chunks,
    )

    # Output canvas
    output = np.zeros((hr_h, hr_w, 3), dtype=np.float32)

    # Split into horizontal chunks
    chunk_boundaries = np.linspace(0, hr_h, num_chunks + 1, dtype=int)

    for chunk_idx in range(num_chunks):
        y_start = int(chunk_boundaries[chunk_idx])
        y_end = int(chunk_boundaries[chunk_idx + 1])
        if y_start >= y_end:
            continue

        logger.debug(
            "Processing chunk %d/%d (rows %d-%d)",
            chunk_idx + 1, num_chunks, y_start, y_end,
        )

        numerator, denominator = _drizzle_chunk_vectorized(
            images, M_list, weight_maps, scale, pixfrac,
            y_start, y_end, hr_w,
        )

        # Normalize: output = sum(w*v) / sum(w)
        safe_denom = np.maximum(denominator, DRIZZLE_WEIGHT_FLOOR)
        for c in range(3):
            output[y_start:y_end, :, c] = (
                numerator[:, :, c] / safe_denom
            ).astype(np.float32)

        # Free chunk memory
        del numerator, denominator, safe_denom
        gc.collect()

    # Clamp to valid range
    np.clip(output, 0.0, 255.0, out=output)

    logger.info("Drizzle stacking complete: output shape %s", output.shape)
    return output
