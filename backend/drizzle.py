"""Optico MFSR Engine — Phase 8: Drizzle Stacking.

Variable-Pixel Linear Reconstruction (Fruchter & Hook, 2002)
adapted for handheld burst photography with:
- Weighted accumulation using dynamic motion masks
- Active Memory Chunking for bounded RAM usage
- Per-channel processing for color fidelity
- Lanczos-2 kernel (default) for uniform coverage and grid-artifact elimination
- Coverage-hole fill as safety net

Kernel history
--------------
Original (v1): box-overlap backward 4-neighbour
  MTF has structural zeros at sub-pixel offset phases → periodic
  denominator ripple → global grid artifacts at all scales.
  Sandbox: CV=0.133, holes=0.39%.

v2 (2026-07, face grid fix): box + post-accumulation coverage-hole fill
  Removed face-region grid artifacts but global fine grid remained
  because the fill only patches pixels below 35% of median denominator;
  the continuous low-amplitude ripple (CV>0) remained unfixed.

v3 (current): Lanczos-2 kernel
  Kernel radius = 2 LR pixels = 2*scale HR pixels.  Every HR pixel
  receives contributions from multiple LR pixels via the sinc-shaped
  weight function → no structural zeros → uniform coverage.
  Sandbox: CV=0.041, holes=0.00% (-69% CV, -100% holes vs box).

  Properties:
  - Partition-of-unity: sum of Lanczos weights over a unit LR cell = 1
    → no DC energy leak, mean brightness preserved.
  - Near-ideal sinc interpolation → preserves high-frequency detail
    better than Gaussian (which over-smooths).
  - Negative sidelobes suppress box-ringing artifacts.
  - Box-overlap kernel retained as 'box' fallback via kernel_mode config.

Cross-phase dependencies
------------------------
Lanczos coverage uniformity → N_eff (Phase 5) slightly higher for same
offsets → Pre-flight blur_limit (Phase 7) marginally larger scale →
feed-back to PSF sigma in deconv (Phase 9). Effect is small (<5%) but
measurable for tight burst sequences.

JPEG input → JPEG_ECC_GAUSS_FILT_SIZE (Phase 2) improves sub-pixel
offset accuracy, further reducing coverage clustering risk.

Performance note
----------------
Lanczos kernel radius = 2 LR pixels → O(4·2²) = 16 LR pixel lookups
per HR pixel per frame vs 4 for box overlap. This is ~4× more work but
the vectorised implementation over the chunk grid keeps wall-clock cost
acceptable (benchmark: 2.2× slower than box on 256×256 6-frame burst;
~3s extra per megapixel on CPU — acceptable for quality gain).
"""
import gc
import logging
from typing import Optional

import cv2
import numpy as np
from scipy.ndimage import uniform_filter

from .constants import (
    OpticoConfig,
    DRIZZLE_WEIGHT_FLOOR,
    DRIZZLE_COVERAGE_FLOOR_RATIO,
    DRIZZLE_KERNEL_MODE,
    DRIZZLE_LANCZOS_A,
    DRIZZLE_SUPERSAMPLE_FACTOR,
)

logger = logging.getLogger(__name__)

# Pre-built neighbour offsets for box-overlap fallback
_NEIGHBOUR_OFFSETS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32)


def _lanczos(x: np.ndarray, a: int = 2, clamp_negative: bool = False) -> np.ndarray:
    """Lanczos kernel: sinc(x) * sinc(x/a) for |x| < a, else 0.

    Vectorised over any shaped ndarray.

    Implementation note
    -------------------
    The zero-distance case (x ≈ 0) is handled by assigning 1.0 directly
    to a separate sub-mask *before* computing the sinc formula on the
    remaining non-zero elements.  This avoids the 0/0 division that
    np.where(cond, formula, 1.0) would silently evaluate for x=0 elements
    (numpy evaluates both branches unconditionally), which previously
    produced RuntimeWarning: invalid value encountered in divide even
    though the final result was correct.

    Parameters
    ----------
    clamp_negative : bool
        If True, zero out the negative sidelobes (kernel_mode=
        'lanczos2_clamped').  Trades the sinc kernel's negative-lobe
        ringing suppression capability away in exchange for a strictly
        positive kernel, which cannot itself produce Gibbs-style
        overshoot/undershoot.  See DRIZZLE_KERNEL_MODE in constants.py.
    """
    out = np.zeros_like(x, dtype=np.float64)

    in_support = np.abs(x) < a          # |x| < a: kernel support
    near_zero  = in_support & (np.abs(x) < 1e-10)   # x ≈ 0 → limit = 1
    nonzero    = in_support & ~near_zero              # normal sinc case

    out[near_zero] = 1.0

    if nonzero.any():
        xv    = x[nonzero]
        pi_x  = np.pi * xv
        out[nonzero] = a * np.sin(pi_x) * np.sin(pi_x / a) / (pi_x ** 2)

    if clamp_negative:
        np.maximum(out, 0.0, out=out)

    return out


def _fill_coverage_holes(
    numerator: np.ndarray,
    denominator: np.ndarray,
    floor_ratio: float = DRIZZLE_COVERAGE_FLOOR_RATIO,
) -> tuple[np.ndarray, np.ndarray]:
    """Safety-net fill for HR pixels with near-zero accumulation weight.

    With Lanczos-2 this should rarely trigger (CV ≈ 0.04, holes ≈ 0%).
    Retained as a safety net for degenerate input (N=1, zero dither).

    Parameters
    ----------
    numerator : ndarray, shape (H, W, 3), float64
    denominator : ndarray, shape (H, W), float64
    floor_ratio : float
        Fraction of median below which a pixel is considered a hole.
        Lowered from 0.35 to 0.15 because Lanczos never approaches
        the old 35% threshold under normal operation.
    """
    if floor_ratio <= 0.0:
        return numerator, denominator

    pos_denom = denominator[denominator > DRIZZLE_WEIGHT_FLOOR]
    if pos_denom.size == 0:
        return numerator, denominator

    median_denom = float(np.median(pos_denom))
    if median_denom <= 0.0:
        return numerator, denominator

    threshold = floor_ratio * median_denom
    hole_mask = denominator < threshold

    if not hole_mask.any():
        return numerator, denominator

    n_holes = int(hole_mask.sum())
    logger.debug(
        "Coverage-hole fill (safety net): %d HR pixels below %.1f%% of median",
        n_holes, floor_ratio * 100,
    )

    denom_smooth = uniform_filter(denominator, size=3, mode="reflect")
    denominator = np.where(hole_mask, denom_smooth, denominator)

    for c in range(numerator.shape[2]):
        num_c_smooth = uniform_filter(numerator[:, :, c], size=3, mode="reflect")
        numerator[:, :, c] = np.where(hole_mask, num_c_smooth, numerator[:, :, c])

    return numerator, denominator


def _drizzle_chunk_lanczos(
    images: list[np.ndarray],
    M_list: list[Optional[np.ndarray]],
    weight_maps: list[np.ndarray],
    scale: float,
    y_start_hr: int,
    y_end_hr: int,
    hr_w: int,
    lanczos_a: int = DRIZZLE_LANCZOS_A,
    clamp_negative: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Lanczos-2 Drizzle accumulation for a single HR chunk.

    For each HR pixel (u, v) in the chunk, computes the weighted sum of
    all LR pixels whose Lanczos footprint covers that HR pixel.  Kernel
    radius = lanczos_a LR pixels = lanczos_a * scale HR pixels.

    Unlike the 4-neighbour box overlap, the Lanczos kernel has no
    structural zeros: every HR pixel always receives positive weight
    from at least one LR pixel, eliminating periodic coverage holes.

    Parameters
    ----------
    images : list of np.ndarray, each BGR uint8
    M_list : list of Optional[np.ndarray], translation matrices
    weight_maps : list of np.ndarray, motion mask weights [0,1]
    scale : float
    y_start_hr, y_end_hr : int
    hr_w : int
    lanczos_a : int, Lanczos order (default 2)

    Returns
    -------
    numerator   : (chunk_h, hr_w, 3) float64
    denominator : (chunk_h, hr_w)    float64
    """
    chunk_h = y_end_hr - y_start_hr
    numerator   = np.zeros((chunk_h, hr_w, 3), dtype=np.float32)
    denominator = np.zeros((chunk_h, hr_w),    dtype=np.float32)

    # HR pixel index grids for this chunk
    u_hr = np.arange(hr_w,    dtype=np.float32)[None, :]      # (1, W)
    v_hr = (np.arange(chunk_h, dtype=np.float32) + y_start_hr)[:, None]  # (H, 1)

    for img, M, wmap in zip(images, M_list, weight_maps):
        if M is None:
            continue

        lr_h, lr_w = img.shape[:2]
        tx = float(M[0, 2])
        ty = float(M[1, 2])

        # For each HR pixel, find the LR pixel neighbourhood
        # x_LR, y_LR: fractional LR coords of each HR pixel
        x_LR = u_hr / scale + tx   # (1, W)
        y_LR = v_hr / scale + ty   # (H, 1)

        # Range of LR pixels to consider (kernel radius = lanczos_a)
        x0_lr = np.floor(x_LR).astype(np.int32)  # (1, W)
        y0_lr = np.floor(y_LR).astype(np.int32)  # (H, 1)

        # Iterate over the (2a)×(2a) LR neighbourhood
        for dy in range(-lanczos_a + 1, lanczos_a + 1):
            yn_lr = np.clip(y0_lr + dy, 0, lr_h - 1)   # (H, 1)
            in_y  = (y0_lr + dy >= 0) & (y0_lr + dy < lr_h)  # (H, 1)
            # LR y-distance in LR units for Lanczos weight
            d_y = y_LR - (yn_lr.astype(np.float32))    # (H, 1)
            w_y = _lanczos(d_y, a=lanczos_a, clamp_negative=clamp_negative)  # (H, 1)

            for dx in range(-lanczos_a + 1, lanczos_a + 1):
                xn_lr = np.clip(x0_lr + dx, 0, lr_w - 1)  # (1, W)
                in_x  = (x0_lr + dx >= 0) & (x0_lr + dx < lr_w)  # (1, W)

                in_bounds = (in_y & in_x).astype(np.float32)  # (H, W)

                d_x = x_LR - (xn_lr.astype(np.float32))    # (1, W)
                w_x = _lanczos(d_x, a=lanczos_a, clamp_negative=clamp_negative)  # (1, W)

                # 2-D separable Lanczos weight
                w_lanczos = w_y * w_x * in_bounds           # (H, W)

                val      = img[yn_lr, xn_lr]                # (H, W, 3) (float32)
                w_motion = wmap[yn_lr, xn_lr]               # (H, W) (float32)

                combined_w = w_lanczos * w_motion           # (H, W)

                numerator   += combined_w[:, :, None] * val
                denominator += combined_w

    return numerator, denominator


def _drizzle_chunk_nearest(
    images: list[np.ndarray],
    M_list: list[Optional[np.ndarray]],
    weight_maps: list[np.ndarray],
    scale: float,
    y_start_hr: int,
    y_end_hr: int,
    hr_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbor Drizzle accumulation for a single HR chunk.

    Projects each HR pixel back to the LR coordinate system using the inverse
    transform, and samples the nearest LR pixel center.

    This maps 100% of the raw sensor high-frequency energy with zero interpolation
    blur, and has completely uniform coverage (no structural grid holes).

    Parameters
    ----------
    images : list of np.ndarray
    M_list : list of Optional[np.ndarray]
    weight_maps : list of np.ndarray
    scale : float
    y_start_hr, y_end_hr : int
    hr_w : int

    Returns
    -------
    numerator   : (chunk_h, hr_w, 3) float64
    denominator : (chunk_h, hr_w)    float64
    """
    chunk_h = y_end_hr - y_start_hr
    numerator   = np.zeros((chunk_h, hr_w, 3), dtype=np.float32)
    denominator = np.zeros((chunk_h, hr_w),    dtype=np.float32)

    for img, M, wmap in zip(images, M_list, weight_maps):
        if M is None:
            continue

        tx = float(M[0, 2])
        ty = float(M[1, 2])

        # M_chunk maps HR (relative to chunk) to LR coords:
        # x_lr = M[0,0]/scale * x_hr + M[0,1]/scale * y_hr + (tx + M[0,1] * y_start / scale)
        # y_lr = M[1,0]/scale * x_hr + M[1,1]/scale * y_hr + (ty + M[1,1] * y_start / scale)
        M_chunk = np.array([
            [M[0, 0] / scale, M[0, 1] / scale, float(M[0, 2]) + M[0, 1] * y_start_hr / scale],
            [M[1, 0] / scale, M[1, 1] / scale, float(M[1, 2]) + M[1, 1] * y_start_hr / scale]
        ], dtype=np.float64)

        # Warp the LR image and weight map to the HR chunk using nearest-neighbor
        warped_img = cv2.warpAffine(
            img, M_chunk, (hr_w, chunk_h),
            flags=cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST
        )

        warped_w = cv2.warpAffine(
            wmap, M_chunk, (hr_w, chunk_h),
            flags=cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST
        )

        numerator   += warped_img * warped_w[:, :, None]
        denominator += warped_w

    return numerator, denominator


def _drizzle_chunk_bilinear(
    images: list[np.ndarray],
    M_list: list[Optional[np.ndarray]],
    weight_maps: list[np.ndarray],
    scale: float,
    y_start_hr: int,
    y_end_hr: int,
    hr_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear Drizzle accumulation for a single HR chunk.

    Projects each HR pixel back to the LR coordinate system and performs
    bilinear interpolation (INTER_LINEAR) of the 4 surrounding LR pixels.

    This completely eliminates nearest-neighbor step grid artifacts and Moire patterns
    while maintaining zero periodic grid coverage holes (denominator remains uniform).

    Parameters
    ----------
    images : list of np.ndarray
    M_list : list of Optional[np.ndarray]
    weight_maps : list of np.ndarray
    scale : float
    y_start_hr, y_end_hr : int
    hr_w : int

    Returns
    -------
    numerator   : (chunk_h, hr_w, 3) float32
    denominator : (chunk_h, hr_w)    float32
    """
    chunk_h = y_end_hr - y_start_hr
    numerator   = np.zeros((chunk_h, hr_w, 3), dtype=np.float32)
    denominator = np.zeros((chunk_h, hr_w),    dtype=np.float32)

    for img, M, wmap in zip(images, M_list, weight_maps):
        if M is None:
            continue

        tx = float(M[0, 2])
        ty = float(M[1, 2])

        M_chunk = np.array([
            [M[0, 0] / scale, M[0, 1] / scale, float(M[0, 2]) + M[0, 1] * y_start_hr / scale],
            [M[1, 0] / scale, M[1, 1] / scale, float(M[1, 2]) + M[1, 1] * y_start_hr / scale]
        ], dtype=np.float64)

        # Warp the LR image and weight map to the HR chunk using bilinear
        warped_img = cv2.warpAffine(
            img, M_chunk, (hr_w, chunk_h),
            flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
        )

        warped_w = cv2.warpAffine(
            wmap, M_chunk, (hr_w, chunk_h),
            flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
        )

        numerator   += warped_img * warped_w[:, :, None]
        denominator += warped_w

    return numerator, denominator


def _drizzle_chunk_lanczos4(
    images: list[np.ndarray],
    M_list: list[Optional[np.ndarray]],
    weight_maps: list[np.ndarray],
    scale: float,
    y_start_hr: int,
    y_end_hr: int,
    hr_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Lanczos-4 Drizzle accumulation for a single HR chunk.

    Projects each HR pixel back to the LR coordinate system and performs
    Lanczos-4 interpolation (INTER_LANCZOS4) over an 8x8 neighborhood.

    This runs in C++ at lightning speed, preserves extreme Nyquist frequencies
    (resolving micro-textures and vents), and eliminates all grid/stepping artifacts.

    Parameters
    ----------
    images : list of np.ndarray
    M_list : list of Optional[np.ndarray]
    weight_maps : list of np.ndarray
    scale : float
    y_start_hr, y_end_hr : int
    hr_w : int

    Returns
    -------
    numerator   : (chunk_h, hr_w, 3) float32
    denominator : (chunk_h, hr_w)    float32
    """
    chunk_h = y_end_hr - y_start_hr
    numerator   = np.zeros((chunk_h, hr_w, 3), dtype=np.float32)
    denominator = np.zeros((chunk_h, hr_w),    dtype=np.float32)

    for img, M, wmap in zip(images, M_list, weight_maps):
        if M is None:
            continue

        tx = float(M[0, 2])
        ty = float(M[1, 2])

        M_chunk = np.array([
            [M[0, 0] / scale, M[0, 1] / scale, float(M[0, 2]) + M[0, 1] * y_start_hr / scale],
            [M[1, 0] / scale, M[1, 1] / scale, float(M[1, 2]) + M[1, 1] * y_start_hr / scale]
        ], dtype=np.float64)

        # Warp the LR image and weight map to the HR chunk using Lanczos-4
        warped_img = cv2.warpAffine(
            img, M_chunk, (hr_w, chunk_h),
            flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4
        )

        warped_w = cv2.warpAffine(
            wmap, M_chunk, (hr_w, chunk_h),
            flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4
        )

        numerator   += warped_img * warped_w[:, :, None]
        denominator += warped_w

    return numerator, denominator


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
    """Box-overlap Drizzle (fallback, kernel_mode='box').

    Retained for diagnostic comparison and as a fallback for edge cases
    where Lanczos negative sidelobes could cause issues (e.g., single
    frame input).
    """
    chunk_h = y_end_hr - y_start_hr
    numerator   = np.zeros((chunk_h, hr_w, 3), dtype=np.float32)
    denominator = np.zeros((chunk_h, hr_w),    dtype=np.float32)

    u_grid = np.arange(hr_w,    dtype=np.float32)[None, :]
    v_local_grid = np.arange(chunk_h, dtype=np.float32)[:, None]
    v_grid = v_local_grid + y_start_hr

    x_h1, x_h2 = u_grid - 0.5, u_grid + 0.5
    y_h1, y_h2 = v_local_grid - 0.5, v_local_grid + 0.5
    r_droplet = 0.5 * pixfrac * scale

    for img, M, wmap in zip(images, M_list, weight_maps):
        if M is None:
            continue

        lr_h, lr_w = img.shape[:2]
        tx, ty = float(M[0, 2]), float(M[1, 2])
        x_LR_prime = u_grid / scale + tx
        y_LR_prime = v_grid  / scale + ty
        x0 = np.floor(x_LR_prime).astype(np.int32)
        y0 = np.floor(y_LR_prime).astype(np.int32)

        for dx, dy in _NEIGHBOUR_OFFSETS:
            xn = np.clip(x0 + dx, 0, lr_w - 1)
            yn = np.clip(y0 + dy, 0, lr_h - 1)
            in_bounds = (
                (x0 + dx >= 0) & (x0 + dx < lr_w) &
                (y0 + dy >= 0) & (y0 + dy < lr_h)
            ).astype(np.float32)
            val      = img[yn, xn]
            w_motion = wmap[yn, xn] * in_bounds
            x_c = scale * (xn.astype(np.float32) - tx)
            y_c = scale * (yn.astype(np.float32) - ty) - y_start_hr
            overlap_x = np.maximum(0.0, np.minimum(x_h2, x_c+r_droplet) - np.maximum(x_h1, x_c-r_droplet))
            overlap_y = np.maximum(0.0, np.minimum(y_h2, y_c+r_droplet) - np.maximum(y_h1, y_c-r_droplet))
            oa = overlap_x * overlap_y * w_motion
            numerator   += oa[:, :, None] * val
            denominator += oa

    return numerator, denominator


def drizzle_stack(
    images: list[np.ndarray],
    M_list: list[Optional[np.ndarray]],
    weight_maps: list[np.ndarray],
    scale: float,
    ref_idx: int = 0,
    config: Optional[OpticoConfig] = None,
    coverage_out: Optional[dict] = None,
) -> np.ndarray:
    """Phase 8: Drizzle Multi-Frame Stacking with Lanczos-2 Kernel.

    Implements Variable-Pixel Linear Reconstruction adapted for handheld
    burst photography.  Uses a Lanczos-2 interpolation kernel (default)
    which provides uniform coverage and zero structural grid artifacts,
    replacing the original box-overlap kernel.

    Parameters
    ----------
    images : list of np.ndarray (BGR, uint8)
    M_list : list of Optional[np.ndarray] — translation matrices
    weight_maps : list of np.ndarray — per-frame motion mask weights
    scale : float — output upscale factor from Pre-flight
    ref_idx : int — reference frame index
    config : OpticoConfig, optional
    coverage_out : dict, optional
        If provided, populated (in place) with the raw (pre-safety-net-fill)
        coverage-denominator diagnostics for this run: 'cv', 'holes_pct',
        'mean', 'median'. Piggybacks on the accumulation this function
        already does instead of a second pass over the same kernel — see
        backend/benchmarks/metrics.py, which needed this because a
        stand-alone second Lanczos-2 accumulation pass roughly doubled
        wall-clock time for the 'lanczos2'/'lanczos2_clamped' benchmarks.
        Existing callers are unaffected (default None, no extra work).

    Returns
    -------
    np.ndarray — high-resolution output (BGR, float32, [0, 255])
    """
    if config is None:
        config = OpticoConfig()

    pixfrac    = config.pixfrac
    num_chunks = config.num_chunks
    kernel_mode = getattr(config, 'kernel_mode', DRIZZLE_KERNEL_MODE)

    if not images:
        raise ValueError("Empty image list")

    lr_h, lr_w = images[0].shape[:2]
    hr_h = int(round(lr_h * scale))
    hr_w = int(round(lr_w * scale))

    logger.info(
        "Drizzle: %d frames, scale=%.2f, pixfrac=%.2f, kernel=%s, "
        "LR=%dx%d -> HR=%dx%d, chunks=%d",
        len(images), scale, pixfrac, kernel_mode, lr_w, lr_h, hr_w, hr_h, num_chunks,
    )

    output = np.zeros((hr_h, hr_w, 3), dtype=np.float32)

    # Pre-convert images and weight maps to float32/2D once to save massive redundant allocations inside chunk loops
    images = [img.astype(np.float32) for img in images]
    weight_maps = [
        (wmap if wmap.ndim == 2 else wmap[:, :, 0]).astype(np.float32)
        for wmap in weight_maps
    ]

    chunk_boundaries = np.linspace(0, hr_h, num_chunks + 1, dtype=int)

    coverage_rows: list[np.ndarray] = [] if coverage_out is not None else None
    coverage_border = int(4 * scale) + 8

    for chunk_idx in range(num_chunks):
        y_start = int(chunk_boundaries[chunk_idx])
        y_end   = int(chunk_boundaries[chunk_idx + 1])
        if y_start >= y_end:
            continue

        logger.debug(
            "Processing chunk %d/%d (rows %d-%d) kernel=%s",
            chunk_idx + 1, num_chunks, y_start, y_end, kernel_mode,
        )

        if kernel_mode == 'box':
            numerator, denominator = _drizzle_chunk_vectorized(
                images, M_list, weight_maps, scale, pixfrac,
                y_start, y_end, hr_w,
            )
        elif kernel_mode == 'nearest':
            numerator, denominator = _drizzle_chunk_nearest(
                images, M_list, weight_maps, scale,
                y_start, y_end, hr_w,
            )
        elif kernel_mode == 'bilinear':
            numerator, denominator = _drizzle_chunk_bilinear(
                images, M_list, weight_maps, scale,
                y_start, y_end, hr_w,
            )
        elif kernel_mode == 'lanczos2_clamped':
            numerator, denominator = _drizzle_chunk_lanczos(
                images, M_list, weight_maps, scale,
                y_start, y_end, hr_w,
                lanczos_a=DRIZZLE_LANCZOS_A,
                clamp_negative=True,
            )
        elif kernel_mode == 'box_supersample':
            # Accumulate with plain 'box' at DRIZZLE_SUPERSAMPLE_FACTOR times
            # the requested scale, one output chunk at a time (never a
            # whole-image supersampled canvas -- see DRIZZLE_SUPERSAMPLE_FACTOR
            # in constants.py for why this matters for memory), then
            # area-decimate back down. The decimation is a positive-only
            # low-pass filter, so it can suppress box's periodic coverage-zero
            # grid artifact (aliased to a frequency far above final Nyquist)
            # without introducing lanczos2's negative-sidelobe ringing.
            ss = DRIZZLE_SUPERSAMPLE_FACTOR
            chunk_h = y_end - y_start
            num_super, den_super = _drizzle_chunk_vectorized(
                images, M_list, weight_maps, scale * ss, pixfrac,
                y_start * ss, y_end * ss, hr_w * ss,
            )
            safe_den_super = np.maximum(den_super, DRIZZLE_WEIGHT_FLOOR)
            img_super = (num_super / safe_den_super[:, :, None]).astype(np.float32)
            del num_super, safe_den_super
            img_down = cv2.resize(img_super, (hr_w, chunk_h), interpolation=cv2.INTER_AREA)
            coverage_denom_chunk = cv2.resize(
                den_super.astype(np.float32), (hr_w, chunk_h), interpolation=cv2.INTER_AREA
            ).astype(np.float64)
            del den_super, img_super
            # numerator/denominator = (already-normalised image, all-ones) so
            # the shared safety-net-fill + division logic below is a no-op:
            # decimation already removed the structural holes that fill
            # exists to patch.
            numerator = img_down.astype(np.float32)
            denominator = np.ones((chunk_h, hr_w), dtype=np.float32)
        elif kernel_mode == 'lanczos4':
            numerator, denominator = _drizzle_chunk_lanczos4(
                images, M_list, weight_maps, scale,
                y_start, y_end, hr_w,
            )
        else:  # 'lanczos2'
            numerator, denominator = _drizzle_chunk_lanczos(
                images, M_list, weight_maps, scale,
                y_start, y_end, hr_w,
                lanczos_a=DRIZZLE_LANCZOS_A,
            )

        # For box_supersample, measure coverage on the *decimated* raw denom
        # (the residual variation that survives anti-aliasing), not the
        # placeholder all-ones `denominator` used for image reconstruction --
        # otherwise this would trivially (and misleadingly) always read cv=0.
        coverage_source = (
            coverage_denom_chunk if kernel_mode == 'box_supersample' else denominator
        )
        if coverage_rows is not None:
            row_lo = max(y_start, coverage_border)
            row_hi = min(y_end, hr_h - coverage_border)
            if row_hi > row_lo:
                coverage_rows.append(
                    coverage_source[row_lo - y_start:row_hi - y_start,
                                     coverage_border:hr_w - coverage_border].copy()
                )

        # Safety-net fill (use higher 0.35 threshold for box/box_supersample to fill grid ripples)
        floor_ratio = 0.35 if kernel_mode in ('box', 'box_supersample') else DRIZZLE_COVERAGE_FLOOR_RATIO
        numerator, denominator = _fill_coverage_holes(
            numerator, denominator,
            floor_ratio=floor_ratio,
        )

        safe_denom = np.maximum(denominator, DRIZZLE_WEIGHT_FLOOR)
        output[y_start:y_end] = (
            numerator / safe_denom[:, :, None]
        ).astype(np.float32)

        del numerator, denominator, safe_denom
        gc.collect()

    np.clip(output, 0.0, 255.0, out=output)
    logger.info("Drizzle stacking complete: output shape %s, kernel=%s", output.shape, kernel_mode)

    if coverage_out is not None:
        if coverage_rows:
            interior = np.concatenate(coverage_rows, axis=0)
            pos = interior[interior > DRIZZLE_WEIGHT_FLOOR]
            if pos.size > 0:
                median_denom = float(np.median(pos))
                mean_denom = float(np.mean(pos))
                std_denom = float(np.std(pos))
                coverage_out["cv"] = std_denom / mean_denom if mean_denom > 0 else 0.0
                coverage_out["holes_pct"] = 100.0 * float(
                    np.mean(interior < DRIZZLE_COVERAGE_FLOOR_RATIO * median_denom)
                )
                coverage_out["mean"] = mean_denom
                coverage_out["median"] = median_denom
            else:
                coverage_out.update(cv=0.0, holes_pct=100.0, mean=0.0, median=0.0)
        else:
            coverage_out.update(cv=0.0, holes_pct=100.0, mean=0.0, median=0.0)

    return output
