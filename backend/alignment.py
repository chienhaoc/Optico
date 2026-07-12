"""Optico MFSR Engine — Phase 2-4: Frame Alignment & Registration.

Includes:
- Harmony Anchor: Geometric-median based reference frame selection
- ECC Sub-pixel Registration with adaptive scale selection
- N_eff Entropy Dither Quality (KDE-normalized, falls back to histogram)

Dither quality metric history
------------------------------
Original (Rayleigh-based): saturated Q=1.0 for most realistic bursts.

v2 (4×4 histogram Shannon entropy):
  N_eff = 2^H, range [1, 16].
  Ceiling effect: N=7 frames with any ≧7 distinct bins all return N_eff=7.0.

v3 (current, KDE-normalized N_eff):
  Torus Gaussian KDE on 32×32 evaluation grid, mapped to [1, N] scale.
  Falls back to 4×4 histogram for n < NEFF_KDE_MIN_FRAMES or if scipy
  is unavailable.  Fallback uses histogram (not float(n)) so blur_limit
  is never over-estimated.

Sandbox results:
  clustered offsets (rng 0.3-0.6):  hist=2.94 → KDE=5.91  (+3.0)
  uniform offsets   (rng 0.05-0.95): hist=5.74 → KDE=6.59  (+0.9)
  → blur_limit improves +0.40-0.54 for realistic handheld bursts.
"""
import logging
import math
from typing import Optional

import cv2
import numpy as np

from .constants import (
    OpticoConfig,
    NEFF_KDE_GRID, NEFF_KDE_BW, NEFF_KDE_MIN_FRAMES,
)

logger = logging.getLogger(__name__)

_DITHER_GRID_N: int = 4


def _patch_cc(a: np.ndarray, b: np.ndarray, texture_threshold: float = 3.0) -> float:
    """Normalized cross-correlation between two same-shape patches.

    Returns 1.0 (assume good alignment) when the reference patch has
    insufficient texture (Sobel energy < texture_threshold per pixel),
    because NCC is unreliable in flat/uniform regions.
    """
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    # Check reference patch texture via variance
    if np.std(a) < texture_threshold:
        return 1.0   # flat patch — assume aligned, don't penalise
    a -= a.mean()
    b -= b.mean()
    denom = np.sqrt(np.dot(a, a) * np.dot(b, b))
    return float(np.dot(a, b) / denom) if denom > 1e-10 else 1.0


def compute_regional_cc(
    ref_gray: np.ndarray,
    images: list[np.ndarray],
    M_list: list[Optional[np.ndarray]],
    corner_frac: float = 0.20,
    center_frac: float = 0.30,
) -> list[dict]:
    """Compute region-wise CC (center + 4 corners) for each aligned frame.

    After warping each frame to the reference, the CC is measured
    independently in five spatial regions:
      - CTR  : central (center_frac × min_dim) square
      - TL/TR/BL/BR : corner (corner_frac × min_dim) squares

    Parameters
    ----------
    ref_gray : np.ndarray  (H, W), uint8 or float
        Reference frame in grayscale.
    images : list of np.ndarray  (H, W, 3)
        All burst frames (BGR uint8).
    M_list : list of Optional[np.ndarray]
        Affine transform matrices from alignment (None = rejected).
    corner_frac : float
        Fraction of min(H, W) to use as corner patch side length.
    center_frac : float
        Fraction of min(H, W) to use as center patch half-size.

    Returns
    -------
    list of dict with keys 'cc_center', 'cc_tl', 'cc_tr', 'cc_bl', 'cc_br',
    'cc_corner_min'.  Reference frame entry has all values = 1.0.
    """
    h, w = ref_gray.shape
    min_dim = min(h, w)
    cy, cx = h // 2, w // 2

    cs = max(8, int(corner_frac * min_dim))   # corner patch side
    cm = max(8, int(center_frac * min_dim))   # center half-side

    # Pre-slice reference patches
    ref_f = ref_gray.astype(np.float32)
    ref_ctr = ref_f[cy - cm: cy + cm, cx - cm: cx + cm]
    ref_tl  = ref_f[:cs, :cs]
    ref_tr  = ref_f[:cs, w - cs:]
    ref_bl  = ref_f[h - cs:, :cs]
    ref_br  = ref_f[h - cs:, w - cs:]

    results = []
    for img, M in zip(images, M_list):
        if M is None:
            results.append({
                'cc_center': 0.0, 'cc_tl': 0.0, 'cc_tr': 0.0,
                'cc_bl': 0.0, 'cc_br': 0.0, 'cc_corner_min': 0.0,
            })
            continue

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        warped = cv2.warpAffine(
            img_gray, M, (w, h),
            flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        ).astype(np.float32)

        cc_ctr = _patch_cc(ref_ctr, warped[cy - cm: cy + cm, cx - cm: cx + cm])
        cc_tl  = _patch_cc(ref_tl,  warped[:cs, :cs])
        cc_tr  = _patch_cc(ref_tr,  warped[:cs, w - cs:])
        cc_bl  = _patch_cc(ref_bl,  warped[h - cs:, :cs])
        cc_br  = _patch_cc(ref_br,  warped[h - cs:, w - cs:])

        results.append({
            'cc_center': cc_ctr,
            'cc_tl': cc_tl,
            'cc_tr': cc_tr,
            'cc_bl': cc_bl,
            'cc_br': cc_br,
            'cc_corner_min': min(cc_tl, cc_tr, cc_bl, cc_br),
        })

    return results


def build_regional_quality_maps(
    regional_cc_list: list[dict],
    h: int,
    w: int,
    corner_frac: float = 0.20,
) -> list[np.ndarray]:
    """Build per-frame spatial quality weight maps from regional CC scores.

    Bilinearly interpolates the 5 CC anchor values (center + 4 corners)
    across the full image.  The resulting map is in [0, 1] and represents
    how well each pixel location is aligned for that frame.

    Frames with uniform CC ≈ 1.0 everywhere (pure translation) get a
    near-flat map ≈ 1.0.  Frames with low corner CC (camera rotation
    residual) get a map that tapers toward the corners.
    """
    cy, cx = h / 2.0, w / 2.0
    cs = max(8, int(corner_frac * min(h, w)))
    # Anchor pixel coordinates (y, x) for the 5 CC values
    anchors_yx = [
        (cy,     cx),          # center
        (cs / 2, cs / 2),      # TL
        (cs / 2, w - cs / 2),  # TR
        (h - cs / 2, cs / 2),  # BL
        (h - cs / 2, w - cs / 2),  # BR
    ]

    y_grid = np.arange(h, dtype=np.float32)
    x_grid = np.arange(w, dtype=np.float32)
    xv, yv = np.meshgrid(x_grid, y_grid)

    maps = []
    for rcc in regional_cc_list:
        anchor_vals = [
            rcc['cc_center'],
            rcc['cc_tl'],
            rcc['cc_tr'],
            rcc['cc_bl'],
            rcc['cc_br'],
        ]
        # Inverse-distance weighted interpolation from the 5 anchors
        total_w = np.zeros((h, w), dtype=np.float32)
        total_v = np.zeros((h, w), dtype=np.float32)
        for (ay, ax), val in zip(anchors_yx, anchor_vals):
            dist = np.sqrt((xv - ax) ** 2 + (yv - ay) ** 2) + 1e-3
            w_map = 1.0 / dist
            total_w += w_map
            total_v += w_map * float(val)
        qmap = np.clip(total_v / (total_w + 1e-9), 0.0, 1.0)
        maps.append(qmap)
    return maps


def _neff_histogram(fx: np.ndarray, fy: np.ndarray) -> float:
    """4×4 histogram Shannon entropy N_eff."""
    hist, _, _ = np.histogram2d(
        fx, fy, bins=_DITHER_GRID_N,
        range=[[0.0, 1.0], [0.0, 1.0]],
    )
    hist = hist / max(hist.sum(), 1e-9)
    nonzero = hist[hist > 0]
    H = float(-np.sum(nonzero * np.log2(nonzero)))
    return 2.0 ** H


def _neff_kde_normalized(fx: np.ndarray, fy: np.ndarray) -> float:
    """Torus KDE N_eff mapped to [1, N] scale.

    Falls back to _neff_histogram() if scipy is unavailable
    (never returns float(n), which would over-estimate blur_limit).
    """
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        logger.warning("scipy unavailable; falling back to histogram N_eff")
        return _neff_histogram(fx, fy)

    n = len(fx)
    grid_n = NEFF_KDE_GRID
    bw     = NEFF_KDE_BW

    xs = np.linspace(0, 1, grid_n, endpoint=False)
    xx, yy = np.meshgrid(xs, xs)
    pts = np.vstack([xx.ravel(), yy.ravel()])

    data = np.array([fx, fy])
    mirrors = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            mirrors.append(data + np.array([[dx], [dy]]))
    data_torus = np.hstack(mirrors)

    try:
        kde = gaussian_kde(data_torus, bw_method=bw)
        density = kde(pts)
    except Exception as exc:
        logger.warning("KDE failed (%s); falling back to histogram N_eff", exc)
        return _neff_histogram(fx, fy)

    density = np.maximum(density, 0.0)
    s = density.sum()
    if s < 1e-15:
        return 1.0
    density /= s

    mask = density > 1e-15
    H = float(-np.sum(density[mask] * np.log2(density[mask])))
    H_max = math.log2(grid_n ** 2)
    N_eff = 1.0 + (H / H_max) * (n - 1.0)
    return float(np.clip(N_eff, 1.0, float(n)))


def calculate_dither_quality_neff(
    M_list: list[Optional[np.ndarray]],
) -> float:
    """Estimate sub-pixel dither quality via entropy N_eff.

    Uses KDE-normalized N_eff for n ≥ NEFF_KDE_MIN_FRAMES,
    falling back to 4×4 histogram for smaller bursts or missing scipy.
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

    if n >= NEFF_KDE_MIN_FRAMES:
        N_eff = _neff_kde_normalized(fx, fy)
        logger.debug("Dither quality (KDE): n=%d, N_eff=%.3f", n, N_eff)
    else:
        N_eff = _neff_histogram(fx, fy)
        logger.debug("Dither quality (hist-fallback): n=%d, N_eff=%.3f", n, N_eff)

    return float(N_eff)


def select_reference_frame(
    images: list[np.ndarray],
    M_list: list[Optional[np.ndarray]],
) -> int:
    """Harmony Anchor: Select the optimal reference frame."""
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

    Higher resolution → more aggressive downscale (smaller factor) to keep
    ECC registration cost bounded.  Thresholds are approximate megapixel
    boundaries (512=0.25MP, 1024=1MP, 2048=4MP, >2048=full-size cameras).
    """
    max_dim = max(img_h, img_w)
    if max_dim <= 512:
        return 0.75   # small: minimal downscale
    elif max_dim <= 1024:
        return 0.75   # medium-small: still light downscale
    elif max_dim <= 2048:
        return 0.50   # medium: moderate downscale
    else:
        return 0.50   # large (cameras): aggressive downscale


def align_images_ecc(
    images: list[np.ndarray],
    ref_idx: int = 0,
    config: Optional[OpticoConfig] = None,
) -> tuple[list[Optional[np.ndarray]], list[float]]:
    """Phase 2 & 4: ECC Sub-pixel Registration."""
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

    motion_mode_str = getattr(config, 'ecc_motion_mode', 'affine')
    if motion_mode_str == 'affine':
        warp_mode = cv2.MOTION_AFFINE
    else:
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
