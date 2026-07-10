"""Optico MFSR Engine — Phase 7: Pre-flight Scale Bounding.

Calculates the Safe Scale Cap based on alignment quality metrics.
Prevents Alignment Drift Blur by dynamically restricting the maximum
upscale factor to what the data physically supports.

Formula revision (2025-07 audit)
----------------------------------
Old blur_limit formula:
    blur_limit = decay * sqrt(Q / (1 - Q))
where Q was the Rayleigh-based dither quality in (0, 1).

This had two problems:
1. Q was systematically biased to ~1.0 (see alignment.py docstring),
   making blur_limit effectively infinite for all realistic bursts.
2. The formula had no clear physical interpretation and was numerically
   undefined at Q=1 (the most common output).

New blur_limit formula:
    blur_limit = decay * sqrt(N_eff)
where N_eff = 2^H is the entropy-derived effective independent sub-pixel
position count returned by calculate_dither_quality_neff() in alignment.py.

Physical basis: Nyquist-Shannon sampling theorem on the sub-pixel grid.
N_eff independent positions on a [0,1)² grid support recovering at most
sqrt(N_eff) times the original resolution. The decay constant (0.75) is
a conservative optical efficiency factor.
"""
import logging
import math
from typing import Optional

import numpy as np

from .constants import (
    OpticoConfig,
    OPTICAL_DECAY_CONSTANT,
    MIN_SCALE, MAX_SCALE, MIN_RETAINED_RATIO,
)

logger = logging.getLogger(__name__)


def calculate_safe_scale_cap(
    num_frames: int,
    retained_ratio: float,
    dither_neff: float = 1.0,
    optical_decay: float = OPTICAL_DECAY_CONSTANT,
) -> float:
    """Calculate the maximum safe upscale factor.

    Based on two independent physical limits:

    1. Density Limit (Nyquist sampling):
       sqrt(num_frames * retained_ratio)
       Reflects how many independent spatial samples exist after masking.

    2. Blur Limit (sub-pixel coverage):
       decay * sqrt(N_eff)
       where N_eff is the effective number of independent sub-pixel
       positions as computed by calculate_dither_quality_neff().
       Reflects whether the burst actually covers sub-pixel space
       well enough to reconstruct high frequencies.

    The safe cap is min(density_limit, blur_limit), clamped to
    [MIN_SCALE, MAX_SCALE].

    Parameters
    ----------
    num_frames : int
        Number of valid frames in the burst stack.
    retained_ratio : float
        Global Retained Pixel Ratio in [0, 1].
    dither_neff : float
        Effective independent sub-pixel position count from
        calculate_dither_quality_neff(). Range: [1, _DITHER_GRID_N²].
    optical_decay : float
        Conservative optical efficiency factor (default 0.75).

    Returns
    -------
    float
        Safe scale cap, clamped to [MIN_SCALE, MAX_SCALE].
    """
    if num_frames < 1:
        return MIN_SCALE

    R = float(np.clip(retained_ratio, MIN_RETAINED_RATIO, 1.0 - MIN_RETAINED_RATIO))
    N_eff = max(dither_neff, 1.0)

    density_limit = math.sqrt(num_frames * R)
    blur_limit = optical_decay * math.sqrt(N_eff)

    safe_cap = float(np.clip(min(density_limit, blur_limit), MIN_SCALE, MAX_SCALE))

    logger.info(
        "Pre-flight: N=%d, R=%.3f, N_eff=%.2f -> "
        "density_limit=%.2f, blur_limit=%.2f -> safe_cap=%.2f",
        num_frames, R, N_eff, density_limit, blur_limit, safe_cap,
    )
    return safe_cap


def resolve_final_scale(
    target_scale: float,
    num_frames: int,
    retained_ratio: float,
    dither_neff: float = 1.0,
    cc_scores: Optional[list[float]] = None,
    config: Optional[OpticoConfig] = None,
) -> tuple[float, float]:
    """Resolve the final effective scale factor.

    Applies Pre-flight bounding to clamp the user's target scale to the
    physically safe maximum.

    Parameters
    ----------
    target_scale : float
        User-requested upscale factor.
    num_frames : int
        Number of valid frames.
    retained_ratio : float
        From dynamic masking or CC approximation.
    dither_neff : float
        N_eff from calculate_dither_quality_neff() in alignment.py.
    cc_scores : list of float, optional
        Per-frame correlation coefficients (fallback for retained_ratio).
    config : OpticoConfig, optional
        Configuration parameters.

    Returns
    -------
    final_scale : float
        The clamped output scale factor.
    safe_cap : float
        The calculated safe scale cap (for diagnostics).
    """
    if config is None:
        config = OpticoConfig()

    if retained_ratio < MIN_RETAINED_RATIO and cc_scores:
        valid_cc = [c for c in cc_scores if c > 0]
        if valid_cc:
            retained_ratio = float(np.mean(valid_cc))
            logger.info(
                "Using mean CC (%.4f) as retained ratio proxy", retained_ratio
            )

    safe_cap = calculate_safe_scale_cap(
        num_frames, retained_ratio, dither_neff, config.optical_decay
    )

    final_scale = float(np.clip(min(target_scale, safe_cap), MIN_SCALE, MAX_SCALE))

    if final_scale < target_scale:
        logger.warning(
            "Target scale %.2f clamped to %.2f (safe_cap=%.2f)",
            target_scale, final_scale, safe_cap,
        )
    else:
        logger.info(
            "Target scale %.2f within safe bounds (cap=%.2f)",
            target_scale, safe_cap,
        )

    return final_scale, safe_cap
