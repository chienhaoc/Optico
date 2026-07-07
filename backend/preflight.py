"""Optico MFSR Engine — Phase 7: Pre-flight Scale Bounding.

Calculates the Safe Scale Cap based on alignment quality metrics,
using Spatial Sampling Theorem (Nyquist) and Cramer-Rao Lower Bound.
This prevents Alignment Drift Blur by dynamically restricting the
maximum upscale factor.
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
    dither_quality: float = 1.0,
    optical_decay: float = OPTICAL_DECAY_CONSTANT,
) -> float:
    """Calculate the maximum safe upscale factor.

    Based on two independent physical limits:
    1. Density Limit (Nyquist): max resolution based on total valid spatial samples (retained ratio).
    2. Blur Limit (CRLB): max resolution constrained by sub-pixel dither quality (phase coverage).

    Parameters
    ----------
    num_frames : int
        Number of valid frames in the burst stack.
    retained_ratio : float
        Global Retained Pixel Ratio in [0, 1].
    dither_quality : float
        Sub-pixel dither quality in [0, 1]. 1.0 = perfectly uniform phase.
    optical_decay : float
        Optical decay constant mapping SNR to spatial scale.

    Returns
    -------
    float
        Safe scale cap, clamped to [MIN_SCALE, MAX_SCALE].
    """
    if num_frames < 1:
        return MIN_SCALE

    # Clamp inputs to avoid singularities
    R = max(retained_ratio, MIN_RETAINED_RATIO)
    R = min(R, 1.0 - MIN_RETAINED_RATIO)

    Q = max(dither_quality, MIN_RETAINED_RATIO)
    Q = min(Q, 1.0 - MIN_RETAINED_RATIO)

    # 1. Density limit: theoretical max from N frames with R retention (Phase 6)
    density_limit = math.sqrt(num_frames * R)

    # 2. Blur limit (CRLB): alignment phase coverage limits achievable resolution (Phase 5)
    blur_limit = optical_decay * math.sqrt(Q / (1.0 - Q))

    safe_cap = min(density_limit, blur_limit)
    safe_cap = max(safe_cap, MIN_SCALE)
    safe_cap = min(safe_cap, MAX_SCALE)

    logger.info(
        "Pre-flight: N=%d, R_global=%.3f, Q_dither=%.3f -> density_limit=%.2f, "
        "blur_limit=%.2f -> safe_cap=%.2f",
        num_frames, R, Q, density_limit, blur_limit, safe_cap,
    )
    return safe_cap


def resolve_final_scale(
    target_scale: float,
    num_frames: int,
    retained_ratio: float,
    dither_quality: float = 1.0,
    cc_scores: Optional[list[float]] = None,
    config: Optional[OpticoConfig] = None,
) -> tuple[float, float]:
    """Resolve the final effective scale factor.

    Applies Pre-flight bounding to clamp the user's target scale
    to the physically safe maximum.

    Parameters
    ----------
    target_scale : float
        User-requested upscale factor.
    num_frames : int
        Number of valid frames.
    retained_ratio : float
        From dynamic masking or CC approximation.
    dither_quality: float
        Dither quality in [0, 1].
    cc_scores : list of float, optional
        Per-frame correlation coefficients.
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

    # If retained_ratio is near zero but we have CC scores, approximate
    if retained_ratio < MIN_RETAINED_RATIO and cc_scores:
        valid_cc = [c for c in cc_scores if c > 0]
        if valid_cc:
            retained_ratio = float(np.mean(valid_cc))
            logger.info(
                "Using mean CC (%.4f) as retained ratio proxy",
                retained_ratio,
            )

    safe_cap = calculate_safe_scale_cap(
        num_frames, retained_ratio, dither_quality, config.optical_decay
    )

    final_scale = min(target_scale, safe_cap)
    final_scale = max(final_scale, MIN_SCALE)

    if final_scale < target_scale:
        logger.warning(
            "Target scale %.2f clamped to %.2f by Pre-flight "
            "(safe_cap=%.2f)",
            target_scale, final_scale, safe_cap,
        )
    else:
        logger.info(
            "Target scale %.2f within safe bounds (cap=%.2f)",
            target_scale, safe_cap,
        )

    return final_scale, safe_cap
