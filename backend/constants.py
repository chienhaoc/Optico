"""Optico MFSR Engine — Named Constants & Configuration.

All magic numbers and empirical parameters are centralized here
for transparency, tuning, and documentation.
"""
from dataclasses import dataclass
from typing import Optional


# ============================================================
# Alignment (Phase 2)
# ============================================================
DEFAULT_ALIGN_SCALE: float = 0.5
"""Downscale factor for ECC alignment. Lower = faster but less precise."""

DEFAULT_MAX_OFFSET: float = 20.0
"""Maximum allowed translation offset (pixels) before rejecting a frame."""

ECC_MAX_ITERATIONS: int = 200
"""Maximum iterations for ECC convergence."""

ECC_EPSILON: float = 1e-6
"""Convergence threshold for ECC."""

ECC_GAUSS_FILT_SIZE: int = 5
"""Gaussian filter kernel size for ECC input smoothing."""

# ============================================================
# Dynamic Masking (Phase 3)
# ============================================================
NOISE_MODEL_GAIN: float = 0.5
"""Poisson-Gaussian noise model gain coefficient (a in sigma = sqrt(aI + b))."""

NOISE_MODEL_OFFSET: float = 1.0
"""Poisson-Gaussian noise model offset (b in sigma = sqrt(aI + b))."""

GRADIENT_WEIGHT: float = 0.3
"""Weight of gradient magnitude in the normalization denominator."""

BG_MOTION_THRESHOLD: float = 1.5
"""Normalized difference threshold for background motion detection."""

SUBJ_MOTION_THRESHOLD: float = 3.0
"""Normalized difference threshold for subject/foreground motion detection."""

BG_KERNEL_SIZE: int = 7
"""Morphological kernel diameter for background motion dilation."""

SUBJ_KERNEL_SIZE: int = 11
"""Morphological kernel diameter for subject motion dilation."""

SUBJ_DILATE_ITERATIONS: int = 2
"""Dilation iterations for subject motion mask."""

MASK_BLUR_KSIZE: int = 5
"""Gaussian blur kernel size for mask edge smoothing."""

# ============================================================
# Pre-flight Scale Bounding (Phase 2b)
# ============================================================
OPTICAL_DECAY_CONSTANT: float = 0.75
"""Maps dimensionless alignment SNR to spatial scale factor.
Derived from Cramer-Rao Lower Bound analysis."""

MIN_SCALE: float = 1.0
"""Minimum output scale factor (no downscaling)."""

MAX_SCALE: float = 4.0
"""Absolute maximum scale factor regardless of alignment quality."""

MIN_RETAINED_RATIO: float = 0.05
"""Floor for retained ratio to avoid division by zero in blur_limit."""

# ============================================================
# Drizzle Stacking (Phase 4)
# ============================================================
DEFAULT_PIXFRAC: float = 0.7
"""Pixel fraction (droplet shrink factor) for Drizzle.
0.0 = point sampling, 1.0 = full pixel, 0.7 = typical good balance."""

DEFAULT_NUM_CHUNKS: int = 8
"""Number of horizontal strips for memory chunking."""

DRIZZLE_WEIGHT_FLOOR: float = 1e-6
"""Minimum weight to avoid division by zero in Drizzle normalization."""

# ============================================================
# Wiener Deconvolution (Phase 5)
# ============================================================
MAD_SCALE_FACTOR: float = 1.4826
"""Scale factor converting MAD to standard deviation for Gaussian noise.
Equals 1 / Phi_inv(3/4) where Phi is the standard normal CDF."""

K_EST_MIN: float = 0.001
"""Minimum Wiener regularization parameter."""

K_EST_MAX: float = 0.08
"""Maximum Wiener regularization parameter."""

K_STRONG_MULTIPLIER: float = 2.0
"""Multiplier for flat-region (aggressive) regularization."""

K_WEAK_MULTIPLIER: float = 6.0
"""Multiplier for edge-region (conservative) regularization."""

K_STRONG_FLOOR: float = 0.01
"""Minimum K for flat regions."""

K_WEAK_FLOOR: float = 0.03
"""Minimum K for edge regions."""

PSF_SIGMA_MIN: float = 0.6
"""Minimum PSF sigma (tightest optical limit)."""

PSF_SIGMA_SCALE: float = 0.4
"""PSF sigma scaling factor relative to upscale factor."""

PSF_TRUNCATION_SIGMAS: float = 3.0
"""Number of sigmas to include in the truncated PSF kernel."""

EDGE_CANNY_LOW: int = 50
"""Canny edge detector low threshold for dual-band blending mask."""

EDGE_CANNY_HIGH: int = 150
"""Canny edge detector high threshold for dual-band blending mask."""

EDGE_MASK_BLUR_KSIZE: int = 7
"""Gaussian blur kernel size for edge mask smoothing in dual-band blend."""


@dataclass
class OpticoConfig:
    """Runtime configuration for the Optico pipeline.

    All parameters have sensible defaults based on optical physics.
    Override individual fields to tune behavior for specific scenes.
    """
    # Alignment
    align_scale: float = DEFAULT_ALIGN_SCALE
    max_offset: float = DEFAULT_MAX_OFFSET
    ecc_iterations: int = ECC_MAX_ITERATIONS
    ecc_epsilon: float = ECC_EPSILON
    ecc_gauss_filt_size: int = ECC_GAUSS_FILT_SIZE

    # Masking
    noise_gain: float = NOISE_MODEL_GAIN
    noise_offset: float = NOISE_MODEL_OFFSET
    gradient_weight: float = GRADIENT_WEIGHT
    bg_threshold: float = BG_MOTION_THRESHOLD
    subj_threshold: float = SUBJ_MOTION_THRESHOLD

    # Pre-flight
    target_scale: float = 2.0
    optical_decay: float = OPTICAL_DECAY_CONSTANT

    # Drizzle
    pixfrac: float = DEFAULT_PIXFRAC
    num_chunks: int = DEFAULT_NUM_CHUNKS

    # Deconvolution
    psf_override: Optional[float] = None
    skip_deconv: bool = False
