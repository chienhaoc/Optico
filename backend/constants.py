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
# Wiener Deconvolution (Phase 9)
# ============================================================
MAD_SCALE_FACTOR: float = 1.4826
"""Scale factor converting MAD to standard deviation for Gaussian noise.
Equals 1 / Phi_inv(3/4) where Phi is the standard normal CDF."""

K_EST_MIN: float = 0.001
"""Retained for backward compatibility / diagnostic logging only.
No longer used to derive the Wiener regularization parameter directly—
see NOISE_FLOOR_HIGH_FREQ_FRACTION below."""

K_EST_MAX: float = 0.08
"""Retained for backward compatibility / diagnostic logging only."""

NOISE_FLOOR_HIGH_FREQ_FRACTION: float = 0.75
"""Lower bound of the radial frequency search range (as a fraction of the
Nyquist radius) used by _find_noise_plateau() to locate the frequency
annulus where the image power spectrum has decayed to its white-noise
floor. The actual cutoff is found adaptively; this is the minimum
frequency considered, not a fixed boundary.
Chosen so the search starts well into the noise-dominated region for
typical camera images while leaving headroom for high-resolution
Drizzle outputs where signal may extend slightly further."""

NOISE_PLATEAU_BINS: int = 20
"""Number of radial frequency bins used by _find_noise_plateau() to
search for the onset of the noise plateau. More bins = finer resolution
but slower; 20 is a good balance for images up to ~8k px."""

NOISE_PLATEAU_GRAD_THRESHOLD: float = 0.05
"""Relative gradient threshold for plateau detection: a bin is considered
part of the flat noise floor when its change in median power is less than
this fraction of the DC-region power. Lower = finds plateau earlier
(more aggressive noise suppression); higher = more conservative."""

MIN_SIGNAL_POWER_FRACTION: float = 0.005
"""Floor on estimated per-frequency signal power, as a fraction of the
noise floor, to avoid division-by-zero / unbounded K at frequencies
where measured power dips below the noise floor (pure noise bins)."""

K_FREQ_MIN: float = 1e-4
"""Lower clip bound for the per-frequency regularization K(f).
Prevents Wiener filter from acting as a pure inverse filter at any
frequency, which would amplify noise catastrophically."""

K_FREQ_MAX: float = 200.0
"""Upper clip bound for the per-frequency regularization K(f).
At K=200 the Wiener response is 1/(1+200) ≈ 0.5%, effectively
suppressing those frequencies to near-zero. Generous rather than tight
because DC-gain preservation (not K clamping) is the operative
brightness-safety mechanism."""

PSF_SIGMA_MIN: float = 0.6
"""Minimum PSF sigma (tightest optical limit)."""

PSF_SIGMA_SCALE: float = 0.4
"""PSF sigma scaling factor relative to upscale factor."""

PSF_TRUNCATION_SIGMAS: float = 3.0
"""Number of sigmas to include in the truncated PSF kernel."""


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
