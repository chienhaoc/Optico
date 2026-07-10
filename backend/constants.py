"""Optico MFSR Engine — Named Constants & Configuration.

All magic numbers and empirical parameters are centralized here
for transparency, tuning, and documentation.

Cross-phase dependency map
--------------------------
JPEG source detection (pipeline.py Phase 0)
  → Phase 2 alignment: JPEG_ECC_GAUSS_FILT_SIZE (suppresses DCT block edges)
  → Phase 8 drizzle:   DRIZZLE_KERNEL_MODE (Lanczos-2 default)
                       DRIZZLE_COVERAGE_FLOOR_RATIO (safety-net fill)
  → Phase 9 deconv:    JPEG_PSF_SCALE_FACTOR, JPEG_NOISE_FLOOR_HIGH_FREQ_FRACTION

Drizzle kernel evolution
------------------------
  v1 box-overlap:         CV=0.133, holes=0.39%  (global grid artifacts)
  v2 box + hole-fill:     face grid gone, fine grid remained (CV unchanged)
  v3 Lanczos-2 (current): CV=0.041, holes=0.00%  (-69% CV, -100% holes)
"""
from dataclasses import dataclass, field
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
"""Gaussian filter kernel size for ECC input smoothing (RAW / PNG input)."""

JPEG_ECC_GAUSS_FILT_SIZE: int = 7
"""Gaussian filter kernel size for JPEG input.

JPEG 8×8 DCT blocks introduce high-frequency inter-block discontinuities
that ECC can lock onto as false sub-pixel offsets.  A larger smoothing
kernel (7 vs 5) suppresses these block-edge artifacts so ECC converges
to the true optical displacement rather than the DCT quantisation grid.

Relation to Phase 8 / 9: cleaner sub-pixel offsets directly improve
dither N_eff (Phase 5) and therefore the preflight blur_limit (Phase 7),
which in turn feeds back to PSF sigma in deconv (Phase 9).
"""

# ============================================================
# Dynamic Masking (Phase 6)
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
# Pre-flight Scale Bounding (Phase 7)
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
# Drizzle Stacking (Phase 8)
# ============================================================
DEFAULT_PIXFRAC: float = 0.7
"""Pixel fraction (droplet shrink factor) for Drizzle.
Used by box-overlap kernel only; Lanczos-2 kernel determines its own
effective footprint from the kernel radius (DRIZZLE_LANCZOS_A)."""

DEFAULT_NUM_CHUNKS: int = 8
"""Number of horizontal strips for memory chunking."""

DRIZZLE_WEIGHT_FLOOR: float = 1e-6
"""Minimum weight to avoid division by zero in Drizzle normalization."""

DRIZZLE_KERNEL_MODE: str = 'lanczos2'
"""Default Drizzle interpolation kernel.

'lanczos2' (recommended): Lanczos-2 kernel, radius=2 LR pixels.
  Uniform coverage (CV=0.041), zero structural holes.
  Preserves high-frequency detail better than Gaussian.
  Sandbox benchmark (scale=1.63, pixfrac=0.75, N=7, clustered offsets):
    CV=0.041, holes=0.00%  vs  box: CV=0.133, holes=0.39%
    (-69.3% CV, -100% holes)

'box': original box-overlap 4-neighbour backward drizzle.
  Faster (~4× fewer kernel evaluations) but produces periodic grid
  artifacts when sub-pixel offsets are clustered.
  Use only for diagnostic comparison.
"""

DRIZZLE_LANCZOS_A: int = 2
"""Lanczos kernel order (radius in LR pixel units).
Lanczos-2 (a=2) covers a 4×4 LR neighbourhood per HR pixel per frame.
Lanczos-3 (a=3) gives slightly better frequency response but 9×4=36
LR lookups per HR pixel vs 16 for a=2 — not worth the cost."""

DRIZZLE_COVERAGE_FLOOR_RATIO: float = 0.15
"""Safety-net coverage-hole fill threshold.

With Lanczos-2 the fill almost never triggers (holes=0.00% in sandbox).
Threshold lowered from 0.35 to 0.15 so the fill only activates for
genuinely degenerate inputs (N=1, zero-dither, or masked-out frames).

Set to 0.0 to disable entirely.
"""

# ============================================================
# Wiener Deconvolution (Phase 9)
# ============================================================
MAD_SCALE_FACTOR: float = 1.4826
"""Scale factor converting MAD to standard deviation for Gaussian noise."""

K_EST_MIN: float = 0.001
"""Retained for backward compatibility / diagnostic logging only."""

K_EST_MAX: float = 0.08
"""Retained for backward compatibility / diagnostic logging only."""

NOISE_FLOOR_HIGH_FREQ_FRACTION: float = 0.75
"""Lower bound of the radial frequency search range for RAW / PNG input."""

JPEG_NOISE_FLOOR_HIGH_FREQ_FRACTION: float = 0.60
"""Lower bound of the noise-plateau radial frequency scan for JPEG input.

JPEG DCT quantisation creates a hard spectral cutoff at approximately
0.55–0.65 × Nyquist (quality-dependent).  Above that cutoff the power
spectrum is dominated by quantisation noise rather than image signal,
so it looks identical to a white-noise plateau.  If the plateau scan
starts at the default 0.75 × Nyquist, _find_noise_plateau() anchors
its noise floor estimate inside this JPEG-artifact band, inflating N_floor
by 2–5× and consequently over-regularising K(f) across the entire
spectrum — manifesting as loss of sharpness and texture detail.

By starting the scan at 0.60 × Nyquist (just above the typical JPEG
cutoff), the detector finds the genuine white-noise floor and produces
a conservative, accurate K(f) map that preserves high-frequency detail.

Cross-phase note: this constant works in tandem with JPEG_PSF_SCALE_FACTOR.
"""

JPEG_PSF_SCALE_FACTOR: float = 1.35
"""Multiplicative factor applied to psf_sigma when the input is JPEG.

Physical basis: JPEG quantisation adds a blur PSF (sigma ≈ 0.3–0.5 px)
on top of the optical PSF.  The composite effective PSF sigma is:
  sigma_eff = sqrt(sigma_optical^2 + sigma_jpeg^2) ≈ sigma_optical * 1.35
"""

NOISE_PLATEAU_BINS: int = 20
"""Number of radial frequency bins for _find_noise_plateau()."""

NOISE_PLATEAU_GRAD_THRESHOLD: float = 0.05
"""Relative gradient threshold for noise plateau detection."""

MIN_SIGNAL_POWER_FRACTION: float = 0.005
"""Floor on estimated per-frequency signal power (fraction of noise floor)."""

K_FREQ_MIN: float = 1e-4
"""Lower clip bound for per-frequency regularization K(f)."""

K_FREQ_MAX: float = 200.0
"""Upper clip bound for per-frequency regularization K(f)."""

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

    JPEG vs RAW auto-detection
    --------------------------
    When jpeg_input is None (default), pipeline.py Phase 0 auto-detects
    the source format by inspecting file headers.  Set jpeg_input=True
    or jpeg_input=False to override the detection result.
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
    kernel_mode: str = DRIZZLE_KERNEL_MODE

    # Deconvolution
    psf_override: Optional[float] = None
    skip_deconv: bool = False

    # Input source (auto-detected by pipeline.py; override if needed)
    jpeg_input: Optional[bool] = None
