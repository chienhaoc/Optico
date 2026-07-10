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

blur_limit evolution
--------------------
  v1 decay=0.75 fixed + 4×4 histogram N_eff:
     blur_limit ≈ 1.80 for typical 7-frame burst
  v2 adaptive decay (CC-based) + KDE N_eff:
     blur_limit ≈ 2.19 at CC=0.90  (+22%, within density_limit=2.51)
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
"""Gaussian filter kernel size for JPEG input."""

# ============================================================
# Dynamic Masking (Phase 6)
# ============================================================
NOISE_MODEL_GAIN: float = 0.5
NOISE_MODEL_OFFSET: float = 1.0
GRADIENT_WEIGHT: float = 0.3
BG_MOTION_THRESHOLD: float = 1.5
SUBJ_MOTION_THRESHOLD: float = 3.0
BG_KERNEL_SIZE: int = 7
SUBJ_KERNEL_SIZE: int = 11
SUBJ_DILATE_ITERATIONS: int = 2
MASK_BLUR_KSIZE: int = 5

# ============================================================
# Pre-flight Scale Bounding (Phase 7)
# ============================================================
OPTICAL_DECAY_CONSTANT: float = 0.75
"""Conservative optical efficiency factor (lower bound / fallback).

Physical basis: sub-pixel offset uncertainty budget from ECC alignment.
Used as a lower bound; replaced by adaptive decay when CC scores are
available (see OPTICAL_DECAY_MAX).
"""

OPTICAL_DECAY_MAX: float = 0.90
"""Upper bound for adaptive optical decay.

At high alignment quality (mean CC ≈ 0.98), alignment error is small
enough that a decay of 0.90 does not over-extend the blur_limit beyond
what the Wiener deconvolution (Phase 9) can recover.
"""

OPTICAL_DECAY_CC_LOW: float = 0.70
"""CC floor below which adaptive decay is clamped to OPTICAL_DECAY_CONSTANT."""

OPTICAL_DECAY_CC_HIGH: float = 0.98
"""CC ceiling at which adaptive decay reaches OPTICAL_DECAY_MAX."""

NEFF_KDE_GRID: int = 32
"""Evaluation grid size (per axis) for KDE-based N_eff computation.
Used in alignment.py calculate_dither_quality_neff() KDE branch."""

NEFF_KDE_BW: float = 0.12
"""Gaussian KDE bandwidth in sub-pixel [0,1) units.
Controls smoothing of the sub-pixel density estimate."""

NEFF_KDE_MIN_FRAMES: int = 4
"""Minimum number of valid frames required to use KDE N_eff.
For n < NEFF_KDE_MIN_FRAMES, falls back to 4×4 histogram."""

MIN_SCALE: float = 1.0
MAX_SCALE: float = 4.0
MIN_RETAINED_RATIO: float = 0.05

# ============================================================
# Drizzle Stacking (Phase 8)
# ============================================================
DEFAULT_PIXFRAC: float = 0.7
DEFAULT_NUM_CHUNKS: int = 8
DRIZZLE_WEIGHT_FLOOR: float = 1e-6

DRIZZLE_KERNEL_MODE: str = 'lanczos2'
"""Default Drizzle interpolation kernel.

'lanczos2': Lanczos-2 kernel, radius=2 LR pixels.
  Sandbox: CV=0.041, holes=0.00%  vs  box: CV=0.133, holes=0.39%

'box': original box-overlap 4-neighbour backward drizzle (diagnostic only).
"""

DRIZZLE_LANCZOS_A: int = 2
"""Lanczos kernel order (radius in LR pixel units)."""

DRIZZLE_COVERAGE_FLOOR_RATIO: float = 0.15
"""Safety-net fill threshold (lowered from 0.35; Lanczos rarely triggers)."""

# ============================================================
# Wiener Deconvolution (Phase 9)
# ============================================================
MAD_SCALE_FACTOR: float = 1.4826
K_EST_MIN: float = 0.001
K_EST_MAX: float = 0.08
NOISE_FLOOR_HIGH_FREQ_FRACTION: float = 0.75

JPEG_NOISE_FLOOR_HIGH_FREQ_FRACTION: float = 0.60
"""JPEG spectral cutoff fix: scan noise floor from 0.60× Nyquist."""

JPEG_PSF_SCALE_FACTOR: float = 1.35
"""JPEG quantisation blur PSF correction factor."""

NOISE_PLATEAU_BINS: int = 20
NOISE_PLATEAU_GRAD_THRESHOLD: float = 0.05
MIN_SIGNAL_POWER_FRACTION: float = 0.005
K_FREQ_MIN: float = 1e-4
K_FREQ_MAX: float = 200.0
PSF_SIGMA_MIN: float = 0.6
PSF_SIGMA_SCALE: float = 0.4
PSF_TRUNCATION_SIGMAS: float = 3.0


@dataclass
class OpticoConfig:
    """Runtime configuration for the Optico pipeline."""
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
