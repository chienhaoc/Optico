"""Optico MFSR Engine — Named Constants & Configuration.

All magic numbers and empirical parameters are centralized here
for transparency, tuning, and documentation.

Cross-phase dependency map
--------------------------
JPEG source detection (pipeline.py Phase 0)
  → Phase 2 alignment: JPEG_ECC_GAUSS_FILT_SIZE (suppresses DCT block edges)
  → Phase 8 drizzle:   DRIZZLE_COVERAGE_FLOOR_RATIO (coverage-hole fill)
  → Phase 9 deconv:    JPEG_PSF_SCALE_FACTOR, JPEG_NOISE_FLOOR_HIGH_FREQ_FRACTION
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
which in turn determines whether the grid-artifact fix in drizzle.py
needs to work harder.
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
0.0 = point sampling, 1.0 = full pixel, 0.7 = typical good balance."""

DEFAULT_NUM_CHUNKS: int = 8
"""Number of horizontal strips for memory chunking."""

DRIZZLE_WEIGHT_FLOOR: float = 1e-6
"""Minimum weight to avoid division by zero in Drizzle normalization."""

DRIZZLE_COVERAGE_FLOOR_RATIO: float = 0.35
"""HR pixels whose accumulated denominator falls below this fraction of the
chunk-median denominator are flagged as coverage holes and filled by a fast
3×3 Gaussian-weighted average of surrounding pixels before normalisation.

Root cause addressed
---------------------
The backward 4-neighbour overlap calculation in _drizzle_chunk_vectorized()
has a structural blind spot: when the nearest LR pixel centre projects to a
position > r_droplet away from an HR pixel centre (which occurs at specific
sub-pixel offset phases), the overlap integral is zero for that LR pixel,
and the coverage-gap may persist across all N frames if their sub-pixel
offsets cluster in the same region.  This manifests as a grid-like
bright/dark pattern in the output (the HR pixel is normalised by a very
small denominator, amplifying numerical noise).

Fix strategy
------------
Rather than rewriting the chunked drizzle as a true forward-projection
(which would break the memory-bounded chunk architecture), we apply a
post-accumulation coverage-hole fill inside each chunk:

  1. Compute chunk-median of the denominator array.
  2. Build a boolean hole mask: denominator < threshold * median.
  3. Fill hole pixels from a 3×3 Gaussian-blurred version of the
     denominator and numerator arrays (scipy.ndimage.uniform_filter
     approximation for speed).  This is equivalent to bilinear
     interpolation from surrounding well-covered pixels.
  4. Continue with normal per-pixel normalisation.

Effect on resolution: the fill only activates for pixels that would
otherwise contain near-zero or zero signal.  Well-covered pixels are
not modified.  Spatial resolution is therefore preserved for the
majority of the image; only the artefact grid lines are smoothed.

Cross-phase note: the grid artefact is more visible at lower scales
(scale ≈ 1.6) because fewer LR pixels contribute per HR pixel.  At
higher scales (≥ 2.0) natural overlap from multiple frames fills the
gaps without intervention.  Setting this ratio to 0.0 disables the
fix entirely.
"""

# ============================================================
# Wiener Deconvolution (Phase 9)
# ============================================================
MAD_SCALE_FACTOR: float = 1.4826
"""Scale factor converting MAD to standard deviation for Gaussian noise.
Equals 1 / Phi_inv(3/4) where Phi is the standard normal CDF."""

K_EST_MIN: float = 0.001
"""Retained for backward compatibility / diagnostic logging only.
No longer used to derive the Wiener regularization parameter directly."""

K_EST_MAX: float = 0.08
"""Retained for backward compatibility / diagnostic logging only."""

NOISE_FLOOR_HIGH_FREQ_FRACTION: float = 0.75
"""Lower bound of the radial frequency search range (fraction of Nyquist)
used by _find_noise_plateau() for RAW / PNG input."""

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
The PSF fix compensates for JPEG quantisation blur in the spatial domain;
this constant fixes the frequency-domain noise estimation.  Both are needed
for full sharpness recovery on JPEG inputs.
"""

JPEG_PSF_SCALE_FACTOR: float = 1.35
"""Multiplicative factor applied to psf_sigma when the input is JPEG.

Physical basis
--------------
A JPEG-compressed image has already been blurred by two PSFs before
Optico processes it:

  1. Optical PSF of the lens (modelled by PSF_SIGMA_SCALE * scale).
  2. JPEG quantisation PSF: the DCT re-synthesis after coefficient
     truncation is equivalent to low-pass filtering with a kernel whose
     effective sigma is roughly 0.3–0.5 px (quality 85–95).

The composite effective PSF sigma is:
  sigma_eff = sqrt(sigma_optical² + sigma_jpeg²)
            ≈ sigma_optical * sqrt(1 + (sigma_jpeg/sigma_optical)²)

For typical sigma_optical ≈ 0.65 and sigma_jpeg ≈ 0.40:
  sigma_eff ≈ 0.65 * sqrt(1 + 0.38) ≈ 0.65 * 1.174 ≈ 0.76

A flat factor of 1.35 is a conservative upper-bound that avoids
under-correcting (which leaves JPEG blur visible) without
over-correcting (which would introduce ringing).

Cross-phase note: the ECC Gaussian filter size is also enlarged for JPEG
input (JPEG_ECC_GAUSS_FILT_SIZE).  That upstream fix improves sub-pixel
offset accuracy, which reduces the effective alignment PSF contribution;
this constant handles the remaining quantisation-domain blur.
"""

NOISE_PLATEAU_BINS: int = 20
"""Number of radial frequency bins for _find_noise_plateau()."""

NOISE_PLATEAU_GRAD_THRESHOLD: float = 0.05
"""Relative gradient threshold for noise plateau detection."""

MIN_SIGNAL_POWER_FRACTION: float = 0.005
"""Floor on estimated per-frequency signal power, as a fraction of the
noise floor, to avoid division-by-zero / unbounded K."""

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

    # Deconvolution
    psf_override: Optional[float] = None
    skip_deconv: bool = False

    # Input source (auto-detected by pipeline.py; override if needed)
    jpeg_input: Optional[bool] = None
