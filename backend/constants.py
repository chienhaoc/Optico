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
                       EDGE_TAPER_WIDTH (spectral leakage suppression)

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
from dataclasses import dataclass
from typing import Optional


# ============================================================
# Alignment (Phase 2)
# ============================================================
DEFAULT_ALIGN_SCALE: float = 0.5
DEFAULT_MAX_OFFSET: float = 20.0
ECC_MAX_ITERATIONS: int = 200
ECC_EPSILON: float = 1e-6
ECC_GAUSS_FILT_SIZE: int = 5
JPEG_ECC_GAUSS_FILT_SIZE: int = 7

# ============================================================
# Dynamic Masking (Phase 6)
# ============================================================
NOISE_MODEL_GAIN: float = 0.5
NOISE_MODEL_OFFSET: float = 10.0
"""Raised from 1.0 → 10.0 (2026-07): prevents dark scene regions (e.g. projector
body) from having their normalised diff inflated by a too-small noise denominator.
With a=0.5, I=58 (dark corner): sqrt(0.5*58+1)=5.6 → denom floor was tiny.
At 10.0: sqrt(0.5*58+10)=6.7+… adding 10 to the base lifts dark-area tolerance
so they are judged on the same footing as mid-tones."""
GRADIENT_WEIGHT: float = 0.3
BG_MOTION_THRESHOLD: float = 1.5
SUBJ_MOTION_THRESHOLD: float = 3.0

BG_KERNEL_SIZE: int = 7
"""Minimum BG dilation kernel size (px). Used as lower bound for adaptive sizing."""

SUBJ_KERNEL_SIZE: int = 11
"""Minimum subject dilation kernel size (px). Lower bound for adaptive sizing."""

SUBJ_DILATE_ITERATIONS: int = 2
"""Number of morphological dilation iterations for the subject-motion mask."""

MASK_BLUR_KSIZE: int = 5
"""Minimum Gaussian blur kernel for mask soft-edges (px). Lower bound."""

BG_KERNEL_MIN_DIM_FRAC: float = 0.002
"""Adaptive BG kernel: max(BG_KERNEL_SIZE, round(min_dim * frac)), forced odd."""

SUBJ_KERNEL_MIN_DIM_FRAC: float = 0.004
"""Adaptive subject kernel: max(SUBJ_KERNEL_SIZE, round(min_dim * frac))."""

MASK_BLUR_MIN_DIM_FRAC: float = 0.001
"""Adaptive mask blur: max(MASK_BLUR_KSIZE, round(min_dim * frac)), forced odd."""

# ============================================================
# Pre-flight Scale Bounding (Phase 7)
# ============================================================
OPTICAL_DECAY_CONSTANT: float = 0.75
"""Conservative optical efficiency factor (lower bound / fallback)."""

OPTICAL_DECAY_MAX: float = 0.90
"""Upper bound for adaptive optical decay at high CC quality."""

OPTICAL_DECAY_CC_LOW: float = 0.70
"""CC floor below which adaptive decay is clamped to OPTICAL_DECAY_CONSTANT."""

OPTICAL_DECAY_CC_HIGH: float = 0.98
"""CC ceiling at which adaptive decay reaches OPTICAL_DECAY_MAX."""

NEFF_KDE_GRID: int = 32
NEFF_KDE_BW: float = 0.12
NEFF_KDE_MIN_FRAMES: int = 4

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
"""Default drizzle accumulation kernel: 'lanczos2' (validated), 'nearest', 'bilinear',
'lanczos4', 'box', 'lanczos2_clamped', or 'box_supersample'.

Changed to 'lanczos2' (2026-07, real-burst benchmark): the 2026-07 revert
to 'box' below was validated only on synthetic data. A real-burst
benchmark (backend/benchmarks/kernel_bench.py, two Sony A7C tripod
bursts, 17mm + 50mm, scale=2.0) confirmed 'box's coverage-hole grid
artifact is real and severe on actual photos (grid_periodicity worst
peak-to-floor ratio 350-609 vs 'lanczos2' at matched settings), and that
combined with the psf-sigma grid-safe cap documented under
JPEG_PSF_SCALE_FACTOR / wiener_deconv() below, plain 'lanczos2' beats
'box' on ringing, grid-periodicity, AND leave-one-out fidelity
simultaneously on both real bursts — not merely a trade-off. Full data:
backend/benchmarks/reports/input{3,4}_kernel_bench.json.

'box' (2026-07, superseded above): box-overlap at the default PIXFRAC
(0.7) was verified via synthetic ground truth to have zero structural
coverage holes at N=7 while producing sharper output than 'lanczos2',
but the real-burst benchmark above showed the zero-coverage-hole
failure mode does manifest on real tripod bursts even at this pixfrac.
Retained as a selectable option (config.kernel_mode='box') for
comparison, no longer the default.

'lanczos2_clamped' (2026-07, real-burst benchmark): same sinc-shaped
footprint as 'lanczos2' but with negative sidelobe weights clamped to
zero (see _lanczos(clamp_negative=True) in drizzle.py), intended to
remove Gibbs-style ringing. The real-burst benchmark found it reduces
ringing slightly vs plain 'lanczos2' at matched psf, but counter-
intuitively has *higher* grid_periodicity at every tested psf_override
except the lowest (0.4) — clamping the negative lobes does not
uniformly improve coverage uniformity. Not selected as the default;
retained as an option for further study.

'box_supersample' (2026-07, benchmarked and discarded): accumulates with
the plain 'box' kernel at DRIZZLE_SUPERSAMPLE_FACTOR times the requested
scale, then area-decimates (cv2.INTER_AREA) back down. Benchmarked on
both real bursts: only reduced grid_periodicity by ~30-50% (e.g. 609 ->
452 on the 17mm burst), far short of 'lanczos2's reduction, with no
ringing benefit and added compute cost. Discarded per user decision;
code retained but not recommended.
"""

DRIZZLE_SUPERSAMPLE_FACTOR: int = 2
"""Supersample multiplier for kernel_mode='box_supersample'.

Kept conservative (2x, not 4x) given real-burst benchmarking on a 16GB
machine already hit OOM/timeout risk with plain 'lanczos2' at full
resolution — 2x keeps each chunk's supersampled intermediate array small
(~100-150MB at typical benchmark crop sizes) since drizzle_stack()
processes it one output chunk at a time, never holding a full-image
supersampled canvas. Increase only after confirming 2x is safe.
"""
"""Default Drizzle kernel. 'box' for diagnostic comparison."""

DRIZZLE_LANCZOS_A: int = 2
DRIZZLE_COVERAGE_FLOOR_RATIO: float = 0.15

# ============================================================
# Wiener Deconvolution (Phase 9)
# ============================================================
MAD_SCALE_FACTOR: float = 1.4826
K_EST_MIN: float = 0.001
K_EST_MAX: float = 0.08
NOISE_FLOOR_HIGH_FREQ_FRACTION: float = 0.75

JPEG_NOISE_FLOOR_HIGH_FREQ_FRACTION: float = 0.60
"""JPEG spectral cutoff fix: scan noise floor from 0.60× Nyquist."""

JPEG_PSF_SCALE_FACTOR: float = 1.10
"""JPEG quantisation blur PSF correction factor.

Lowered from 1.35 to 1.10 (2026-07) to suppress Gibbs-style edge overshoot.

With the old value of 1.35, the Wiener filter assumed a PSF sigma 35% larger
than the actual optical PSF (e.g. 1.08 vs 0.80 HR px at scale=2).  This
over-estimate causes excessive inverse-filter gain near the PSF cutoff
frequency, producing ~3 px wide undershoot bands immediately after every
high-contrast edge.  When a smooth region (e.g. a face) lies within ~5 px of
such an edge the dark undershoot lands on it and appears as a horizontal stripe.

Sandbox validation (512×512 synthetic, 4 edge-contrast scenes):
  High-contrast post-edge undershoot:  −8.74 → −5.61 ADU  (−35.7 %)
  High-contrast PSNR:                  39.72 → 40.55 dB   (+0.83 dB)
  Mid-contrast PSNR:                   42.08 → 42.10 dB   (+0.02 dB)
  No-edge face PSNR:                   46.57 → 44.77 dB   (−1.8 dB, acceptable)

A factor of 1.10 still compensates for the real JPEG quantisation blur
(which adds ~10 % effective sigma broadening) while keeping the assumed PSF
close enough to the true PSF that overshoot stays below visual threshold.
"""

NOISE_PLATEAU_BINS: int = 20
NOISE_PLATEAU_GRAD_THRESHOLD: float = 0.05
MIN_SIGNAL_POWER_FRACTION: float = 0.005
K_FREQ_MIN: float = 1e-4
K_FREQ_MAX: float = 200.0

PSF_SIGMA_MIN: float = 0.4
"""Minimum PSF sigma (px).

Lowered from 0.6 to 0.4 so the linear region PSF_SIGMA_SCALE*scale starts
at scale=1.0.  A sigma of 0.4 px corresponds to a diffraction-limited optical
PSF at 1x upscale; values below sub-pixel sampling (0.4 px) are physically
meaningless anyway, so this is also the natural physical floor.

Previous value of 0.6 caused a flat plateau in sigma(scale) for scale < 1.5,
which under-corrected blur for scale=1.0-1.5 runs.
"""

PSF_SIGMA_SCALE: float = 0.4
PSF_TRUNCATION_SIGMAS: float = 3.0

MIN_NOISE_FLOOR_ABS: float = 1.0
"""Absolute minimum noise floor power (ADU^2) for Wiener K_freq computation.

Prevents NaN in K_freq = noise_floor / signal_power when the input image is
near-uniform or near-black (noise_floor ~ 0 -> signal_power ~ 0 -> 0/0).
Value of 1.0 ADU^2 corresponds to sigma ~ 1 ADU, below any real sensor noise,
so this floor never activates on real images and only protects edge cases.
"""

EDGE_TAPER_WIDTH: int = 48
"""Cosine taper width (px) applied to all four image edges before FFT2.

FFT2-based Wiener deconvolution assumes the image is periodic (circulant).
Real images have discontinuous top/bottom/left/right edges, which causes
spectral leakage concentrated on the fx=0 and fy=0 axes. After IFFT2 this
leakage appears as full-width horizontal (and vertical) bands that cross
smooth regions such as faces — completely unrelated to image content.

The cosine taper smoothly blends the EDGE_TAPER_WIDTH outermost pixels toward
the image mean, eliminating the boundary discontinuity. The mean is added back
before the FFT so the DC component is preserved.

For JPEG input the boundary discontinuity is stronger: JPEG 8px DCT block
edges are co-aligned across all burst frames and Drizzle stacking reinforces
them instead of averaging them out, making the leakage and resulting bands more
visible than with RAW input.

Sandbox measurement (512×512 face scene, JPEG block residuals, n=8 runs):
  Without taper: row-mean std = 30.76  (+10.7% vs input 27.79)
  With taper:    row-mean std = 27.66  ( −0.5% vs input)  → banding eliminated

Tuning guide:
  - Increase (64-96) for very large images or extreme edge contrast.
  - Decrease (24-32) if the image content itself is low-contrast at the edges
    and computation time is a concern.
  - Must not exceed min(H, W) // 4 (enforced in _edge_taper).
"""


@dataclass
class OpticoConfig:
    """Runtime configuration for the Optico pipeline."""
    align_scale: float = DEFAULT_ALIGN_SCALE
    max_offset: float = DEFAULT_MAX_OFFSET
    ecc_iterations: int = ECC_MAX_ITERATIONS
    ecc_epsilon: float = ECC_EPSILON
    ecc_gauss_filt_size: int = ECC_GAUSS_FILT_SIZE

    noise_gain: float = NOISE_MODEL_GAIN
    noise_offset: float = NOISE_MODEL_OFFSET
    gradient_weight: float = GRADIENT_WEIGHT
    bg_threshold: float = BG_MOTION_THRESHOLD
    subj_threshold: float = SUBJ_MOTION_THRESHOLD

    target_scale: float = 2.0
    optical_decay: float = OPTICAL_DECAY_CONSTANT

    pixfrac: float = DEFAULT_PIXFRAC
    num_chunks: int = DEFAULT_NUM_CHUNKS
    kernel_mode: str = DRIZZLE_KERNEL_MODE

    psf_override: Optional[float] = None
    psf_base: float = 0.63
    skip_deconv: bool = False

    jpeg_input: Optional[bool] = None
    ecc_motion_mode: str = "affine"
    frame_cc_threshold: float = 0.0
    """Maximum allowed CC gap from the best frame. Frames with cc < best_cc - threshold
    are discarded. 0.0 = keep all frames (default). Typical values: 0.003-0.010."""
