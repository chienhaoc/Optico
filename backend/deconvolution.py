"""Optico MFSR Engine — Phase 9: Frequency-Dependent Wiener Deconvolution.

History
-------
This module previously used a single global (scalar) Wiener regularization
parameter K, split into two hand-tuned "bands" (K_strong / K_weak) blended
spatially via a Canny-derived edge mask. That scheme was validated against
a synthetic ground-truth benchmark (realistic test image: hard edges,
textures, gradients; noise sigma 1–09) and found to be consistently
outperformed by the frequency-dependent approach implemented here:

  Low noise   (sigma=1): PSNR +1.31 dB
  Medium noise (sigma=3): PSNR +0.88 dB, SSIM +0.064
  High noise   (sigma=6): PSNR +1.46 dB, SSIM +0.236
  Very high    (sigma=9): PSNR +2.50 dB, SSIM +0.360
  Average improvement  : PSNR +1.54 dB

JPEG-input fixes (2026-07)
---------------------------
Two failure modes on JPEG input were identified and fixed:

1. Inflated noise_floor → over-regularisation → sharpness loss
   Fix: lower NOISE_FLOOR_HIGH_FREQ_FRACTION to 0.60 for JPEG input.

2. Under-estimated PSF sigma → residual blur
   Fix: multiply psf_sigma by JPEG_PSF_SCALE_FACTOR (1.35).

NaN fix (2026-07)
------------------
For near-uniform or near-black input images, the outer-frequency power
spectrum is essentially zero, making noise_floor ≈ 0.  This caused
signal_power → 0 and K_freq = 0/0 = NaN, producing a black output.
Fix: clamp noise_floor to MIN_NOISE_FLOOR_ABS=1.0 (ADU²) before use.
At 1 ADU² (sigma ≈ 1 ADU) this floor is below any real sensor noise
and never activates on normal photographic inputs.

Edge taper fix (2026-07)
------------------------
Full-width horizontal bands were observed crossing smooth regions (e.g.
faces) when jpeg_input=True.  Root cause: scipy.fft.fft2 assumes the image
is circulant (periodic).  Real images have discontinuous top/bottom edges;
this boundary discontinuity creates spectral leakage concentrated on the
fx=0 axis, which after IFFT2 manifests as full-width horizontal stripes
entirely unrelated to image content.

For JPEG input the effect is stronger: JPEG 8px DCT block boundaries are
co-aligned across all burst frames.  Drizzle stacking reinforces rather
than averages them, increasing vertical discontinuity strength and making
the resulting bands more visible than with RAW input.

Fix: apply a cosine taper (_edge_taper) to all four image edges before
FFT2.  The taper blends EDGE_TAPER_WIDTH outermost pixels toward the image
mean, eliminating boundary discontinuity while preserving the DC component.

Sandbox measurement (512×512 face scene, JPEG block residuals, n=8 runs):
  Without taper: row-mean std = 30.76  (+10.7% vs input 27.79)
  With taper:    row-mean std = 27.66  ( −0.5% vs input)  → banding eliminated

Cross-phase note: JPEG_ECC_GAUSS_FILT_SIZE (Phase 2) improves sub-pixel
offset accuracy on JPEG input, which reduces alignment PSF contribution;
this module handles the remaining quantisation-domain component.

Grid-safe PSF sigma cap (2026-07)
----------------------------------
Real-burst benchmarking (backend/benchmarks/kernel_bench.py) found that
Wiener deconvolution amplifies the drizzle-kernel's residual grid
artifact whenever its passband reaches the grid's spatial frequency.
For a Gaussian PSF of sigma, the Wiener cutoff frequency is
approximately f_c(sigma) = sqrt(ln(1/K)) / (2*pi*sigma); the drizzle
grid artifact sits at spatial frequency 1/scale cycles per HR pixel
(alias-wrapped into the Nyquist range for scale < 2). Solving
f_c(sigma) = f_grid for sigma gives the largest PSF sigma whose passband
does not reach the grid frequency — see _grid_safe_psf_sigma_cap().

A theory-grounded sweep (psf_override in {0.88 auto, 0.80, 0.50, 0.40}
across both kernels and two real bursts, scale=2.0) confirmed this
prediction closely: grid_periodicity collapses sharply right around the
predicted threshold (e.g. lanczos2_clamped on the 50mm burst:
worst_ratio 587.7 -> 311.9 -> 41.5 -> 30.6 for psf 0.88/0.8/0.5/0.4 vs a
predicted threshold of ~0.51 from the measured K_est on that burst), with
only marginal further gain below it. This is applied as a cap on the
*auto*-computed psf_sigma only (not on an explicit --psf-override, which
remains a raw manual value for experimentation). Full data:
backend/benchmarks/reports/input{3,4}_kernel_bench.json.
"""
import logging
import time
from typing import Optional

import cv2
import numpy as np
import scipy.fft

from .constants import (
    MAD_SCALE_FACTOR,
    K_EST_MIN, K_EST_MAX,
    NOISE_FLOOR_HIGH_FREQ_FRACTION,
    JPEG_NOISE_FLOOR_HIGH_FREQ_FRACTION,
    JPEG_PSF_SCALE_FACTOR,
    NOISE_PLATEAU_BINS,
    NOISE_PLATEAU_GRAD_THRESHOLD,
    MIN_SIGNAL_POWER_FRACTION,
    MIN_NOISE_FLOOR_ABS,
    K_FREQ_MIN, K_FREQ_MAX,
    PSF_SIGMA_MIN, PSF_SIGMA_SCALE, PSF_TRUNCATION_SIGMAS,
    EDGE_TAPER_WIDTH,
)

logger = logging.getLogger(__name__)


def _estimate_noise_mad(img_y: np.ndarray) -> float:
    """Estimate noise std using Laplacian MAD (diagnostic / logging only).

    Formula: sigma = (1.4826 * MAD(Lap(I))) / sqrt(20)
    """
    lap = cv2.Laplacian(img_y, cv2.CV_32F)
    median_lap = float(np.median(lap))
    mad = float(np.median(np.abs(lap - median_lap)))
    sigma = (MAD_SCALE_FACTOR * mad) / np.sqrt(20.0)
    return max(sigma, 1e-8)


def _edge_taper(
    img: np.ndarray,
    taper_width: int = EDGE_TAPER_WIDTH,
) -> np.ndarray:
    """Apply a cosine taper to all four image edges to suppress FFT spectral leakage.

    FFT2-based Wiener deconvolution assumes the image is circulant (periodic).
    Real images have discontinuous boundaries; the discontinuity creates
    spectral leakage concentrated on the fx=0 / fy=0 axes.  After IFFT2,
    this leakage appears as full-width horizontal (and vertical) bands that
    cross smooth regions such as faces, entirely unrelated to image content.

    The taper blends the outermost ``taper_width`` pixels on each side
    smoothly toward the image mean using a raised-cosine (Hann) ramp:
        w[i] = 0.5 * (1 - cos(π·i / taper_width)),  i = 0 … taper_width-1
    The image mean is subtracted before tapering and added back afterwards,
    preserving the DC component.

    Parameters
    ----------
    img:
        2-D float32 array, pixel values in [0, 255].
    taper_width:
        Number of pixels to taper on each edge.  Clamped to min(H, W) // 4.

    Returns
    -------
    2-D float32 array with tapered boundaries, same shape as ``img``.
    """
    h, w = img.shape
    mean_val = float(img.mean())
    t = min(taper_width, min(h, w) // 4)

    # Raised-cosine ramp: 0 at the edge pixel, 1 at pixel t
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(t) / t))

    result = img.astype(np.float64) - mean_val

    # Apply ramp to top/bottom rows and left/right columns
    for i in range(t):
        result[i, :]       *= ramp[i]
        result[h - 1 - i, :] *= ramp[i]
        result[:, i]         *= ramp[i]
        result[:, w - 1 - i] *= ramp[i]

    return (result + mean_val).astype(np.float32)


def _build_gaussian_psf(
    sigma: float,
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """Build a Gaussian PSF kernel, circulant-wrapped to image size."""
    radius = int(np.ceil(PSF_TRUNCATION_SIGMAS * sigma))
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    kernel = np.exp(
        -(x.astype(np.float64) ** 2 + y.astype(np.float64) ** 2)
        / (2.0 * sigma ** 2)
    )
    kernel /= kernel.sum()

    psf = np.zeros((img_h, img_w), dtype=np.float64)
    for ky in range(-radius, radius + 1):
        for kx in range(-radius, radius + 1):
            psf[ky % img_h, kx % img_w] = kernel[ky + radius, kx + radius]
    return psf


def _radial_frequency_grid(img_h: int, img_w: int) -> np.ndarray:
    """Return a (H, W) array of normalised radial spatial frequencies."""
    fy = np.fft.fftfreq(img_h)[:, None]
    fx = np.fft.fftfreq(img_w)[None, :]
    return np.sqrt(fy ** 2 + fx ** 2)


def _find_noise_plateau(
    power: np.ndarray,
    freq_r: np.ndarray,
    high_freq_fraction: float = NOISE_FLOOR_HIGH_FREQ_FRACTION,
) -> tuple[float, float]:
    """Adaptively locate the onset of the white-noise plateau."""
    frmax = freq_r.max()
    lo = high_freq_fraction
    bins = np.linspace(lo, 0.98, NOISE_PLATEAU_BINS + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    medians = []
    for lo_b, hi_b in zip(bins[:-1], bins[1:]):
        mask = (freq_r >= lo_b * frmax) & (freq_r < hi_b * frmax)
        medians.append(float(np.median(power[mask])) if mask.any() else np.nan)

    medians = np.array(medians)
    valid = np.isfinite(medians)
    if not valid.any():
        fallback_mask = freq_r >= lo * frmax
        return float(np.median(power[fallback_mask])), lo

    ref = medians[valid][0]
    grads = np.abs(np.diff(np.where(valid, medians, np.nan)))
    threshold = NOISE_PLATEAU_GRAD_THRESHOLD * ref

    plateau_bin = next(
        (i for i, g in enumerate(grads) if np.isfinite(g) and g < threshold),
        len(grads) - 1,
    )
    plateau_frac = float(bin_centers[plateau_bin])
    plateau_mask = freq_r >= (plateau_frac * frmax)
    noise_floor = float(np.median(power[plateau_mask]))
    return noise_floor, plateau_frac


def _grid_safe_psf_sigma_cap(K: float, scale: float) -> float:
    """Largest Gaussian PSF sigma whose Wiener passband does not reach the
    drizzle grid-artifact frequency, for regularization level K at the
    given upscale factor.

    f_c(sigma) = sqrt(ln(1/K)) / (2*pi*sigma); solving f_c(sigma) = f_grid
    for sigma gives sigma_cap = sqrt(ln(1/K)) / (2*pi*f_grid). f_grid is
    1/scale, alias-wrapped into (0, 0.5] since spatial frequencies above
    Nyquist are not physically meaningful.
    """
    wrapped = (1.0 / scale) % 1.0
    if wrapped > 0.5:
        wrapped -= 1.0
    f_grid = abs(wrapped)
    if f_grid <= 1e-6:
        return float("inf")
    K_safe = max(K, 1e-6)
    return float(np.sqrt(np.log(1.0 / K_safe)) / (2.0 * np.pi * f_grid))


def _estimate_frequency_dependent_K(
    img_f32: np.ndarray,
    freq_r: np.ndarray,
    high_freq_fraction: float = NOISE_FLOOR_HIGH_FREQ_FRACTION,
) -> tuple[np.ndarray, float, float]:
    """Estimate a per-frequency Wiener regularization map K(f)."""
    G_fft = scipy.fft.fft2(img_f32.astype(np.float64), workers=-1)
    power = np.abs(G_fft) ** 2

    noise_floor_raw, plateau_frac = _find_noise_plateau(
        power, freq_r, high_freq_fraction=high_freq_fraction
    )

    # Clamp noise floor to minimum to avoid 0/0 NaN for near-uniform inputs
    noise_floor = max(noise_floor_raw, MIN_NOISE_FLOOR_ABS)
    if noise_floor > noise_floor_raw:
        logger.debug(
            "noise_floor clamped %.3e → %.3e (MIN_NOISE_FLOOR_ABS)",
            noise_floor_raw, noise_floor,
        )

    signal_power = np.maximum(
        power - noise_floor,
        noise_floor * MIN_SIGNAL_POWER_FRACTION,
    )
    K_freq = np.clip(noise_floor / signal_power, K_FREQ_MIN, K_FREQ_MAX)
    return K_freq, noise_floor, plateau_frac


def wiener_deconv(
    img: np.ndarray,
    scale: float,
    psf_override: Optional[float] = None,
    jpeg_input: bool = False,
) -> np.ndarray:
    """Frequency-Dependent Wiener Deconvolution with DC Gain Preservation.

    An edge taper is applied before FFT2 to eliminate full-width horizontal
    banding caused by spectral leakage at image boundaries.  See
    _edge_taper() and EDGE_TAPER_WIDTH in constants.py for details.
    """
    t0 = time.time()
    H, W = img.shape[:2]
    img_f32 = img.astype(np.float32)

    sigma_noise = _estimate_noise_mad(img_f32)
    K_est_diagnostic = float(np.clip(sigma_noise ** 2, K_EST_MIN, K_EST_MAX))

    psf_sigma = max(PSF_SIGMA_MIN, PSF_SIGMA_SCALE * scale)
    if psf_override is not None:
        psf_sigma = float(psf_override)
    else:
        if jpeg_input:
            psf_sigma *= JPEG_PSF_SCALE_FACTOR
            logger.debug(
                "JPEG input: psf_sigma scaled by %.2f → %.3f",
                JPEG_PSF_SCALE_FACTOR, psf_sigma,
            )
        grid_safe_cap = _grid_safe_psf_sigma_cap(K_est_diagnostic, scale)
        if psf_sigma > grid_safe_cap:
            logger.debug(
                "psf_sigma capped %.3f → %.3f (grid-safe cap, K_est=%.4f, scale=%.2f)",
                psf_sigma, grid_safe_cap, K_est_diagnostic, scale,
            )
            psf_sigma = grid_safe_cap

    noise_hf_fraction = (
        JPEG_NOISE_FLOOR_HIGH_FREQ_FRACTION
        if jpeg_input
        else NOISE_FLOOR_HIGH_FREQ_FRACTION
    )

    freq_r = _radial_frequency_grid(H, W)

    # Apply edge taper before FFT2 to suppress spectral leakage banding.
    # The taper is used ONLY for the noise-floor estimation and the
    # Wiener filter computation; the deconvolution result is corrected
    # back to the original (untapered) domain by applying the same filter
    # to both the tapered and the original image and blending the edge
    # pixels back.  This avoids the brightness ramp at image borders that
    # would otherwise appear in the output.
    img_tapered = _edge_taper(img_f32, taper_width=EDGE_TAPER_WIDTH)

    K_freq, noise_floor, plateau_frac = _estimate_frequency_dependent_K(
        img_tapered, freq_r, high_freq_fraction=noise_hf_fraction
    )

    logger.info(
        "Wiener: jpeg=%s  sigma_noise(diag)=%.4f  K_est(diag)=%.4f "
        "noise_floor=%.3e  plateau_frac=%.2f  "
        "K_freq=[%.4f, %.2f]  PSF_sigma=%.3f",
        jpeg_input, sigma_noise, K_est_diagnostic,
        noise_floor, plateau_frac,
        float(K_freq.min()), float(K_freq.max()),
        psf_sigma,
    )

    psf = _build_gaussian_psf(psf_sigma, H, W)
    H_fft = scipy.fft.fft2(psf, workers=-1)

    # Run Wiener on the tapered signal (to suppress banding), then also
    # on the original signal so we can restore the edge region.
    H_conj = np.conj(H_fft)
    H_power = np.abs(H_fft) ** 2
    W_resp = H_conj / (H_power + K_freq)
    W_resp[0, 0] = 1.0  # DC-gain preservation

    G_fft_tapered = scipy.fft.fft2(img_tapered.astype(np.float64), workers=-1)
    G_fft_orig    = scipy.fft.fft2(img_f32.astype(np.float64), workers=-1)

    result_tapered = np.real(scipy.fft.ifft2(W_resp * G_fft_tapered, workers=-1)).astype(np.float32)
    result_orig    = np.real(scipy.fft.ifft2(W_resp * G_fft_orig,    workers=-1)).astype(np.float32)

    # Blend: use result_tapered in the interior (where it suppresses
    # banding), and restore result_orig in the taper border (where the
    # tapered output would otherwise show a brightness ramp).
    t = min(EDGE_TAPER_WIDTH, min(H, W) // 4)
    ramp = (0.5 * (1.0 - np.cos(np.pi * np.arange(t) / t))).astype(np.float32)

    # alpha = 1 → fully use tapered result; alpha = 0 → use orig result
    alpha = np.ones((H, W), dtype=np.float32)
    for i in range(t):
        alpha[i, :]       = ramp[i]
        alpha[H - 1 - i, :] = ramp[i]
        alpha[:, i]         = np.minimum(alpha[:, i], ramp[i])
        alpha[:, W - 1 - i] = np.minimum(alpha[:, W - 1 - i], ramp[i])

    result = (alpha * result_tapered + (1.0 - alpha) * result_orig)

    elapsed = time.time() - t0
    logger.info("Wiener deconvolution complete (%.2fs)", elapsed)
    return result


def deconvolve_color(
    img_bgr: np.ndarray,
    scale: float,
    psf_override: Optional[float] = None,
    jpeg_input: bool = False,
) -> np.ndarray:
    """Apply Wiener deconvolution to a color image (luma channel only)."""
    img_scaled = (img_bgr / 255.0).astype(np.float32)
    ycrcb_f32 = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2YCrCb)

    y_scaled = ycrcb_f32[:, :, 0] * 255.0
    y_deconv = wiener_deconv(y_scaled, scale, psf_override, jpeg_input=jpeg_input)

    ycrcb_f32[:, :, 0] = np.clip(y_deconv, 0.0, 255.0) / 255.0
    result_scaled = cv2.cvtColor(ycrcb_f32, cv2.COLOR_YCrCb2BGR)
    return result_scaled * 255.0
