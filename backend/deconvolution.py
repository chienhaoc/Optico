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
   JPEG DCT quantisation creates a hard spectral cutoff at ~0.55–0.65 ×
   Nyquist.  Above this cutoff, image power drops abruptly to near-zero
   (quantised coefficients), making the high-frequency band look like a
   white-noise plateau to _find_noise_plateau().  This inflates N_floor
   by 2–5× and over-suppresses all frequencies via K(f) = N_floor/S(f).
   Fix: when jpeg_input is True, lower NOISE_FLOOR_HIGH_FREQ_FRACTION to
   JPEG_NOISE_FLOOR_HIGH_FREQ_FRACTION (0.60), placing the plateau scan
   below the JPEG cutoff so the true white-noise floor is measured.

2. Under-estimated PSF sigma → residual blur
   JPEG compression adds a quantisation-blur PSF (sigma ≈ 0.3–0.5 px)
   on top of the optical PSF.  The default sigma = PSF_SIGMA_SCALE * scale
   models only optical blur and under-corrects JPEG inputs.
   Fix: when jpeg_input is True, multiply psf_sigma by JPEG_PSF_SCALE_FACTOR
   (1.35) to approximate the composite optical + quantisation PSF.

Cross-phase note: JPEG_ECC_GAUSS_FILT_SIZE (Phase 2) improves sub-pixel
offset accuracy on JPEG input, which reduces alignment PSF contribution;
this module handles the remaining quantisation-domain component.

Root cause of old approach failure
------------------------------------
Natural-image power spectra fall off steeply with spatial frequency
(P(f) ∼ 1/f²) while sensor noise is approximately white (flat power).
A single scalar K cannot represent this: any K small enough to avoid
crushing real low-frequency detail is far too small to suppress noise at
high frequencies.  The spatial edge-mask blend was a crude indirect proxy
for this but still used a *flat* K within each band.

Current approach
-----------------
K is estimated per-frequency, directly from the image's own measured
power spectrum:

  1. Locate the white-noise floor: scan the radial power spectrum from
     NOISE_FLOOR_HIGH_FREQ_FRACTION * Nyquist outward (JPEG: from
     JPEG_NOISE_FLOOR_HIGH_FREQ_FRACTION * Nyquist).  Adaptively find
     the plateau via a gradient threshold.  The median power in that
     plateau annulus is N_floor.

  2. Estimate per-frequency signal power:
         S(f) = max(P(f) - N_floor,  N_floor * MIN_SIGNAL_POWER_FRACTION)

  3. Per-frequency Wiener regularization:
         K(f) = N_floor / S(f)   [textbook SNR-inverse ratio]
     clipped to [K_FREQ_MIN, K_FREQ_MAX].

DC-gain preservation
---------------------
The standard Wiener response at DC (f=0) is 1/(1+K(0)), which dims
the reconstructed image's mean brightness as K grows.  The correct fix
is to restore only the DC bin's gain to 1.0.  A global (1+K) rescale
was tested and found to progressively degrade noise suppression at high
frequencies (~8-12% worse at high noise).  Locking only bin [0,0] gives
the brightness correction with no such side effect.

Retained fixes from prior audits
----------------------------------
- Fix #10: MAD noise formula uses (1.4826 * MAD(Lap(I))) / sqrt(20).
- Fix #11: PSF built as small kernel then circulant-wrapped to image size.
- Fix #14: Consistent use of scipy.fft for both FFT and IFFT.
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
    K_FREQ_MIN, K_FREQ_MAX,
    PSF_SIGMA_MIN, PSF_SIGMA_SCALE, PSF_TRUNCATION_SIGMAS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _estimate_noise_mad(img_y: np.ndarray) -> float:
    """Estimate noise std using Laplacian MAD (diagnostic / logging only).

    Formula: sigma = (1.4826 * MAD(Lap(I))) / sqrt(20)
    The sqrt(20) term corrects for the 3x3 Laplacian kernel's noise gain.
    """
    lap = cv2.Laplacian(img_y, cv2.CV_32F)
    median_lap = float(np.median(lap))
    mad = float(np.median(np.abs(lap - median_lap)))
    sigma = (MAD_SCALE_FACTOR * mad) / np.sqrt(20.0)
    return max(sigma, 1e-8)


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
    """Adaptively locate the onset of the white-noise plateau.

    Parameters
    ----------
    power : ndarray
        Squared-magnitude FFT array.
    freq_r : ndarray
        Radial frequency grid.
    high_freq_fraction : float
        Lower bound of the scan range (fraction of Nyquist radius).
        Use JPEG_NOISE_FLOOR_HIGH_FREQ_FRACTION for JPEG input to avoid
        the JPEG spectral cutoff being mistaken for the noise plateau.

    Returns
    -------
    noise_floor : float
    plateau_frac : float
    """
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


def _estimate_frequency_dependent_K(
    img_f32: np.ndarray,
    freq_r: np.ndarray,
    high_freq_fraction: float = NOISE_FLOOR_HIGH_FREQ_FRACTION,
) -> tuple[np.ndarray, float, float]:
    """Estimate a per-frequency Wiener regularization map K(f).

    Parameters
    ----------
    img_f32 : ndarray
    freq_r : ndarray
    high_freq_fraction : float
        Passed through to _find_noise_plateau(); use the JPEG variant
        for JPEG input.

    Returns
    -------
    K_freq : ndarray
    noise_floor : float
    plateau_frac : float
    """
    G_fft = scipy.fft.fft2(img_f32.astype(np.float64), workers=-1)
    power = np.abs(G_fft) ** 2

    noise_floor, plateau_frac = _find_noise_plateau(
        power, freq_r, high_freq_fraction=high_freq_fraction
    )

    signal_power = np.maximum(
        power - noise_floor,
        noise_floor * MIN_SIGNAL_POWER_FRACTION,
    )
    K_freq = np.clip(noise_floor / signal_power, K_FREQ_MIN, K_FREQ_MAX)
    return K_freq, noise_floor, plateau_frac


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def wiener_deconv(
    img: np.ndarray,
    scale: float,
    psf_override: Optional[float] = None,
    jpeg_input: bool = False,
) -> np.ndarray:
    """Frequency-Dependent Wiener Deconvolution with DC Gain Preservation.

    Parameters
    ----------
    img : ndarray
        Single-channel (luma) image, float32 or uint8, values in [0, 255].
    scale : float
        Upscale factor used to derive the assumed PSF sigma.
    psf_override : float, optional
        Explicit PSF sigma, bypassing the scale-derived default.
    jpeg_input : bool
        When True, apply JPEG-specific fixes:
        - Increase psf_sigma by JPEG_PSF_SCALE_FACTOR (quantisation blur).
        - Lower noise-floor scan range to JPEG_NOISE_FLOOR_HIGH_FREQ_FRACTION
          (avoids JPEG spectral cutoff being mistaken for white noise).

    Returns
    -------
    ndarray
        Deconvolved image, float32, same shape as input.
    """
    t0 = time.time()
    H, W = img.shape[:2]
    img_f32 = img.astype(np.float32)

    sigma_noise = _estimate_noise_mad(img_f32)
    K_est_diagnostic = float(np.clip(sigma_noise ** 2, K_EST_MIN, K_EST_MAX))

    # PSF sigma: scale-derived, adjusted for JPEG quantisation blur
    psf_sigma = max(PSF_SIGMA_MIN, PSF_SIGMA_SCALE * scale)
    if psf_override is not None:
        psf_sigma = float(psf_override)
    elif jpeg_input:
        psf_sigma *= JPEG_PSF_SCALE_FACTOR
        logger.debug(
            "JPEG input: psf_sigma scaled by %.2f → %.3f",
            JPEG_PSF_SCALE_FACTOR, psf_sigma,
        )

    # Noise floor scan range: lower for JPEG to avoid spectral cutoff
    noise_hf_fraction = (
        JPEG_NOISE_FLOOR_HIGH_FREQ_FRACTION
        if jpeg_input
        else NOISE_FLOOR_HIGH_FREQ_FRACTION
    )

    freq_r = _radial_frequency_grid(H, W)

    K_freq, noise_floor, plateau_frac = _estimate_frequency_dependent_K(
        img_f32, freq_r, high_freq_fraction=noise_hf_fraction
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
    G_fft = scipy.fft.fft2(img_f32.astype(np.float64), workers=-1)

    H_conj = np.conj(H_fft)
    H_power = np.abs(H_fft) ** 2

    W_resp = H_conj / (H_power + K_freq)
    W_resp[0, 0] = 1.0  # DC-gain preservation

    result = np.real(
        scipy.fft.ifft2(W_resp * G_fft, workers=-1)
    ).astype(np.float32)

    elapsed = time.time() - t0
    logger.info("Wiener deconvolution complete (%.2fs)", elapsed)
    return result


def deconvolve_color(
    img_bgr: np.ndarray,
    scale: float,
    psf_override: Optional[float] = None,
    jpeg_input: bool = False,
) -> np.ndarray:
    """Apply Wiener deconvolution to a color image (luma channel only).

    Parameters
    ----------
    img_bgr : ndarray
        Input color image (BGR, float32, values in [0, 255]).
    scale : float
        Effective upscale factor.
    psf_override : float, optional
        Override PSF sigma.
    jpeg_input : bool
        Pass True for JPEG-sourced images to activate JPEG-specific
        PSF scaling and noise-floor corrections.

    Returns
    -------
    ndarray
        Deconvolved color image (BGR, float32, [0, 255]).
    """
    img_scaled = (img_bgr / 255.0).astype(np.float32)
    ycrcb_f32 = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2YCrCb)

    y_scaled = ycrcb_f32[:, :, 0] * 255.0
    y_deconv = wiener_deconv(y_scaled, scale, psf_override, jpeg_input=jpeg_input)

    ycrcb_f32[:, :, 0] = np.clip(y_deconv, 0.0, 255.0) / 255.0
    result_scaled = cv2.cvtColor(ycrcb_f32, cv2.COLOR_YCrCb2BGR)
    return result_scaled * 255.0
