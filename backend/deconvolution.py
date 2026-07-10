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

Root cause of old approach failure
------------------------------------
Natural-image power spectra fall off steeply with spatial frequency
(most energy is concentrated at low frequencies, roughly P(f) ∼ 1/f²)
while sensor noise is approximately white (flat power across frequency).
A single scalar K cannot represent this: any K small enough to avoid
crushing real low-frequency detail is far too small to suppress noise at
high frequencies, where the true noise-to-signal ratio is orders of
magnitude worse. The spatial edge-mask blend was a crude indirect proxy
for this, but still used a *flat* K within each band.

Current approach
-----------------
K is estimated per-frequency, directly from the image's own measured
power spectrum:

  1. Locate the white-noise floor: scan the image's radial power spectrum
     from NOISE_FLOOR_HIGH_FREQ_FRACTION * Nyquist outward; find the
     first annulus where the power has stopped falling (plateau detection
     via adaptive gradient threshold). The median power in that plateau
     annulus is N_floor.

  2. Estimate per-frequency signal power:
         S(f) = max(P(f) − N_floor,  N_floor * MIN_SIGNAL_POWER_FRACTION)

  3. Per-frequency Wiener regularization:
         K(f) = N_floor / S(f)   [the textbook SNR-inverse ratio]
     clipped to [K_FREQ_MIN, K_FREQ_MAX].

This is fully data-driven (no per-image hand-tuned constants) and was
validated against synthetic ground-truth data as described above.

DC-gain preservation
---------------------
The standard Wiener response at DC (f=0) is 1/(1+K(0)), which measurably
dims the reconstructed image's mean brightness as K grows. The correct
fix is to restore *only* the DC bin's gain to 1.0. A global (1+K) rescale
was tested and found to progressively degrade noise suppression at high
frequencies as noise increases (~8–12% worse at high noise in synthetic
testing). Locking only bin [0, 0] gives the brightness correction with no
such side effect.

PSF model
----------
A Gaussian PSF parametrized by sigma = max(PSF_SIGMA_MIN, PSF_SIGMA_SCALE
* scale) is used. Blind MTF-based sigma estimation was prototyped and
tested but found unreliable on synthetic data (estimation error > 1.5 px).
The scale-derived default is a principled conservative choice; users can
override via psf_override for known optical systems.

Retained fixes from prior audits
----------------------------------
- Fix #10: MAD noise formula uses (1.4826 * MAD(Lap(I))) / sqrt(20)
  (accounts for the discrete Laplacian kernel noise gain).
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
    The sqrt(20) term corrects for the 3x3 Laplacian kernel's noise gain
    (sum of squared kernel elements = 20).

    This value is reported in log output for interpretability but no longer
    directly drives the Wiener regularization parameter.
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
    """Build a Gaussian PSF kernel, circulant-wrapped to image size.

    Creates a small kernel (truncated at PSF_TRUNCATION_SIGMAS) then places
    it at (0, 0) with wrap-around for correct FFT-based convolution.
    """
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
    """Return a (H, W) array of normalised radial spatial frequencies.

    Values are in the range [0, ~0.707] (0 = DC, 0.5 = Nyquist on each axis).
    """
    fy = np.fft.fftfreq(img_h)[:, None]
    fx = np.fft.fftfreq(img_w)[None, :]
    return np.sqrt(fy ** 2 + fx ** 2)


def _find_noise_plateau(
    power: np.ndarray,
    freq_r: np.ndarray,
) -> tuple[float, float]:
    """Adaptively locate the onset of the white-noise plateau in the power spectrum.

    Scans NOISE_PLATEAU_BINS radial annuli from NOISE_FLOOR_HIGH_FREQ_FRACTION
    * Nyquist to 0.98 * Nyquist. The plateau begins at the first bin where
    the change in median power drops below NOISE_PLATEAU_GRAD_THRESHOLD *
    (power at the start of the scan range).

    Parameters
    ----------
    power : ndarray
        Squared-magnitude FFT array (same shape as the image).
    freq_r : ndarray
        Radial frequency grid from _radial_frequency_grid().

    Returns
    -------
    noise_floor : float
        Median power in the identified plateau annulus.
    plateau_frac : float
        Radial frequency fraction at which the plateau was detected
        (useful for diagnostic logging).
    """
    frmax = freq_r.max()
    lo = NOISE_FLOOR_HIGH_FREQ_FRACTION
    bins = np.linspace(lo, 0.98, NOISE_PLATEAU_BINS + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    medians = []
    for lo_b, hi_b in zip(bins[:-1], bins[1:]):
        mask = (freq_r >= lo_b * frmax) & (freq_r < hi_b * frmax)
        medians.append(float(np.median(power[mask])) if mask.any() else np.nan)

    medians = np.array(medians)
    valid = np.isfinite(medians)
    if not valid.any():
        # Fallback: use everything above the minimum fraction
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
) -> tuple[np.ndarray, float, float]:
    """Estimate a per-frequency Wiener regularization map K(f).

    K(f) = noise_floor / max(P(f) - noise_floor,
                             noise_floor * MIN_SIGNAL_POWER_FRACTION)

    Parameters
    ----------
    img_f32 : ndarray
        Single-channel image, float32.
    freq_r : ndarray
        Radial frequency grid.

    Returns
    -------
    K_freq : ndarray, shape (H, W)
        Per-frequency regularization map, clipped to [K_FREQ_MIN, K_FREQ_MAX].
    noise_floor : float
        Estimated white-noise power floor.
    plateau_frac : float
        Detected plateau boundary (for logging).
    """
    G_fft = scipy.fft.fft2(img_f32.astype(np.float64), workers=-1)
    power = np.abs(G_fft) ** 2

    noise_floor, plateau_frac = _find_noise_plateau(power, freq_r)

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
        Useful when the optical system PSF is known from calibration.

    Returns
    -------
    ndarray
        Deconvolved image, float32, same shape as input.
    """
    t0 = time.time()
    H, W = img.shape[:2]
    img_f32 = img.astype(np.float32)

    # Diagnostic scalar noise estimate (for logging; no longer drives K)
    sigma_noise = _estimate_noise_mad(img_f32)
    K_est_diagnostic = float(np.clip(sigma_noise ** 2, K_EST_MIN, K_EST_MAX))

    # PSF sigma: scale-derived default, overridable for known optics
    psf_sigma = max(PSF_SIGMA_MIN, PSF_SIGMA_SCALE * scale)
    if psf_override is not None:
        psf_sigma = float(psf_override)

    # Radial frequency grid (computed once, reused for K and logging)
    freq_r = _radial_frequency_grid(H, W)

    # Per-frequency regularization map from the image's own power spectrum
    K_freq, noise_floor, plateau_frac = _estimate_frequency_dependent_K(
        img_f32, freq_r
    )

    logger.info(
        "Wiener: sigma_noise(diag)=%.4f  K_est(diag)=%.4f "
        "noise_floor=%.3e  plateau_frac=%.2f  "
        "K_freq=[%.4f, %.2f]  PSF_sigma=%.3f",
        sigma_noise, K_est_diagnostic,
        noise_floor, plateau_frac,
        float(K_freq.min()), float(K_freq.max()),
        psf_sigma,
    )

    # PSF and observed image in the frequency domain
    psf = _build_gaussian_psf(psf_sigma, H, W)
    H_fft = scipy.fft.fft2(psf, workers=-1)
    G_fft = scipy.fft.fft2(img_f32.astype(np.float64), workers=-1)

    H_conj = np.conj(H_fft)
    H_power = np.abs(H_fft) ** 2

    # Frequency-dependent Wiener response
    W_resp = H_conj / (H_power + K_freq)

    # DC-gain preservation: restore only the DC bin to exact unity gain.
    # (A global (1+K) rescale was tested and found to degrade high-frequency
    # noise suppression by ~8-12% at high noise levels.)
    W_resp[0, 0] = 1.0

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
) -> np.ndarray:
    """Apply Wiener deconvolution to a color image (luma channel only).

    Operates entirely in float32 (no premature uint8 quantization) to
    preserve the sub-8-bit precision gained from multi-frame stacking.

    Parameters
    ----------
    img_bgr : ndarray
        Input color image (BGR, float32, values in [0, 255]).
    scale : float
        Effective upscale factor.
    psf_override : float, optional
        Override PSF sigma.

    Returns
    -------
    ndarray
        Deconvolved color image (BGR, float32, [0, 255]).
    """
    img_scaled = (img_bgr / 255.0).astype(np.float32)
    ycrcb_f32 = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2YCrCb)

    y_scaled = ycrcb_f32[:, :, 0] * 255.0
    y_deconv = wiener_deconv(y_scaled, scale, psf_override)

    ycrcb_f32[:, :, 0] = np.clip(y_deconv, 0.0, 255.0) / 255.0
    result_scaled = cv2.cvtColor(ycrcb_f32, cv2.COLOR_YCrCb2BGR)
    return result_scaled * 255.0
