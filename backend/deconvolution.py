"""Optico MFSR Engine — Phase 9: Adaptive Dual-Band Wiener Deconvolution.

Corrections applied from algorithm audit:
- Fix #10: MAD formula uses (median(|lap - median(lap)|) * 1.4826) / sqrt(20)
- Fix #11: PSF built as small kernel then zero-padded
- Fix #12: Complete dual-band blending in spatial domain
- Fix #14: Consistent use of scipy.fft for both FFT and IFFT
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
    K_STRONG_MULTIPLIER, K_WEAK_MULTIPLIER,
    K_STRONG_FLOOR, K_WEAK_FLOOR,
    PSF_SIGMA_MIN, PSF_SIGMA_SCALE, PSF_TRUNCATION_SIGMAS,
    EDGE_CANNY_LOW, EDGE_CANNY_HIGH, EDGE_MASK_BLUR_KSIZE,
)

logger = logging.getLogger(__name__)


def _estimate_noise_mad(img_y: np.ndarray) -> float:
    """Estimate noise standard deviation using Laplacian MAD.

    Uses the corrected formula that accounts for the noise amplification
    factor of the discrete Laplacian kernel:
        sigma = (1.4826 * MAD(Lap(I))) / sqrt(20)

    The standard 3x3 discrete Laplacian kernel with ksize=1 has elements:
        [[ 0,  1,  0],
         [ 1, -4,  1],
         [ 0,  1,  0]]
    Sum of squares is 0^2 + 1^2 + 0^2 + 1^2 + (-4)^2 + 1^2 + 0^2 + 1^2 + 0^2 = 20.
    Thus, for i.i.d. Gaussian noise with std sigma, the output of the
    Laplacian has standard deviation std_out = sqrt(20) * sigma.
    To estimate the original noise floor, we must scale by 1 / sqrt(20).

    Parameters
    ----------
    img_y : np.ndarray
        Single-channel image (float32).

    Returns
    -------
    float
        Estimated noise standard deviation.
    """
    lap = cv2.Laplacian(img_y, cv2.CV_32F)
    median_lap = float(np.median(lap))
    mad = float(np.median(np.abs(lap - median_lap)))
    # Scale by MAD multiplier (1.4826) and divide by the kernel noise amplification factor (sqrt(20))
    sigma = (MAD_SCALE_FACTOR * mad) / np.sqrt(20.0)
    return max(sigma, 1e-8)


def _build_gaussian_psf(
    sigma: float,
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """Build a Gaussian PSF kernel, zero-padded to image size.

    Creates a small kernel (truncated at PSF_TRUNCATION_SIGMAS) then
    places it centered at (0,0) with wrap-around for FFT.

    Parameters
    ----------
    sigma : float
        PSF standard deviation.
    img_h, img_w : int
        Target image dimensions.

    Returns
    -------
    np.ndarray
        PSF array of shape (img_h, img_w), float64, normalized,
        centered at (0, 0) ready for FFT.
    """
    radius = int(np.ceil(PSF_TRUNCATION_SIGMAS * sigma))
    ksize = 2 * radius + 1

    # Build centered kernel
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    kernel = np.exp(
        -(x.astype(np.float64)**2 + y.astype(np.float64)**2)
        / (2.0 * sigma**2)
    )
    kernel /= kernel.sum()

    # Place in image-sized array with center at (0, 0)
    # using circulant embedding (wrap-around)
    psf = np.zeros((img_h, img_w), dtype=np.float64)
    for ky in range(-radius, radius + 1):
        for kx in range(-radius, radius + 1):
            py = ky % img_h
            px = kx % img_w
            psf[py, px] = kernel[ky + radius, kx + radius]

    return psf


def _build_edge_mask(img_y: np.ndarray) -> np.ndarray:
    """Build a soft edge mask for dual-band blending.

    Uses Canny edge detection with Gaussian smoothing to create
    a continuous mask: 1.0 = edge region, 0.0 = flat region.

    Parameters
    ----------
    img_y : np.ndarray
        Single-channel image (float32).

    Returns
    -------
    np.ndarray
        Soft edge mask (float32, [0, 1]).
    """
    img_u8 = np.clip(img_y, 0, 255).astype(np.uint8)
    edges = cv2.Canny(img_u8, EDGE_CANNY_LOW, EDGE_CANNY_HIGH)

    # Dilate edges for protective margin
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Smooth to create soft transition
    edge_mask = cv2.GaussianBlur(
        edges.astype(np.float32) / 255.0,
        (EDGE_MASK_BLUR_KSIZE, EDGE_MASK_BLUR_KSIZE),
        0,
    )
    return edge_mask


def wiener_deconv(
    img: np.ndarray,
    scale: float,
    psf_override: Optional[float] = None,
) -> np.ndarray:
    """Adaptive Dual-Band Wiener Deconvolution.

    Applies frequency-domain Wiener filtering with two regularization
    strengths, then blends in spatial domain using an edge-aware mask:
    - Flat regions: aggressive restoration (K_strong)
    - Edge regions: conservative restoration (K_weak)

    Parameters
    ----------
    img : np.ndarray
        Input image (single channel, float32, [0, 255]).
    scale : float
        The effective upscale factor (used to set PSF sigma).
    psf_override : float, optional
        Override PSF sigma.

    Returns
    -------
    np.ndarray
        Deconvolved image (float32, same shape).
    """
    t0 = time.time()
    H, W = img.shape[:2]
    img_f32 = img.astype(np.float32)

    # 1. Dynamic Noise Estimation (Fix #10)
    sigma_noise = _estimate_noise_mad(img_f32)
    K_est = float(np.clip(sigma_noise**2, K_EST_MIN, K_EST_MAX))

    K_strong = max(K_est * K_STRONG_MULTIPLIER, K_STRONG_FLOOR)
    K_weak = max(K_est * K_WEAK_MULTIPLIER, K_WEAK_FLOOR)

    # 2. PSF Sigma (coupled to scale)
    psf_sigma = max(PSF_SIGMA_MIN, PSF_SIGMA_SCALE * scale)
    if psf_override is not None:
        psf_sigma = psf_override

    logger.info(
        "Wiener: sigma_noise=%.5f, K_est=%.4f, K_strong=%.4f, "
        "K_weak=%.4f, PSF_sigma=%.3f",
        sigma_noise, K_est, K_strong, K_weak, psf_sigma,
    )

    # 3. PSF in Frequency Domain (Fix #11)
    psf = _build_gaussian_psf(psf_sigma, H, W)
    H_fft = scipy.fft.fft2(psf, workers=-1)
    G_fft = scipy.fft.fft2(img_f32.astype(np.float64), workers=-1)

    H_conj = np.conj(H_fft)
    H_power = np.abs(H_fft)**2

    # 4. Dual-Band Wiener Filtering (Fix #12)
    # Strong restoration (flat regions)
    F_strong = (H_conj / (H_power + K_strong)) * G_fft
    result_strong = np.real(scipy.fft.ifft2(F_strong, workers=-1))
    del F_strong

    # Weak restoration (edge regions)
    F_weak = (H_conj / (H_power + K_weak)) * G_fft
    result_weak = np.real(scipy.fft.ifft2(F_weak, workers=-1))
    del F_weak, H_fft, G_fft, H_conj, H_power

    # 5. Spatial Domain Blending (Fix #12)
    edge_mask = _build_edge_mask(img_f32)

    # edge=1.0 -> use weak/safe, edge=0.0 -> use strong/sharp
    result = (
        edge_mask * result_weak
        + (1.0 - edge_mask) * result_strong
    ).astype(np.float32)

    del result_strong, result_weak, edge_mask

    elapsed = time.time() - t0
    logger.info("Wiener deconvolution complete (%.2fs)", elapsed)
    return result


def deconvolve_color(
    img_bgr: np.ndarray,
    scale: float,
    psf_override: Optional[float] = None,
) -> np.ndarray:
    """Apply Wiener deconvolution to a color image.

    Converts to YCrCb in float32, deconvolves only the Y (luminance) channel
    while maintaining sub-pixel float32 stacking precision, then converts back.

    Parameters
    ----------
    img_bgr : np.ndarray
        Input color image (BGR, float32, [0, 255]).
    scale : float
        Effective upscale factor.
    psf_override : float, optional
        Override PSF sigma.

    Returns
    -------
    np.ndarray
        Deconvolved color image (BGR, float32, [0, 255]).
    """
    # Normalize to [0.0, 1.0] for standard float32 color space conversion in OpenCV
    img_scaled = (img_bgr / 255.0).astype(np.float32)
    ycrcb_f32 = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2YCrCb)

    # Wiener deconvolution expects input in [0.0, 255.0] range
    y_scaled = ycrcb_f32[:, :, 0] * 255.0
    y_deconv = wiener_deconv(y_scaled, scale, psf_override)

    # Scale back to [0.0, 1.0] and re-insert
    ycrcb_f32[:, :, 0] = np.clip(y_deconv, 0.0, 255.0) / 255.0

    # Convert back to BGR and scale back to [0.0, 255.0]
    result_scaled = cv2.cvtColor(ycrcb_f32, cv2.COLOR_YCrCb2BGR)
    result = result_scaled * 255.0

    return result
