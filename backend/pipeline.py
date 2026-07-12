"""Optico MFSR Engine — End-to-End Pipeline.

Orchestrates the full MFSR processing chain:
  Phase 0 : JPEG vs RAW source detection
  Phase 1 : Load burst images
  Phase 2 : Coarse sub-pixel alignment (relative to frame 0)
  Phase 3 : Reference frame selection (Harmony Anchor)
  Phase 4 : Refined sub-pixel alignment (relative to reference frame)
  Phase 5 : Dither quality assessment (N_eff entropy metric)
  Phase 6 : Dynamic foreground masking
  Phase 7 : Pre-flight scale bounding
  Phase 8 : Drizzle multi-frame stacking
  Phase 9 : Adaptive Wiener deconvolution
  Phase 10: Final output + EXIF embedding

Phase 0 — JPEG vs RAW detection
--------------------------------
JPEG input has three distinct failure modes compared to RAW/PNG:
  1. 8×8 DCT inter-block edges bias ECC sub-pixel alignment (Phase 2)
  2. Quantisation PSF adds residual blur on top of optical PSF (Phase 9)
  3. JPEG spectral cutoff inflates noise_floor in plateau detection (Phase 9)

detect_jpeg_source() reads the first 2 bytes of each input file to check
for the JPEG SOI marker (0xFF 0xD8).  The result is stored in
config.jpeg_input and propagated to Phases 2 and 9.  Override
config.jpeg_input manually if auto-detection is incorrect.

Drizzle Cache
-------------
Phases 2-8 are expensive and deterministic — for a fixed burst set and
configuration they always produce the same hr_image. To speed up
repeated deconvolution experiments, the Phase 8 output is cached on
disk (see backend/cache.py). On subsequent runs with the same input
directory and identical Drizzle-affecting config, Phases 2-8 are
skipped entirely and execution resumes from Phase 9.

Cache is keyed on the SHA-256 of each input file's content plus the
relevant OpticoConfig fields (psf_override and skip_deconv are
excluded so deconv tweaks always hit the cache).

IMPORTANT: The cache key is computed AFTER Phase 0 resolves jpeg_input
(auto-detection or CLI flag).  This ensures the effective jpeg_input
value (which changes ECC filter size) is always part of the key.
Previously the key was computed before detection, so --jpeg and --raw
could silently return a cache entry built with the wrong ECC settings.

Use --no-cache to force a full reprocess even if a cache entry exists.
Use --cache-dir to specify a custom cache location (default: ~/.optico_cache).

EXIF output
-----------
When the output file is a JPEG, _write_exif() embeds pipeline parameters
into two EXIF fields using piexif:
  ImageDescription (0x010e): compact key=value string for quick inspection
  UserComment (0x9286):       full JSON blob for programmatic reading

Parameters written:
  optico_version, frames, lr_w, lr_h, hr_w, hr_h, final_scale, safe_cap,
  retained_ratio, dither_neff, jpeg_input, psf_sigma_hr, noise_hf_fraction,
  edge_taper_width, pixfrac, cache_hit, processing_time_s

If piexif is not installed or the output format is not JPEG, the step
is silently skipped (no error, no data loss).

psf_override unit convention
-----------------------------
The --psf-override CLI argument and OpticoConfig.psf_override field accept
the PSF sigma in **HR pixels**. run_pipeline passes the value unchanged to
deconvolve_color and wiener_deconv, whose override path consumes an absolute
HR-pixel sigma. Example: --psf-override 0.8 with --scale 2 applies a
0.8 HR-pixel PSF; it does not become 1.6 HR pixels.
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import math

from .constants import (
    OpticoConfig,
    JPEG_ECC_GAUSS_FILT_SIZE,
    ECC_GAUSS_FILT_SIZE,
    JPEG_PSF_SCALE_FACTOR,
    JPEG_NOISE_FLOOR_HIGH_FREQ_FRACTION,
    NOISE_FLOOR_HIGH_FREQ_FRACTION,
    PSF_SIGMA_MIN,
    PSF_SIGMA_SCALE,
    EDGE_TAPER_WIDTH,
    DEFAULT_PIXFRAC,
)
from .alignment import (
    align_images_ecc,
    calculate_dither_quality_neff,
    select_reference_frame,
    compute_regional_cc,
    build_regional_quality_maps,
)
from .masking import calculate_dynamic_mask, calculate_retained_ratio
from .preflight import resolve_final_scale
from .drizzle import drizzle_stack
from .deconvolution import deconvolve_color
from .cache import (
    DEFAULT_CACHE_DIR,
    compute_cache_key,
    load_drizzle_cache,
    save_drizzle_cache,
)

logger = logging.getLogger(__name__)

def _estimate_noise_mad(img_gray: np.ndarray) -> float:
    """Estimate noise standard deviation using Median Absolute Deviation (MAD)."""
    lap = cv2.Laplacian(img_gray, cv2.CV_32F, ksize=1)
    mad = np.median(np.abs(lap - np.median(lap)))
    # Correct factor for 3x3 Laplacian filter noise magnification (sqrt(20))
    sigma = float(1.4826 * mad / math.sqrt(20.0))
    return max(sigma, 1e-6)

OPTICO_VERSION = "1.1.0"  # bump when cache format or algorithm changes

# Sentinel value for target_scale: derive from valid frame count after alignment.
# > 10 valid frames → 2.4x, > 8 → 2.2x, > 6 → 2.0x, ≤ 6 → 1.4x fallback.
AUTO_SCALE = 0.0

DEFAULT_OUTPUT_DIR = "backend/output"


# ---------------------------------------------------------------------------
# Phase 0: JPEG vs RAW detection
# ---------------------------------------------------------------------------

def detect_jpeg_source(image_paths: list[Path]) -> bool:
    """Detect whether the burst images are JPEG-sourced.

    Reads the first 2 bytes of each file and checks for the JPEG SOI
    marker (0xFF 0xD8).  Returns True if the majority of files are JPEG.

    JPEG input requires three downstream adjustments:
      - Phase 2: larger ECC Gaussian filter to suppress DCT block edges
      - Phase 9: enlarged PSF sigma for quantisation blur
      - Phase 9: lower noise-floor scan range to avoid JPEG spectral cutoff

    Parameters
    ----------
    image_paths : list[Path]
        File paths of the burst images.

    Returns
    -------
    bool
        True if images are likely JPEG-sourced.
    """
    if not image_paths:
        return False
    jpeg_count = 0
    for p in image_paths:
        try:
            with open(p, "rb") as f:
                header = f.read(2)
            if header == b"\xff\xd8":
                jpeg_count += 1
        except OSError:
            pass
    result = jpeg_count > len(image_paths) / 2
    logger.info(
        "Phase 0: source detection — %d/%d files are JPEG → jpeg_input=%s",
        jpeg_count, len(image_paths), result,
    )
    return result


def load_burst_images(input_dir: str) -> tuple[list[np.ndarray], list[Path]]:
    """Load burst images from a directory.

    Supports JPEG, PNG, TIFF. Images are sorted alphabetically.
    All images must have the same dimensions.

    Parameters
    ----------
    input_dir : str
        Path to directory containing burst images.

    Returns
    -------
    tuple[list[np.ndarray], list[Path]]
        Loaded images (BGR, uint8) and their corresponding file paths.

    Raises
    ------
    FileNotFoundError
        If the directory doesn't exist or contains no images.
    ValueError
        If images have inconsistent dimensions.
    """
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    extensions = ('*.jpg', '*.jpeg', '*.png', '*.tiff', '*.tif', '*.bmp')
    image_files: list[Path] = []
    for ext in extensions:
        image_files.extend(input_path.glob(ext))
        image_files.extend(input_path.glob(ext.upper()))

    image_files = sorted(set(image_files))

    if not image_files:
        raise FileNotFoundError(
            f"No image files found in {input_dir}. "
            f"Supported formats: {extensions}"
        )

    images: list[np.ndarray] = []
    loaded_paths: list[Path] = []
    ref_shape = None
    for f in image_files:
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("Failed to read %s, skipping", f)
            continue
        if ref_shape is None:
            ref_shape = img.shape
        elif img.shape != ref_shape:
            raise ValueError(
                f"Image {f.name} has shape {img.shape}, "
                f"expected {ref_shape}"
            )
        images.append(img)
        loaded_paths.append(f)

    if not images:
        raise FileNotFoundError(
            f"No valid images loaded from {input_dir}"
        )

    logger.info(
        "Loaded %d images (%dx%d) from %s",
        len(images), ref_shape[1], ref_shape[0], input_dir,
    )
    return images, loaded_paths


# ---------------------------------------------------------------------------
# EXIF helper
# ---------------------------------------------------------------------------

def _write_exif(
    output_path: str,
    exif_params: dict,
) -> None:
    """Embed pipeline parameters into the output JPEG's EXIF metadata.

    Writes to two EXIF fields:
      - ImageDescription (0x010e): compact ``key=value; ...`` string
        suitable for quick inspection in any image viewer.
      - UserComment (0x9286): full JSON blob for programmatic reading.
        Prefixed with the 8-byte ASCII charset header required by the
        EXIF spec (``ASCII\x00\x00\x00``).

    The function is a no-op (silent) if:
      - piexif is not installed
      - the output file is not a JPEG
      - the file cannot be read/written after saving

    Parameters
    ----------
    output_path : str
        Path to the already-saved JPEG output file.
    exif_params : dict
        Dictionary of pipeline parameters to embed.  All values must be
        JSON-serialisable.
    """
    suffix = Path(output_path).suffix.lower()
    if suffix not in (".jpg", ".jpeg"):
        return

    try:
        import piexif
    except ImportError:
        logger.debug("piexif not installed — skipping EXIF write")
        return

    try:
        # Build compact description string
        desc_parts = [
            f"optico={exif_params.get('optico_version', '?')}",
            f"scale={exif_params.get('final_scale', '?')}",
            f"frames={exif_params.get('frames', '?')}",
            f"jpeg_in={exif_params.get('jpeg_input', '?')}",
            f"psf_hr={exif_params.get('psf_sigma_hr', '?')}",
            f"neff={exif_params.get('dither_neff', '?')}",
            f"retained={exif_params.get('retained_ratio', '?')}",
            f"cache={exif_params.get('cache_hit', '?')}",
            f"t={exif_params.get('processing_time_s', '?')}s",
        ]
        description = "; ".join(desc_parts)

        # Full JSON blob with ASCII charset header (8 bytes) per EXIF spec
        json_str = json.dumps(exif_params, ensure_ascii=True)
        user_comment = b"ASCII\x00\x00\x00" + json_str.encode("ascii", errors="replace")

        # Load existing EXIF or start fresh
        try:
            exif_dict = piexif.load(output_path)
        except Exception:
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}}

        exif_dict["0th"][piexif.ImageIFD.ImageDescription] = description.encode("ascii", errors="replace")
        exif_dict["Exif"][piexif.ExifIFD.UserComment] = user_comment

        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, output_path)

        logger.info(
            "EXIF written to %s (%d bytes in UserComment)",
            output_path, len(user_comment),
        )
    except Exception as exc:
        logger.warning("EXIF write failed (non-fatal): %s", exc)


def _copy_exif_from_source(source_path: Path, output_path: str) -> None:
    """Copy all EXIF metadata from the original reference frame to the output JPEG.

    This preserves camera metadata (focal length, shutter speed, ISO, GPS, etc.)
    so the output behaves like a normal photograph in any viewer or DAM system.
    Called unconditionally; the Optico-specific EXIF params are written
    separately and only when ``debug=True``.

    Parameters
    ----------
    source_path : Path
        Reference frame whose EXIF to copy.
    output_path : str
        Already-saved output JPEG to insert EXIF into.
    """
    if Path(output_path).suffix.lower() not in (".jpg", ".jpeg"):
        return
    try:
        import piexif
    except ImportError:
        logger.debug("piexif not installed — skipping source EXIF copy")
        return
    try:
        src_exif = piexif.load(str(source_path))
        exif_bytes = piexif.dump(src_exif)
        piexif.insert(exif_bytes, output_path)
        logger.info(
            "EXIF copied from %s → %s",
            source_path.name, Path(output_path).name,
        )
    except Exception as exc:
        logger.warning("Source EXIF copy failed (non-fatal): %s", exc)


def _auto_scale_from_frame_count(n_valid: int) -> float:
    """Derive the optimal MFSR upscale factor from the number of valid frames.

    Based on information-theoretic frame diversity requirements:
      > 10 valid frames → 2.4x  (138 MP from 24 MP sensor)
      >  8 valid frames → 2.2x
      >  6 valid frames → 2.0x
      ≤  6 valid frames → 1.4x  (safe minimum for meaningful SR)

    Parameters
    ----------
    n_valid : int
        Number of frames that passed alignment and CC quality filtering.

    Returns
    -------
    float
        Recommended upscale factor.
    """
    if n_valid > 10:
        return 2.4
    if n_valid > 8:
        return 2.2
    if n_valid > 6:
        return 2.0
    return 1.4


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    images: list[np.ndarray],
    config: Optional[OpticoConfig] = None,
    output_path: Optional[str] = None,
    image_paths: Optional[list[Path]] = None,
    use_cache: bool = True,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    debug: bool = False,
) -> np.ndarray:
    """Run the complete Optico MFSR pipeline.

    Parameters
    ----------
    images : list of np.ndarray
        Input burst images (BGR, uint8), at least 2 frames.
    config : OpticoConfig, optional
        Pipeline configuration. Uses defaults if None.
    output_path : str, optional
        Output file path.  When ``None`` and ``image_paths`` is provided,
        the path is auto-derived from the reference frame's filename and
        saved to ``backend/output/<refname>.jpg``.
        Original camera EXIF is always copied from the reference frame.
        Optico pipeline parameters are written into EXIF only when
        ``debug=True``.
    image_paths : list[Path], optional
        File paths corresponding to `images`. Required for cache lookup
        and JPEG auto-detection.  If None, both are disabled.
    use_cache : bool
        If False, skip cache lookup and always recompute Phases 2-8.
    cache_dir : Path
        Root directory for the on-disk Drizzle cache.
    debug : bool
        When True, write Optico pipeline parameters (scale, PSF, frames…)
        into the output JPEG's EXIF UserComment field.  When False (default),
        only the original camera EXIF from the reference frame is kept.

    Returns
    -------
    np.ndarray
        Final high-resolution output (BGR, uint8).

    Notes
    -----
    psf_override unit: config.psf_override is interpreted as **HR pixels**.
    It is passed unchanged to deconvolve_color and then to wiener_deconv,
    which uses it as an absolute HR-pixel PSF sigma.

    target_scale == AUTO_SCALE (0.0): derive the scale from the valid frame
    count after alignment — >10→2.4x, >8→2.2x, >6→2.0x, else→1.4x.
    """
    if config is None:
        config = OpticoConfig()

    if len(images) < 2:
        raise ValueError(f"Need at least 2 frames, got {len(images)}")

    t_start = time.time()
    n = len(images)
    lr_h, lr_w = images[0].shape[:2]


    logger.info("=" * 60)
    logger.info("Optico MFSR Pipeline v%s", OPTICO_VERSION)
    logger.info(
        "  Frames: %d | Resolution: %dx%d | Target scale: %.1fx",
        n, lr_w, lr_h, config.target_scale,
    )
    logger.info("=" * 60)

    # ------------------------------------------------------------------ #
    # Phase 0: JPEG vs RAW detection                                      #
    # ------------------------------------------------------------------ #
    if config.jpeg_input is None:
        if image_paths is not None:
            jpeg_detected = detect_jpeg_source(image_paths)
        else:
            jpeg_detected = False
            logger.info(
                "Phase 0: no image_paths provided — assuming RAW/PNG input."
            )
        # Store detection result in a local copy of config so the
        # original caller's config object is not mutated.
        from dataclasses import replace
        config = replace(config, jpeg_input=jpeg_detected)
    else:
        jpeg_detected = config.jpeg_input
        logger.info(
            "Phase 0: jpeg_input manually set to %s", jpeg_detected
        )

    # Apply JPEG-specific ECC filter size if not manually overridden
    if jpeg_detected and config.ecc_gauss_filt_size == ECC_GAUSS_FILT_SIZE:
        from dataclasses import replace
        config = replace(config, ecc_gauss_filt_size=JPEG_ECC_GAUSS_FILT_SIZE)
        logger.info(
            "Phase 0: JPEG input — ECC Gaussian filter: %d → %d px",
            ECC_GAUSS_FILT_SIZE, JPEG_ECC_GAUSS_FILT_SIZE,
        )

    # ------------------------------------------------------------------ #
    # Drizzle Cache Lookup (Phases 2-8)                                   #
    # NOTE: cache key computed AFTER Phase 0 so that the resolved        #
    # jpeg_input value (True/False, never None) is part of the key.      #
    # Previously computed before detection → --jpeg/--raw could silently #
    # hit a cache entry built with the wrong ECC filter size.            #
    # ------------------------------------------------------------------ #
    drizzle_result = None
    cache_key = None
    cache_hit = False
    ref_idx = 0  # default; overridden in Phase 3 when running full pipeline

    if use_cache and image_paths is not None:
        logger.info("-- Cache: Computing cache key (hashing input files)... --")
        cache_key = compute_cache_key(image_paths, config)
        logger.info("  Cache key: %s...", cache_key[:16])
        drizzle_result = load_drizzle_cache(cache_key, cache_dir=cache_dir)

    if drizzle_result is not None:
        hr_image = drizzle_result["hr_image"]
        final_scale = drizzle_result["final_scale"]
        retained_ratio = drizzle_result["retained_ratio"]
        dither_neff = drizzle_result["dither_quality"]
        safe_cap = drizzle_result["safe_cap"]
        cache_hit = True
        logger.info(
            "-- Phases 2-8: SKIPPED (cache hit) | "
            "scale=%.2f retained=%.3f dither_neff=%.2f --",
            final_scale, retained_ratio, dither_neff,
        )
    else:
        if use_cache and image_paths is not None:
            logger.info("-- Cache: MISS — running full pipeline --")
        else:
            logger.info("-- Cache: disabled for this run --")

        # -- Phase 2: Coarse Alignment --
        logger.info("Selecting initial coarse reference frame based on sharpness...")
        sharpness_scores = []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sharpness_scores.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        coarse_ref_idx = int(np.argmax(sharpness_scores))
        logger.info(
            "  Coarse reference: frame %d (sharpness=%.1f)",
            coarse_ref_idx, sharpness_scores[coarse_ref_idx],
        )

        logger.info("-- Phase 2: Coarse ECC Alignment --")
        M_list_initial, cc_list = align_images_ecc(
            images, ref_idx=coarse_ref_idx, config=config
        )

        # -- Phase 3: Reference Frame Selection (Harmony Anchor) --
        logger.info("-- Phase 3: Harmony Anchor Selection --")
        ref_idx = select_reference_frame(images, M_list_initial)

        # -- Phase 4: Refined Alignment --
        if ref_idx != coarse_ref_idx:
            logger.info("-- Phase 4: Refined ECC Alignment --")
            M_list, cc_list = align_images_ecc(
                images, ref_idx=ref_idx, config=config
            )
        else:
            logger.info("-- Phase 4: Skipping Re-alignment (Coarse reference is optimal) --")
            M_list = M_list_initial

        # -- Phase 4.5: Frame Quality Filtering --
        min_cc = getattr(config, 'frame_cc_threshold', 0.0)
        before = sum(1 for m in M_list if m is not None)

        if min_cc == 0.0:
            # Enable Adaptive MAD Outlier Rejection by default
            valid_cc = np.array([cc for cc, m in zip(cc_list, M_list) if m is not None])
            if len(valid_cc) > 2:
                median_cc = np.median(valid_cc)
                mad = np.median(np.abs(valid_cc - median_cc))
                sigma = 1.4826 * mad
                # Strict Z-score threshold = 1.2 to eliminate alignment residuals
                z_scores = (valid_cc - median_cc) / (sigma + 1e-9)

                idx_valid = 0
                for i, m in enumerate(M_list):
                    if m is not None:
                        z = z_scores[idx_valid]
                        cc_val = cc_list[i]
                        # Ref frame is always kept
                        if i != ref_idx and (z < -1.2 or cc_val < 0.5):
                            M_list[i] = None
                            cc_list[i] = 0.0
                        idx_valid += 1
                after = sum(1 for m in M_list if m is not None)
                logger.info(
                    "-- Phase 4.5: Adaptive MAD Filter (threshold=1.2, median=%.4f, sigma=%.6f) -- kept %d/%d frames --",
                    median_cc, sigma, after, before,
                )
        elif min_cc > 0.0:
            # Downward compatibility: Hard gap filtering
            valid_cc = [cc for cc, m in zip(cc_list, M_list) if m is not None]
            if valid_cc:
                cc_ref = max(valid_cc)  # best frame CC
                abs_threshold = cc_ref - min_cc
                for i, (cc, m) in enumerate(zip(cc_list, M_list)):
                    if m is not None and i != ref_idx and cc < abs_threshold:
                        M_list[i] = None
                        cc_list[i] = 0.0
                after = sum(1 for m in M_list if m is not None)
                logger.info(
                    "-- Phase 4.5: Hard CC Gap Filter (min_cc_gap=%.4f, abs_threshold=%.4f) "
                    "-- kept %d/%d frames --",
                    min_cc, abs_threshold, after, before,
                )

        # -- Auto-scale: derive target scale from valid frame count --
        if config.target_scale == AUTO_SCALE:
            valid_now = sum(1 for m in M_list if m is not None)
            auto_s = _auto_scale_from_frame_count(valid_now)
            from dataclasses import replace as dc_replace
            config = dc_replace(config, target_scale=auto_s)
            logger.info(
                "-- Auto-scale: %d valid frames → %.1fx --",
                valid_now, auto_s,
            )

        # -- Phase 5: Dither Quality Assessment --
        logger.info("-- Phase 5: Dither Quality Assessment (N_eff entropy) --")
        dither_neff = calculate_dither_quality_neff(M_list)
        logger.info("  Dither N_eff (effective sub-pixel positions): %.2f", dither_neff)

        # -- Phase 6: Dynamic Foreground Masking --
        logger.info("-- Phase 6: Dynamic Foreground Masking --")
        weight_maps = calculate_dynamic_mask(
            images, M_list, ref_idx=ref_idx, config=config
        )

        # -- Phase 6.5: Regional CC Spatial Quality Weighting --
        # Compute per-frame CC for center + 4 corner regions.
        # For frames with rotation-induced corner misalignment, this builds
        # a smooth spatial weight map that tapers near the corners, so those
        # frames contribute less where they are poorly aligned.
        ref_gray_full = cv2.cvtColor(images[ref_idx], cv2.COLOR_BGR2GRAY)
        regional_cc_list = compute_regional_cc(
            ref_gray_full, images, M_list,
            corner_frac=0.20, center_frac=0.30,
        )
        lr_h_local, lr_w_local = images[0].shape[:2]
        quality_maps = build_regional_quality_maps(
            regional_cc_list, lr_h_local, lr_w_local, corner_frac=0.20,
        )
        # Log regional CC summary
        for i, rcc in enumerate(regional_cc_list):
            if M_list[i] is not None:
                logger.debug(
                    "Frame %d regional CC: center=%.4f TL=%.4f TR=%.4f "
                    "BL=%.4f BR=%.4f corner_min=%.4f",
                    i, rcc['cc_center'], rcc['cc_tl'], rcc['cc_tr'],
                    rcc['cc_bl'], rcc['cc_br'], rcc['cc_corner_min'],
                )
        # Multiply motion mask by spatial quality map
        weight_maps = [
            (wm * qm).astype(np.float32)
            for wm, qm in zip(weight_maps, quality_maps)
        ]
        logger.info("-- Phase 6.5: Regional CC spatial weighting applied --")

        # -- Phase 7: Pre-flight Scale Bounding --
        logger.info("-- Phase 7: Pre-flight Scale Bounding --")
        retained_ratio = calculate_retained_ratio(weight_maps)
        valid_count = sum(1 for m in M_list if m is not None)

        final_scale, safe_cap = resolve_final_scale(
            target_scale=config.target_scale,
            num_frames=valid_count,
            retained_ratio=retained_ratio,
            dither_neff=dither_neff,
            cc_scores=cc_list,
            config=config,
        )

        # -- Conjugate Drizzle Pre-compensation: Scale-Adaptive Pixfrac --
        # If pixfrac is not manually overridden, adjust it inversely with scale
        # to ensure that the physical footprint of Drizzle points projected onto
        # the HR canvas remains constant, matching the fixed post-deconv PSF.
        if config.pixfrac == DEFAULT_PIXFRAC:
            adaptive_pixfrac = np.clip(0.70 * (2.0 / final_scale), 0.40, 1.00)
            from dataclasses import replace as dc_replace
            config = dc_replace(config, pixfrac=float(adaptive_pixfrac))
            logger.info(
                "-- Drizzle Pre-compensation: scale=%.2f → adaptive pixfrac=%.2f --",
                final_scale, config.pixfrac,
            )

        # -- Drizzle Pre-emphasis (LR Data-Side Pre-filtering) --
        # Pre-emphasize high frequencies in LR space for JPEG inputs to compensate blur early.
        # Spatially adaptive using a Sobel-derived edge mask to avoid magnifying flat-region noise.
        # This keeps deconv PSF safe at a unified 0.88, avoiding Ringing and Clamping loss.
        drizzle_images = images
        if config.jpeg_input:
            logger.info("-- Drizzle Pre-emphasis: Applying Spatially Adaptive LR high-pass pre-filter (alpha=0.55) --")
            drizzle_images = []
            for img in images:
                img_f = img.astype(np.float32)
                blur = cv2.GaussianBlur(img_f, (3, 3), 0.8)
                hp = img_f - blur
                
                # Compute local edge mask in LR space
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                sigma_noise = _estimate_noise_mad(gray)
                sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
                grad_mag = np.sqrt(sobelx**2 + sobely**2)
                threshold = max(3.5 * sigma_noise, 3.0)
                edge_mask = np.clip((grad_mag - threshold) / threshold, 0.0, 1.0)
                edge_mask_sm = cv2.GaussianBlur(edge_mask, (5, 5), 1.5)
                edge_mask_3d = np.expand_dims(edge_mask_sm, axis=2)
                
                # Apply spatially adaptive pre-emphasis
                emp = np.clip(img_f + 0.55 * edge_mask_3d * hp, 0, 255).astype(np.uint8)
                drizzle_images.append(emp)

        # -- Phase 8: Drizzle Stacking --
        logger.info("-- Phase 8: Drizzle Stacking --")
        hr_image = drizzle_stack(
            drizzle_images, M_list, weight_maps,
            scale=final_scale, ref_idx=ref_idx, config=config,
        )

        if use_cache and cache_key is not None and image_paths is not None:
            save_drizzle_cache(
                cache_key=cache_key,
                hr_image=hr_image,
                final_scale=final_scale,
                retained_ratio=retained_ratio,
                dither_quality=dither_neff,
                safe_cap=safe_cap,
                image_paths=image_paths,
                config=config,
                cache_dir=cache_dir,
            )

    PHYSICAL_PSF_BASE = config.psf_base

    if config.psf_override is not None:
        effective_psf = config.psf_override
    else:
        effective_psf = PHYSICAL_PSF_BASE * final_scale

    psf_sigma_hr_log = round(effective_psf, 4)

    noise_hf_fraction = (
        JPEG_NOISE_FLOOR_HIGH_FREQ_FRACTION
        if config.jpeg_input
        else NOISE_FLOOR_HIGH_FREQ_FRACTION
    )

    if not config.skip_deconv:
        logger.info(
            "Phase 9: Lens PSF Anchor %.3f HR px (scale=%.2f, jpeg_input=%s)",
            effective_psf, final_scale, config.jpeg_input,
        )

        logger.info(
            "-- Phase 9: Adaptive Wiener Deconvolution (jpeg_input=%s) --",
            config.jpeg_input,
        )
        hr_image = deconvolve_color(
            hr_image,
            scale=final_scale,
            psf_override=effective_psf,
            jpeg_input=bool(config.jpeg_input),
        )


    else:
        logger.info("-- Phase 9: Skipped Deconvolution (skip_deconv=True) --")

    # -- Phase 10: Final Output --
    logger.info("-- Phase 10: Final Output --")
    result = np.clip(hr_image, 0, 255).astype(np.uint8)

    hr_h, hr_w = result.shape[:2]
    elapsed = round(time.time() - t_start, 2)

    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info(
        "  Output: %dx%d (%.2fx from %dx%d)",
        hr_w, hr_h, final_scale, lr_w, lr_h,
    )
    logger.info(
        "  Safe cap: %.2f | Retained ratio: %.3f | jpeg_input: %s",
        safe_cap, retained_ratio, config.jpeg_input,
    )
    logger.info("  Dither N_eff: %.2f", dither_neff)
    logger.info("  PSF sigma (HR px): %.4f", psf_sigma_hr_log)
    logger.info("  Total time: %.1fs", elapsed)
    logger.info("=" * 60)

    # Auto-derive output path from reference frame filename when not given
    if output_path is None and image_paths is not None:
        ref_path = image_paths[ref_idx]
        out_dir = Path(DEFAULT_OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / (ref_path.stem + ".jpg"))
        logger.info("Auto output path: %s", output_path)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cv2.imwrite(output_path, result)
        logger.info("Saved result to %s", output_path)

        # Always copy original camera EXIF from the reference frame
        if image_paths is not None:
            _copy_exif_from_source(image_paths[ref_idx], output_path)

        # Write Optico pipeline parameters into EXIF only in debug mode
        if debug:
            exif_params = {
            "optico_version": OPTICO_VERSION,
            "frames": n,
            "lr_w": lr_w,
            "lr_h": lr_h,
            "hr_w": hr_w,
            "hr_h": hr_h,
            "final_scale": round(final_scale, 4),
            "safe_cap": round(safe_cap, 4),
            "retained_ratio": round(retained_ratio, 4),
            "dither_neff": round(dither_neff, 4),
            "jpeg_input": bool(config.jpeg_input),
            "psf_sigma_hr": psf_sigma_hr_log,
            "noise_hf_fraction": noise_hf_fraction,
            "edge_taper_width": EDGE_TAPER_WIDTH,
            "pixfrac": config.pixfrac,
            "cache_hit": cache_hit,
            "processing_time_s": elapsed,
        }
            _write_exif(output_path, exif_params)

    return result


def main() -> None:
    """CLI entry point for Optico."""
    parser = argparse.ArgumentParser(
        description="Optico -- Multi-Frame Super-Resolution Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m backend.pipeline --input ./burst --output result.jpg\n"
            "  python -m backend.pipeline --input ./burst --scale 2.5\n"
            "  python -m backend.pipeline --input ./burst --no-deconv\n"
            "  python -m backend.pipeline --input ./burst --no-cache\n"
            "  python -m backend.pipeline --input ./burst --jpeg\n"
            "  python -m backend.pipeline --input ./burst --raw\n"
            "  python -m backend.pipeline --input ./burst --jpeg --psf-override 0.8\n"
            "  python -m backend.pipeline --input ./burst --raw --psf-override 0.8\n"
        ),
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Directory containing burst images",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help=(
            "Output file path. When omitted, the output is saved to "
            f"backend/output/<ref_frame_name>.jpg (derived from the "
            "reference frame filename)."
        ),
    )
    parser.add_argument(
        "--scale", "-s", type=float, default=AUTO_SCALE,
        help=(
            "Target upscale factor. When omitted (default: auto), the factor "
            "is determined by the number of valid frames after alignment: "
            ">10→2.4x, >8→2.2x, >6→2.0x, ≤6→1.4x."
        ),
    )
    parser.add_argument(
        "--pixfrac", type=float, default=0.7,
        help="Drizzle pixel fraction 0-1 (default: 0.7)",
    )
    parser.add_argument(
        "--chunks", type=int, default=8,
        help="Number of memory chunks (default: 8)",
    )
    parser.add_argument(
        "--no-deconv", action="store_true",
        help="Skip Wiener deconvolution",
    )
    parser.add_argument(
        "--align-scale", type=float, default=None,
        help="Override ECC alignment downscale factor (default: auto)",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable Drizzle cache; always recompute Phases 2-8",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help=f"Cache directory (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--psf-override", type=float, default=None,
        metavar="SIGMA_HR",
        help=(
            "Override the Wiener PSF sigma in HR pixels. The value is passed "
            "directly to deconvolution and is independent of final scale. "
            "When set, the JPEG PSF scale factor (×1.10) is skipped; use "
            "--jpeg or --raw independently to control noise-floor behaviour. "
            "Example: --psf-override 0.8 with --scale 2 applies a "
            "0.8 HR-pixel PSF to the deconvolution."
        ),
    )
    parser.add_argument(
        "--psf-base", type=float, default=0.63,
        help="Wiener deconvolution base PSF scale factor. effective_psf = psf_base * Scale. (default: 0.63)",
    )

    # JPEG / RAW override flags
    jpeg_group = parser.add_mutually_exclusive_group()
    jpeg_group.add_argument(
        "--jpeg", action="store_true",
        help="Force JPEG-mode processing (ECC filter=7, PSF×1.35, noise HF=0.60)",
    )
    jpeg_group.add_argument(
        "--raw", action="store_true",
        help="Force RAW/PNG-mode processing regardless of file extension",
    )

    parser.add_argument(
        "--kernel-mode", choices=["lanczos2", "nearest", "bilinear", "lanczos4", "box", "lanczos2_clamped", "box_supersample"],
        default=None,
        help="Drizzle kernel mode (default: lanczos2)",
    )
    parser.add_argument(
        "--motion-mode", choices=["translation", "affine"],
        default="affine",
        help="ECC alignment motion model (default: affine)",
    )
    parser.add_argument(
        "--min-cc", type=float, default=0.0, metavar="GAP",
        help="Drop frames whose CC score is more than GAP below the best frame's CC. "
             "0.0 = keep all (default). Typical: 0.003-0.010. "
             "Higher = stricter, fewer frames, sharper corners.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help=(
            "Write Optico pipeline parameters (scale, PSF, frames, timing…) "
            "into the output JPEG EXIF UserComment field. When omitted (default), "
            "only the original camera EXIF from the reference frame is preserved."
        ),
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config_kwargs: dict = dict(
        target_scale=args.scale,
        pixfrac=args.pixfrac,
        num_chunks=args.chunks,
        skip_deconv=args.no_deconv,
        psf_base=args.psf_base,
    )
    if args.align_scale is not None:
        config_kwargs["align_scale"] = args.align_scale
    if args.psf_override is not None:
        config_kwargs["psf_override"] = args.psf_override
    if args.kernel_mode is not None:
        config_kwargs["kernel_mode"] = args.kernel_mode
    if args.motion_mode is not None:
        config_kwargs["ecc_motion_mode"] = args.motion_mode
    if args.min_cc > 0.0:
        config_kwargs["frame_cc_threshold"] = args.min_cc
    if args.jpeg:
        config_kwargs["jpeg_input"] = True
    elif args.raw:
        config_kwargs["jpeg_input"] = False

    config = OpticoConfig(**config_kwargs)

    cache_dir = Path(args.cache_dir) if args.cache_dir else DEFAULT_CACHE_DIR

    try:
        images, image_paths = load_burst_images(args.input)
        run_pipeline(
            images,
            config=config,
            output_path=args.output,
            image_paths=image_paths,
            use_cache=not args.no_cache,
            cache_dir=cache_dir,
            debug=args.debug,
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        sys.exit(2)


if __name__ == "__main__":
    main()
