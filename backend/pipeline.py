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
)
from .alignment import (
    align_images_ecc,
    calculate_dither_quality_neff,
    select_reference_frame,
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

OPTICO_VERSION = "1.1.0"  # bump when cache format or algorithm changes


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
) -> np.ndarray:
    """Run the complete Optico MFSR pipeline.

    Parameters
    ----------
    images : list of np.ndarray
        Input burst images (BGR, uint8), at least 2 frames.
    config : OpticoConfig, optional
        Pipeline configuration. Uses defaults if None.
    output_path : str, optional
        If provided, saves the result to this path.  When the output is
        a JPEG file, pipeline parameters are embedded into EXIF metadata
        (requires piexif; silently skipped if not installed).
    image_paths : list[Path], optional
        File paths corresponding to `images`. Required for cache lookup
        and JPEG auto-detection.  If None, both are disabled.
    use_cache : bool
        If False, skip cache lookup and always recompute Phases 2-8.
    cache_dir : Path
        Root directory for the on-disk Drizzle cache.

    Returns
    -------
    np.ndarray
        Final high-resolution output (BGR, uint8).

    Notes
    -----
    psf_override unit: config.psf_override is interpreted as **HR pixels**.
    It is passed unchanged to deconvolve_color and then to wiener_deconv,
    which uses it as an absolute HR-pixel PSF sigma.
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

        # -- Phase 5: Dither Quality Assessment --
        logger.info("-- Phase 5: Dither Quality Assessment (N_eff entropy) --")
        dither_neff = calculate_dither_quality_neff(M_list)
        logger.info("  Dither N_eff (effective sub-pixel positions): %.2f", dither_neff)

        # -- Phase 6: Dynamic Foreground Masking --
        logger.info("-- Phase 6: Dynamic Foreground Masking --")
        weight_maps = calculate_dynamic_mask(
            images, M_list, ref_idx=ref_idx, config=config
        )

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

        # -- Phase 8: Drizzle Stacking --
        logger.info("-- Phase 8: Drizzle Stacking --")
        hr_image = drizzle_stack(
            images, M_list, weight_maps,
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

    # ------------------------------------------------------------------ #
    # Phase 9: Adaptive Wiener Deconvolution                              #
    # ------------------------------------------------------------------ #
    # Compute the effective PSF sigma in HR pixels for EXIF logging.
    if config.psf_override is not None:
        psf_sigma_hr_log = round(config.psf_override, 4)
    else:
        base_sigma = max(PSF_SIGMA_MIN, PSF_SIGMA_SCALE * final_scale)
        if config.jpeg_input:
            psf_sigma_hr_log = round(base_sigma * JPEG_PSF_SCALE_FACTOR, 4)
        else:
            psf_sigma_hr_log = round(base_sigma, 4)
        psf_override_hr = None

    noise_hf_fraction = (
        JPEG_NOISE_FLOOR_HIGH_FREQ_FRACTION
        if config.jpeg_input
        else NOISE_FLOOR_HIGH_FREQ_FRACTION
    )

    if not config.skip_deconv:
        if config.psf_override is not None:
            logger.info(
                "Phase 9: psf_override %.3f HR px (scale=%.2f)",
                config.psf_override, final_scale,
            )

        logger.info(
            "-- Phase 9: Adaptive Wiener Deconvolution (jpeg_input=%s) --",
            config.jpeg_input,
        )
        hr_image = deconvolve_color(
            hr_image,
            scale=final_scale,
            psf_override=config.psf_override,
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

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cv2.imwrite(output_path, result)
        logger.info("Saved result to %s", output_path)

        # Embed pipeline parameters into JPEG EXIF
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
        "--output", "-o", default="optico_output.jpg",
        help="Output file path (default: optico_output.jpg)",
    )
    parser.add_argument(
        "--scale", "-s", type=float, default=2.0,
        help="Target upscale factor (default: 2.0)",
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
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
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
    )
    if args.align_scale is not None:
        config_kwargs["align_scale"] = args.align_scale
    if args.psf_override is not None:
        config_kwargs["psf_override"] = args.psf_override
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
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        sys.exit(2)


if __name__ == "__main__":
    main()
