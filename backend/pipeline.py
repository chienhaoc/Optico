"""Optico MFSR Engine — End-to-End Pipeline.

Orchestrates the full MFSR processing chain:
  Phase 1 : Load burst images
  Phase 2 : Coarse sub-pixel alignment (relative to frame 0)
  Phase 3 : Reference frame selection (Harmony Anchor)
  Phase 4 : Refined sub-pixel alignment (relative to reference frame)
  Phase 5 : Dither quality assessment (2D circular statistics)
  Phase 6 : Dynamic foreground masking
  Phase 7 : Pre-flight scale bounding
  Phase 8 : Drizzle multi-frame stacking
  Phase 9 : Adaptive Wiener deconvolution
  Phase 10: Final output
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .constants import OpticoConfig
from .alignment import (
    align_images_ecc,
    calculate_dither_quality_2d,
    select_reference_frame,
)
from .masking import calculate_dynamic_mask, calculate_retained_ratio
from .preflight import resolve_final_scale
from .drizzle import drizzle_stack
from .deconvolution import deconvolve_color

logger = logging.getLogger(__name__)


def load_burst_images(input_dir: str) -> list[np.ndarray]:
    """Load burst images from a directory.

    Supports JPEG, PNG, TIFF. Images are sorted alphabetically.
    All images must have the same dimensions.

    Parameters
    ----------
    input_dir : str
        Path to directory containing burst images.

    Returns
    -------
    list of np.ndarray
        Loaded images (BGR, uint8).

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

    # Deduplicate and sort
    image_files = sorted(set(image_files))

    if not image_files:
        raise FileNotFoundError(
            f"No image files found in {input_dir}. "
            f"Supported formats: {extensions}"
        )

    images: list[np.ndarray] = []
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

    if not images:
        raise FileNotFoundError(
            f"No valid images loaded from {input_dir}"
        )

    logger.info(
        "Loaded %d images (%dx%d) from %s",
        len(images), ref_shape[1], ref_shape[0], input_dir,
    )
    return images


def run_pipeline(
    images: list[np.ndarray],
    config: Optional[OpticoConfig] = None,
    output_path: Optional[str] = None,
) -> np.ndarray:
    """Run the complete Optico MFSR pipeline.

    Parameters
    ----------
    images : list of np.ndarray
        Input burst images (BGR, uint8), at least 2 frames.
    config : OpticoConfig, optional
        Pipeline configuration. Uses defaults if None.
    output_path : str, optional
        If provided, saves the result to this path.

    Returns
    -------
    np.ndarray
        Final high-resolution output (BGR, uint8).
    """
    if config is None:
        config = OpticoConfig()

    if len(images) < 2:
        raise ValueError(f"Need at least 2 frames, got {len(images)}")

    t_start = time.time()
    n = len(images)
    lr_h, lr_w = images[0].shape[:2]

    logger.info("=" * 60)
    logger.info("Optico MFSR Pipeline")
    logger.info(
        "  Frames: %d | Resolution: %dx%d | Target scale: %.1fx",
        n, lr_w, lr_h, config.target_scale,
    )
    logger.info("=" * 60)

    # -- Phase 2: Coarse Alignment --
    # Find the sharpest frame first as the initial coarse reference to avoid aligning to a blurry or shaken Frame 0
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

    # -- Phase 4: Refined Alignment (re-align to the selected reference) --
    if ref_idx != coarse_ref_idx:
        logger.info("-- Phase 4: Refined ECC Alignment --")
        M_list, cc_list = align_images_ecc(
            images, ref_idx=ref_idx, config=config
        )
    else:
        logger.info("-- Phase 4: Skipping Re-alignment (Coarse reference is optimal) --")
        M_list = M_list_initial

    # -- Phase 5: Dither Quality Assessment --
    logger.info("-- Phase 5: Dither Quality Assessment --")
    dither_q = calculate_dither_quality_2d(M_list)
    logger.info("  Dither quality (2D circular stats): %.3f", dither_q)

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
        dither_quality=dither_q,
        cc_scores=cc_list,
        config=config,
    )

    # -- Phase 8: Drizzle Stacking --
    logger.info("-- Phase 8: Drizzle Stacking --")
    hr_image = drizzle_stack(
        images, M_list, weight_maps,
        scale=final_scale, ref_idx=ref_idx, config=config,
    )

    # -- Phase 9: Adaptive Wiener Deconvolution --
    if not config.skip_deconv:
        logger.info("-- Phase 9: Adaptive Wiener Deconvolution --")
        hr_image = deconvolve_color(
            hr_image, scale=final_scale,
            psf_override=config.psf_override,
        )
    else:
        logger.info("-- Phase 9: Skipped Deconvolution (skip_deconv=True) --")

    # -- Phase 10: Final Output --
    logger.info("-- Phase 10: Final Output --")
    result = np.clip(hr_image, 0, 255).astype(np.uint8)

    hr_h, hr_w = result.shape[:2]
    elapsed = time.time() - t_start

    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info(
        "  Output: %dx%d (%.2fx from %dx%d)",
        hr_w, hr_h, final_scale, lr_w, lr_h,
    )
    logger.info(
        "  Safe cap: %.2f | Retained ratio: %.3f",
        safe_cap, retained_ratio,
    )
    logger.info("  Dither quality: %.3f", dither_q)
    logger.info("  Total time: %.1fs", elapsed)
    logger.info("=" * 60)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cv2.imwrite(output_path, result)
        logger.info("Saved result to %s", output_path)

    return result


def main() -> None:
    """CLI entry point for Optico."""
    parser = argparse.ArgumentParser(
        description="Optico -- Multi-Frame Super-Resolution Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m backend.pipeline --input ./burst --output result.png\n"
            "  python -m backend.pipeline --input ./burst --scale 2.5\n"
            "  python -m backend.pipeline --input ./burst --no-deconv\n"
        ),
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Directory containing burst images",
    )
    parser.add_argument(
        "--output", "-o", default="optico_output.png",
        help="Output file path (default: optico_output.png)",
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
        "--align-scale", type=float, default=0.5,
        help="Downscale factor for ECC alignment (default: 0.5)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = OpticoConfig(
        target_scale=args.scale,
        pixfrac=args.pixfrac,
        num_chunks=args.chunks,
        skip_deconv=args.no_deconv,
        align_scale=args.align_scale,
    )

    try:
        images = load_burst_images(args.input)
        run_pipeline(images, config=config, output_path=args.output)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        sys.exit(2)


if __name__ == "__main__":
    main()
