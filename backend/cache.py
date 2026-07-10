"""Optico MFSR Engine — Drizzle Result Cache.

Provides a content-addressed on-disk cache for the Drizzle stacking
result (Phase 8 output), so that repeated deconvolution experiments
on the same burst set skip the expensive alignment + stacking phases.

Cache key
---------
The cache key is a SHA-256 hex digest computed from:
  1. The sorted list of input image file paths and their SHA-256 digests.
  2. The OpticoConfig fields that affect Phases 2-8:
       target_scale, pixfrac, num_chunks, align_scale, max_offset,
       ecc_iterations, ecc_epsilon, ecc_gauss_filt_size,
       noise_gain, noise_offset, gradient_weight,
       bg_threshold, subj_threshold, optical_decay,
       jpeg_input, kernel_mode

Fields that only affect Phase 9 (psf_override, skip_deconv) are
excluded so that deconvolution parameter changes always hit the cache.

Bug fix (2026-07): jpeg_input and kernel_mode were previously missing
from the cache key, causing stale cache hits when switching --jpeg/--raw
or changing DRIZZLE_KERNEL_MODE.  Both fields are now included.

Cache layout
------------
  <cache_dir>/<cache_key>/
      drizzle.npz          — hr_image (float32), final_scale (scalar),
                             retained_ratio, dither_quality, safe_cap
      meta.json            — human-readable provenance (paths, config)

Default cache_dir: ~/.optico_cache
"""
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

from .constants import OpticoConfig

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".optico_cache"


def _sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _config_cache_fields(config: OpticoConfig) -> dict:
    """Extract only the OpticoConfig fields that affect Phases 2-8.

    Includes jpeg_input (affects Phase 2 ECC filter size and Phase 8
    Drizzle kernel assumptions) and kernel_mode (Lanczos-2 vs box).
    Both were previously omitted, causing silent stale-cache bugs.
    """
    return {
        "target_scale": config.target_scale,
        "pixfrac": config.pixfrac,
        "num_chunks": config.num_chunks,
        "align_scale": config.align_scale,
        "max_offset": config.max_offset,
        "ecc_iterations": config.ecc_iterations,
        "ecc_epsilon": config.ecc_epsilon,
        "ecc_gauss_filt_size": config.ecc_gauss_filt_size,
        "noise_gain": config.noise_gain,
        "noise_offset": config.noise_offset,
        "gradient_weight": config.gradient_weight,
        "bg_threshold": config.bg_threshold,
        "subj_threshold": config.subj_threshold,
        "optical_decay": config.optical_decay,
        # --- fields added in 2026-07 bug fix ---
        "jpeg_input": config.jpeg_input,   # affects ECC filter size
        "kernel_mode": config.kernel_mode, # lanczos2 vs box
    }


def compute_cache_key(
    image_paths: list[Path],
    config: OpticoConfig,
) -> str:
    """Compute a content-addressed cache key.

    Parameters
    ----------
    image_paths : list[Path]
        Sorted list of input image file paths.
    config : OpticoConfig
        Pipeline configuration.  Must already have jpeg_input resolved
        (i.e., Phase 0 detection complete) before calling this function,
        otherwise jpeg_input=None will be hashed and the key will not
        match a subsequent run where jpeg_input was auto-detected.

    Returns
    -------
    str
        64-character hex SHA-256 digest.
    """
    h = hashlib.sha256()

    # Hash each file's content (order matters — already sorted by caller)
    for p in image_paths:
        file_digest = _sha256_file(p)
        # Include both the resolved absolute path and the file digest
        # so that renaming the directory does NOT bust the cache, but
        # replacing any image with different content DOES.
        entry = f"{p.name}:{file_digest}\n"
        h.update(entry.encode())

    # Hash the config fields that affect Drizzle output
    config_str = json.dumps(_config_cache_fields(config), sort_keys=True)
    h.update(config_str.encode())

    return h.hexdigest()


def load_drizzle_cache(
    cache_key: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Optional[dict]:
    """Load a cached Drizzle result.

    Parameters
    ----------
    cache_key : str
        64-char hex key from compute_cache_key.
    cache_dir : Path
        Root cache directory.

    Returns
    -------
    dict or None
        Dictionary with keys:
          hr_image      : np.ndarray (float32)
          final_scale   : float
          retained_ratio: float
          dither_quality: float
          safe_cap      : float
        Returns None if no valid cache entry exists.
    """
    entry_dir = cache_dir / cache_key
    npz_path = entry_dir / "drizzle.npz"

    if not npz_path.is_file():
        return None

    try:
        data = np.load(npz_path, allow_pickle=False)
        result = {
            "hr_image": data["hr_image"].astype(np.float32),
            "final_scale": float(data["final_scale"]),
            "retained_ratio": float(data["retained_ratio"]),
            "dither_quality": float(data["dither_quality"]),
            "safe_cap": float(data["safe_cap"]),
        }
        logger.info(
            "[Cache HIT] Loaded drizzle cache: key=%s... | "
            "scale=%.2f | shape=%s",
            cache_key[:12], result["final_scale"],
            result["hr_image"].shape,
        )
        return result
    except Exception as exc:
        logger.warning(
            "[Cache] Failed to load %s: %s — will recompute",
            npz_path, exc,
        )
        return None


def save_drizzle_cache(
    cache_key: str,
    hr_image: np.ndarray,
    final_scale: float,
    retained_ratio: float,
    dither_quality: float,
    safe_cap: float,
    image_paths: list[Path],
    config: OpticoConfig,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> None:
    """Persist a Drizzle result to the on-disk cache.

    Parameters
    ----------
    cache_key : str
        Key from compute_cache_key.
    hr_image : np.ndarray
        Drizzle output (float32).
    final_scale, retained_ratio, dither_quality, safe_cap : float
        Pipeline metadata needed to resume from Phase 9.
    image_paths : list[Path]
        Original input paths (stored in meta.json for provenance).
    config : OpticoConfig
        Pipeline config (stored in meta.json for provenance).
    cache_dir : Path
        Root cache directory.
    """
    entry_dir = cache_dir / cache_key
    entry_dir.mkdir(parents=True, exist_ok=True)

    npz_path = entry_dir / "drizzle.npz"
    np.savez_compressed(
        npz_path,
        hr_image=hr_image.astype(np.float32),
        final_scale=np.array(final_scale, dtype=np.float64),
        retained_ratio=np.array(retained_ratio, dtype=np.float64),
        dither_quality=np.array(dither_quality, dtype=np.float64),
        safe_cap=np.array(safe_cap, dtype=np.float64),
    )

    meta = {
        "cache_key": cache_key,
        "image_paths": [str(p) for p in image_paths],
        "config": _config_cache_fields(config),
        "final_scale": final_scale,
        "retained_ratio": retained_ratio,
        "dither_quality": dither_quality,
        "safe_cap": safe_cap,
        "hr_image_shape": list(hr_image.shape),
    }
    meta_path = entry_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info(
        "[Cache SAVE] key=%s... | %.1f MB | %s",
        cache_key[:12],
        npz_path.stat().st_size / 1024 / 1024,
        npz_path,
    )
