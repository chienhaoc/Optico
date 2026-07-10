# Optico

**Multi-Frame Super-Resolution (MFSR) engine for handheld burst photography.**

Optico fuses multiple frames of a burst into a single high-resolution image using sub-pixel registration, dynamic masking, Drizzle stacking, and adaptive Wiener deconvolution.

---

## Quick Start

```bash
pip install -r requirements.txt
python -m backend.pipeline --input ./burst --output result.png
```

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `--input` / `-i` | *(required)* | Directory of burst images |
| `--output` / `-o` | `optico_output.png` | Output file path |
| `--scale` / `-s` | `2.0` | Target upscale factor |
| `--pixfrac` | `0.7` | Drizzle pixel fraction (0–1) |
| `--chunks` | `8` | Memory chunks for Drizzle |
| `--no-deconv` | off | Skip Wiener deconvolution |
| `--align-scale` | auto | Override ECC downscale factor |
| `--jpeg` | auto | Force JPEG-mode processing |
| `--raw` | auto | Force RAW/PNG-mode processing |
| `--no-cache` | off | Disable Drizzle cache |
| `--cache-dir` | `~/.optico_cache` | Custom cache directory |
| `--verbose` / `-v` | off | Debug logging |

### JPEG vs RAW

Optico auto-detects input format by reading file headers (JPEG SOI marker `0xFF 0xD8`). JPEG input activates three automatic adjustments:

- **Phase 2 alignment:** ECC Gaussian filter enlarged 5 → 7 px to suppress 8×8 DCT inter-block edges that bias sub-pixel offset estimation.
- **Phase 8 Drizzle:** coverage-hole fill active for all inputs.
- **Phase 9 deconvolution:** PSF sigma ×1.35 (composite optical + JPEG quantisation blur); noise-floor scan range lowered from 0.75 → 0.60 × Nyquist to avoid JPEG spectral cutoff inflating the noise estimate.

Use `--jpeg` or `--raw` to override auto-detection.

---

## Pipeline Phases

| Phase | Module | Description |
|---|---|---|
| 0 | `pipeline.py` | JPEG vs RAW source detection |
| 1 | `pipeline.py` | Load burst images |
| 2 | `alignment.py` | Coarse ECC sub-pixel alignment |
| 3 | `alignment.py` | Harmony Anchor reference selection |
| 4 | `alignment.py` | Refined ECC alignment |
| 5 | `alignment.py` | N_eff entropy dither quality |
| 6 | `masking.py` | Dynamic foreground masking |
| 7 | `preflight.py` | Pre-flight scale bounding |
| 8 | `drizzle.py` | Drizzle stacking + coverage-hole fill |
| 9 | `deconvolution.py` | Frequency-dependent Wiener deconvolution |
| 10 | `pipeline.py` | Final output |

---

## Drizzle Cache

Phases 2–8 are deterministic. Results are cached to `~/.optico_cache` (keyed on input file SHA-256 + config). Subsequent runs with the same burst and settings skip to Phase 9 instantly. Use `--no-cache` to force a full reprocess.

---

## Configuration

All parameters are centralized in `backend/constants.py` via the `OpticoConfig` dataclass. Key fields:

```python
OpticoConfig(
    target_scale=2.0,        # upscale factor
    pixfrac=0.7,             # drizzle droplet size
    jpeg_input=None,         # None = auto-detect
    psf_override=None,       # explicit PSF sigma
    skip_deconv=False,
)
```

---

## References

- Fruchter & Hook (2002). *Drizzle: A Method for the Linear Reconstruction of Undersampled Images.* PASP 114.
- Wiener, N. (1949). *Extrapolation, Interpolation, and Smoothing of Stationary Time Series.*
