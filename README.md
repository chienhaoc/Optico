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
| `--output` / `-o` | *(auto-derived)* | Output file path |
| `--scale` / `-s` | `0.0` (auto) | Target upscale factor (0.0 = auto-resolve from dither quality) |
| `--pixfrac` | `0.7` | Drizzle pixel fraction (0–1) |
| `--chunks` | `8` | Memory chunks for Drizzle |
| `--no-deconv` | off | Skip Wiener deconvolution |
| `--psf-base` | `0.63` | Wiener deconvolution base PSF scale factor (e.g. 0.35 for 17mm, 0.63 for 50mm) |
| `--psf-override` | None | Direct HR-pixel PSF sigma override (bypasses scale and cap) |
| `--align-scale` | auto | Override ECC downscale factor |
| `--jpeg` | auto | Force JPEG-mode processing |
| `--raw` | auto | Force RAW/PNG-mode processing |
| `--no-cache` | off | Disable Drizzle cache |
| `--cache-dir` | `~/.optico_cache` | Custom cache directory |
| `--verbose` / `-v` | off | Debug logging |

### JPEG vs RAW & Focal-Length Dedicated PSF

Optico auto-detects input format by reading file headers (JPEG SOI marker `0xFF 0xD8`). JPEG input activates automatic adjustments:

- **Phase 2 alignment:** ECC Gaussian filter enlarged 5 → 7 px to suppress 8×8 DCT inter-block edges.

- **Phase 9 deconvolution:** Bypasses unstable noise-contrast calculations on JPEG quantization floors by allowing users to manually map `psf_base` based on physical focal lengths:
  - **Focal Length <= 28mm (17mm wide-angle, small faces)** $\to$ `--psf-base 0.35` (to protect small facial details from over-sharpening).
  - **Focal Length = 45mm** $\to$ `--psf-base 0.57`.
  - **Focal Length = 50mm (mid-telephoto, larger faces)** $\to$ `--psf-base 0.63` (for maximum detail retrieval).

Use `--jpeg` or `--raw` to override auto-detection.

### Drizzle Kernel

`kernel_mode` (default **`lanczos4`**, changed 2026-07) selects Phase 8's accumulation kernel. `lanczos4`, combined with a grid-safe cap on the auto-estimated Phase 9 PSF sigma (which prevents the Wiener filter from amplifying the Drizzle kernel's grid frequency), beats `box` and `lanczos2` on ringing, grid-periodicity, and leave-one-out fidelity simultaneously on real Sony A7C bursts.


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
