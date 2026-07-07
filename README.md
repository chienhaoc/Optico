# Optico 📸

**Multi-Frame Super-Resolution (MFSR) Engine — Pure Optical Data, Authentic Reconstruction.**

Optico is a state-of-the-art computational photography engine designed to extract extreme high-resolution details and noise-free purity from burst photos (e.g., 4 to 8 frames, including tripod-mounted dithered bursts or handheld bursts) using strict optical physics. While modern AI upscalers are excellent for creative generation, Optico takes a different path: reconstructing genuine sub-pixel reality purely from captured optical data. Stacking tripod-mounted bursts with micro-dither patterns (such as sensor shift or subtle vibrations) provides the most ideal conditions to resolve true sub-pixel details.

---

## 🌟 Core Architecture

The Optico engine is built upon three foundational pillars of optical processing:

### 1. Two-Pass Pre-flight Architecture (Optical Bounding)
Blindly pushing high upscaling factors on frame bursts leads to catastrophic Alignment Drift Blur. Optico employs a highly advanced Pre-flight mask analysis to calculate the `Global Retained Pixel Ratio` ($R_{global}$). 
Based on the **Spatial Sampling Theorem (Nyquist)** and **Cramer-Rao Lower Bound (CRLB)**, the engine calculates the exact Signal-to-Noise Ratio (SNR) of the image alignment, capping the maximum safe scale factor using rigorous optical limits rather than empirical guesswork.
* **Static scenes**: Pushed to absolute density limits (e.g., 2.5x - 2.8x).
* **Dynamic scenes**: Capped precisely (e.g., 1.3x - 1.8x) to trade resolution for pristine denoising.

### 2. Adaptive Dual-Band Wiener Deconvolution
Hardcoded sharpening is fundamentally flawed. Optico dynamically estimates the true noise floor using the corrected Laplacian Median Absolute Deviation (MAD) of the Drizzle stack. 
It then splits the Fourier domain into dual bands and blends them in the spatial domain:
* **Flat Regions**: Restored with aggressive energy (low K) to recover textures.
* **Edges**: Strictly protected with conservative settings (high K) to eliminate ringing halos and grid artifacts.

### 3. Active Memory Chunking
Massive Multi-Frame Drizzle operations (e.g., stacking eight 24-Megapixel frames into a 1.5-Gigapixel canvas) typically crash personal computers via OS Swap Thrashing.
Optico horizontally slices the target High-Resolution canvas into independent memory strips, locking peak memory usage to strictly under 3GB regardless of the target resolution.

---

## 🛠️ Codebase Structure

Optico is fully modularized under `backend/`:
* [__init__.py](file:///C:/Users/chchen/Optico_git/backend/__init__.py): Engine versioning and main API exports.
* [constants.py](file:///C:/Users/chchen/Optico_git/backend/constants.py): Configuration class (`OpticoConfig`) and centralization of all optical constants.
* [alignment.py](file:///C:/Users/chchen/Optico_git/backend/alignment.py): Sub-pixel ECC image registration, Harmony Anchor reference selection, and 2D circular statistics.
* [masking.py](file:///C:/Users/chchen/Optico_git/backend/masking.py): Dual-threshold motion masking using a Poisson-Gaussian noise model.
* [preflight.py](file:///C:/Users/chchen/Optico_git/backend/preflight.py): Resolution bounding based on Nyquist and CRLB limits.
* [drizzle.py](file:///C:/Users/chchen/Optico_git/backend/drizzle.py): Vectorized Variable-Pixel Linear Reconstruction with memory chunking.
* [deconvolution.py](file:///C:/Users/chchen/Optico_git/backend/deconvolution.py): Spatially blended edge-aware Wiener deconvolution in the frequency domain.
* [pipeline.py](file:///C:/Users/chchen/Optico_git/backend/pipeline.py): Orchestrates the pipeline and handles the Command Line Interface (CLI).

---

## ⚙️ Setup & Installation

Optico requires Python 3.10+ and standard scientific libraries. 

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Verify installation:
   ```bash
   python -c "import backend; print(backend.__version__)"
   ```

---

## 🚀 Usage Guide

### Command Line Interface (CLI)

Run the end-to-end pipeline directly from the command line:

```bash
# Basic run: process a burst folder with default settings (2.0x zoom, output to optico_output.png)
python -m backend.pipeline --input path/to/burst_folder --output output.png

# Advanced run: request 3.0x zoom, target pixfrac of 0.6, split into 12 memory chunks, with verbose logs
python -m backend.pipeline --input path/to/burst_folder --scale 3.0 --pixfrac 0.6 --chunks 12 --verbose

# Run without deconvolution sharpening
python -m backend.pipeline --input path/to/burst_folder --no-deconv
```

### Python API Example

You can easily integrate Optico's pipeline directly into your Python computational photography workflows:

```python
import cv2
from backend import OpticoConfig, run_pipeline

# 1. Load burst images (BGR uint8)
image_paths = ["frame0.png", "frame1.png", "frame2.png", "frame3.png"]
images = [cv2.imread(p) for p in image_paths]

# 2. Configure options
config = OpticoConfig(
    target_scale=2.5,        # Desired upscale (may be capped by Pre-flight)
    pixfrac=0.7,             # Drizzle shrunken droplet ratio
    num_chunks=8,            # Bounded memory chunk count
    skip_deconv=False        # Apply Wiener sharpening
)

# 3. Run MFSR pipeline
hr_result = run_pipeline(images, config=config, output_path="mfsr_result.png")
```

---

## 🛠️ Upcoming: Dual-Track Frequency Merging
Optico is currently evolving to support Frequency Separation:
* **High-Frequency Track (Details)**: Generated exclusively from "Elite" frames (e.g., $cc > 0.95$) to maximize resolution limits without alignment blur.
* **Low-Frequency Track (Noise Floor/Color)**: Generated from all available frames to flatten the noise floor, merged seamlessly via Laplacian pyramids.

---
*Built for the pursuit of absolute optical truth.*