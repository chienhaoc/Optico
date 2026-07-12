# Optico: Core Algorithm & Mathematics

This document details the mathematical models, formulations, and exact algorithmic steps implemented in the Optico Multi-Frame Super-Resolution (MFSR) engine.

---

## 0. JPEG vs RAW Source Detection (`pipeline.py`)

Before any processing begins, Optico auto-detects whether the burst images are JPEG-sourced by reading the first 2 bytes of each file and checking for the JPEG SOI marker (`0xFF 0xD8`). The result is stored in `config.jpeg_input` and propagates downstream to three phases:

| Phase | RAW / PNG | JPEG |
|---|---|---|
| Phase 2 — ECC alignment | `gauss_filt_size = 5` | `gauss_filt_size = 7` |
| Phase 8 — Drizzle | coverage-hole fill (universal) | same |
| Phase 9 — Deconvolution | PSF σ = `0.4·S`, HF fraction = 0.75, taper = 48 px | PSF σ ×1.10, HF fraction = 0.60, taper = 48 px |

The detection can be overridden via `config.jpeg_input = True/False` or the CLI flags `--jpeg` / `--raw`.

In both columns, the resulting auto-estimated PSF σ is additionally capped by the grid-safe formula described in §5 before being handed to the Wiener filter.

---

## 1. Sub-pixel Registration & Dither Quality (`alignment.py`)

### Sub-pixel Translation Mapping
For burst photography (either handheld or tripod-mounted bursts), camera motion is modeled purely as rigid translation to prevent overfitting to high-frequency noise. Each target frame $I_i$ is mapped to the reference frame $I_{ref}$ by solving:
$$E(M_i) = \max \text{ECC}(I_{ref}, I_i(M_i))$$
where the transformation matrix is:
$$M_i = \begin{bmatrix} 1 & 0 & t_{x,i} \\ 0 & 1 & t_{y,i} \end{bmatrix}$$
To speed up convergence and stabilize registration against high-frequency noise, registration is solved on downscaled frames (scale factor $\gamma = 0.5$). The resolved translations are then rescaled back to the original resolution:
$$t_{x,\text{original}} = \frac{t_{x,\text{downscaled}}}{\gamma}, \quad t_{y,\text{original}} = \frac{t_{y,\text{downscaled}}}{\gamma}$$

**JPEG note:** JPEG 8×8 DCT blocks introduce inter-block discontinuities that ECC can lock onto as false sub-pixel offsets. For JPEG input the pre-smoothing Gaussian filter is enlarged from 5 → 7 px (`JPEG_ECC_GAUSS_FILT_SIZE`) to suppress these block-edge artifacts before ECC runs.

### N_eff Entropy Dither Quality
Sub-pixel offsets must cover the unit pixel uniformly. Optico quantifies coverage via Shannon entropy on a 4×4 sub-pixel histogram:
$$H = -\sum_{k} p_k \log_2 p_k$$
$$N_{\text{eff}} = 2^H \in [1.0,\ 16.0]$$
Higher $N_{\text{eff}}$ means more independent sub-pixel positions; the pre-flight scale cap uses $\sqrt{N_{\text{eff}}}$ as the dither contribution.

> **Historical note:** earlier versions used a 2D Rayleigh resultant with small-sample bias correction ($\pi/4N$ floor). That approach saturated $Q = 1.0$ in 6/8 test distributions, effectively disabling the pre-flight branch. The $N_{\text{eff}}$ entropy metric is numerically stable and physically interpretable.

### Harmony Anchor (Geometric Median Selection)
The reference frame is chosen by solving the geometric median of translation coordinates:
$$\mathbf{t}_{\text{median}} = \arg\min_{\mathbf{x}} \sum_{i=1}^N \|\mathbf{t}_i - \mathbf{x}\|_2$$
Solved via Weiszfeld's algorithm. The sharpest frame within the 50th-percentile distance to the median is selected:
$$\text{Sharpness}(I) = \text{Var}(\nabla^2 I)$$

---

## 2. Dynamic Foreground Masking (`masking.py`)

To prevent ghosting artifacts, we compute a normalized difference mask.
Sensor noise is modeled locally using a Poisson-Gaussian model:
$$\sigma_{\text{noise}}(x,y) = \sqrt{a \cdot I_{\text{ref}}(x,y) + b}, \quad a=0.5,\ b=1.0$$
Gradient magnitude is included in the denominator to suppress false positives at sub-pixel edges:
$$\text{denom}(x,y) = \sigma_{\text{noise}}(x,y) + 0.3 \cdot \|\nabla I_{\text{ref}}(x,y)\|_2 + 1.0$$
$$D_{\text{norm}, i}(x,y) = \frac{|I_{i,\text{warped}}(x,y) - I_{\text{ref}}(x,y)|}{\text{denom}(x,y)}$$
Dual-thresholding:
- Background motion: $D_{\text{norm}} > 1.5$ (7×7 dilation)
- Subject motion: $D_{\text{norm}} > 3.0$ (11×11 dilation, 2 iterations)

A soft weight map $W_i(x,y) \in [0, 1]$ is produced by Gaussian-smoothing the combined binary mask.

---

## 3. Pre-flight Scale Bounding (`preflight.py`)

The theoretical resolution limit is bounded by two physical constraints:
1. **Sampling Density Limit:**
   $$S_{\text{density}} = \sqrt{N \cdot R_{\text{global}}}$$
2. **Alignment Blur Limit (CRLB):** using $N_{\text{eff}}$ as the dither quality measure:
   $$S_{\text{blur}} = \alpha \cdot \sqrt{N_{\text{eff}}}$$
   where $\alpha$ is the adaptive decay factor. Instead of a hard constant, $\alpha$ scales adaptively between `OPTICAL_DECAY_CONSTANT` (0.75) and `OPTICAL_DECAY_MAX` (0.90) based on alignment quality (`cc_mean`):
$$\alpha_{\text{adaptive}} = \text{clip}\left(0.75 + \frac{\text{cc\_mean} - 0.85}{0.98 - 0.85} \times (0.90 - 0.75),\ 0.75,\ 0.90\right)$$
This ensures that tighter alignment allows for higher upscale bounds.

The final scale factor is:
$$S_{\text{final}} = \min(S_{\text{target}},\ S_{\text{density}},\ S_{\text{blur}})$$

---

## 4. Vectorized Drizzle Stacking (`drizzle.py`)

Optico implements Variable-Pixel Linear Reconstruction (Fruchter & Hook 2002) adapted for handheld burst photography.

Each LR pixel is projected onto the HR grid via:
$$x_{\text{HR}} = S \cdot (x_{\text{LR}} - t_x), \quad y_{\text{HR}} = S \cdot (y_{\text{LR}} - t_y)$$

The droplet radius is:
$$r_{\text{drop}} = \frac{p \cdot S}{2}, \quad p = \text{pixfrac} \in [0,1]$$

Per-chunk accumulation:
$$\text{Num}(x,y) = \sum_{i=1}^N \text{overlap}_i(x,y) \cdot W_i(x,y) \cdot I_i(x,y)$$
$$\text{Den}(x,y) = \sum_{i=1}^N \text{overlap}_i(x,y) \cdot W_i(x,y)$$
$$I_{\text{HR}}(x,y) = \frac{\text{Num}(x,y)}{\max(\text{Den}(x,y),\ 10^{-6})}$$

### Coverage-Hole Fill

The backward 4-neighbour overlap kernel (`kernel_mode='box'`) has a structural blind spot: when the nearest LR pixel centre projects to a position $> r_{\text{drop}}$ away from an HR pixel centre, overlap = 0. If all frames share similar sub-pixel offsets this creates a periodic grid of under-covered HR pixels, producing visible bright/dark grid artifacts.

**Fix:** After accumulation and before normalisation, HR pixels where
$$\text{Den}(x,y) < \tau \cdot \text{median}(\text{Den})$$
are flagged as coverage holes ($\tau = 0.15$, `DRIZZLE_COVERAGE_FLOOR_RATIO`). Their numerator and denominator are replaced by a 3×3 uniform (box) neighbourhood average:
$$\text{Den}_{\text{filled}}(x,y) = (\text{uniform\_filter}_{3\times3} * \text{Den})(x,y)$$
This is equivalent to bilinear interpolation from surrounding well-covered pixels. Only hole pixels are modified; well-covered pixels are unchanged. This safety net runs for every `kernel_mode`, but is rarely triggered for the sinc-shaped kernels below since they have no structural zeros to begin with.

### Kernel Selection (`kernel_mode`)

* **`lanczos4`** (default, changed 2026-07): windowed sinc kernel, $a=4$ (`_lanczos`, `DRIZZLE_LANCZOS_A`). Every HR pixel receives weight from a window of $8\times8$ pixels. This provides the sharpest high-frequency coverage, which when combined with the grid-safe PSF cap, outperforms both `box` and `lanczos2` on aliasing-resistance, grid-periodicity, and leave-one-out fidelity simultaneously on real Sony A7C bursts.
- **`lanczos2`**: windowed sinc kernel with $a=2$, softer than `lanczos4`.
- **`lanczos2_clamped`**: windowed sinc with negative sidelobe weights zeroed.
- **`box`**: backward nearest-neighbor kernel, subject to grid ripples.

### LR Data-Side Pre-emphasis (Phase 8.0)
For JPEG inputs, compression quantizes away high-frequency DCT coefficients. To restore edge contrast prior to Drizzle stacking, Optico applies a pre-emphasis high-pass filter to each input LR frame:
1. **Extract high frequencies:**
   $$\text{hp}_i = I_{\text{LR}, i} - \text{GaussianBlur}(I_{\text{LR}, i},\ \text{kernel}=3\times3,\ \sigma=0.8)$$
2. **Apply compensation:**
   $$I_{\text{pre}, i} = \text{clip}(I_{\text{LR}, i} + \alpha \cdot \text{hp}_i,\ 0,\ 255)$$
   where $\alpha = 0.55$ represents the pre-emphasis gain. This pre-enhancement increases spatial gradient gradients prior to stacking, which dramatically lowers the regularisation burden during Phase 9 deconvolution.

---

## 5. Frequency-Dependent Wiener Deconvolution (`deconvolution.py`)

### Edge Taper (Spectral Leakage Suppression)

`scipy.fft.fft2` assumes the image is circulant (periodic). Real images have
discontinuous top/bottom/left/right boundaries; this discontinuity creates
spectral leakage concentrated on the `fx=0` axis. After `IFFT2` the leakage
appears as **full-width horizontal bands that cross smooth regions such as
faces** — completely unrelated to image content.

Before FFT2, `_edge_taper()` blends the outermost `EDGE_TAPER_WIDTH = 48`
pixels on each edge toward the image mean using a raised-cosine (Hann) ramp:
$$w[i] = \frac{1}{2}\left(1 - \cos\frac{\pi i}{T}\right), \quad i = 0, \ldots, T-1$$
where $T$ = `EDGE_TAPER_WIDTH`. The image mean is subtracted before tapering
and restored afterwards, preserving the DC component.

### Noise Estimation
The global noise floor standard deviation $\sigma_{\text{noise}}$ is estimated from the Laplacian MAD:
$$\sigma_{\text{noise}} = \frac{1.4826 \cdot \text{MAD}(\nabla^2 I)}{\sqrt{20}}$$
The $\sqrt{20}$ factor corrects for the 3×3 Laplacian kernel's noise amplification.

### Physical Focal-Length PSF Model
Because in-camera JPEG compression heavily quantizes and denoises flat/dark areas, indirect mathematical estimations of Noise-to-Contrast ratio ($R$) on JPEG inputs suffer from severe quantization blindspots (leading to deconvolution over-sharpening artifacts). Optico bypasses this instability by anchoring the base PSF scale factor directly to the camera's physical lens focal length:
$$\sigma_{\text{eff}} = \text{psf\_base} \times S_{\text{final}}$$
where:
* **Focal Length <= 28mm (17mm ultra-wide, small faces)** $\to$ $\text{psf\_base} = 0.35$ (suppresses artifacts and preserves facial features).
* **Focal Length = 45mm** $\to$ $\text{psf\_base} = 0.57$.
* **Focal Length = 50mm (mid-telephoto, larger faces)** $\to$ $\text{psf\_base} = 0.63$ (maximizes resolution and edge contrast).

**Manual Override:** Use `--psf-base <val>` to manually specify the base PSF scale factor (e.g. `--psf-base 0.45`), or `--psf-override <sigma_hr>` to supply the raw PSF sigma directly in HR pixels (bypassing scale factors and the grid-safe cap entirely).

### Grid-Safe PSF Sigma Cap

Real-burst benchmarking found that Wiener deconvolution amplifies the Drizzle kernel's residual grid artifact (§4) whenever the filter's passband reaches the grid's spatial frequency. For a Gaussian PSF of sigma $\sigma$, the Wiener cutoff frequency is approximately:
$$f_c(\sigma) = \frac{\sqrt{\ln(1/K)}}{2\pi\sigma}$$
The Drizzle grid artifact sits at spatial frequency $f_{\text{grid}} = 1/S$ cycles per HR pixel, alias-wrapped into the Nyquist range $(0, 0.5]$ for $S < 2$. Solving $f_c(\sigma) = f_{\text{grid}}$ for $\sigma$ gives the largest PSF sigma whose passband does not reach the grid frequency:
$$\sigma_{\text{cap}} = \frac{\sqrt{\ln(1/K)}}{2\pi f_{\text{grid}}}$$
using the run's own diagnostic $K_{\text{est}}$ (from noise MAD, above) in place of $K$. This cap is applied only to the **auto-estimated** `psf_sigma` (both the RAW baseline and the JPEG-scaled value above); an explicit `--psf-override` is left uncapped.

**Validation:** a theory-grounded sweep (`psf_override` $\in \{0.88, 0.80, 0.50, 0.40\}$ across `lanczos2`/`lanczos2_clamped` and two real bursts at $S=2.0$) confirmed the prediction closely — grid_periodicity collapses sharply right around the predicted threshold (e.g. one burst measured $K_{\text{est}} \approx 0.08 \Rightarrow \sigma_{\text{cap}} \approx 0.51$, and its `grid_periodicity` worst-ratio dropped from 587.7 (psf=0.88) to 41.5 (psf=0.50), with only marginal further gain below the threshold). Full data: `backend/benchmarks/reports/`.

### Frequency-Dependent Regularisation $K(f)$

Rather than a scalar $K$, Optico estimates a per-frequency regularisation map from the image's own power spectrum:

1. **Locate noise floor:** scan the radial power spectrum from $f_{\text{lo}} \times f_{\text{Nyquist}}$ outward. Find the plateau where the gradient of median power falls below 5% of the DC-region power. The median in that plateau annulus is $N_{\text{floor}}$.

   > **JPEG fix:** For JPEG input, the scan starts at $f_{\text{lo}} = 0.60 \times f_{\text{Nyquist}}$ (`JPEG_NOISE_FLOOR_HIGH_FREQ_FRACTION`) instead of 0.75. JPEG DCT quantisation creates a hard spectral cutoff at ~0.55–0.65 × Nyquist; starting above this avoids mistaking the JPEG truncation band for the white-noise floor, which would inflate $N_{\text{floor}}$ by 2–5× and over-regularise all frequencies.

2. **Per-frequency signal power:**
   $$S(f) = \max\bigl(P(f) - N_{\text{floor}},\ N_{\text{floor}} \cdot 0.005\bigr)$$

3. **Per-frequency $K$:**
   $$K(f) = \text{clip}\!\left(\frac{N_{\text{floor}}}{S(f)},\ 10^{-4},\ 200\right)$$

### Wiener Filter with DC-Gain Preservation
$$\hat{F}(u,v) = \frac{H^*(u,v)}{|H(u,v)|^2 + K(u,v)} \cdot G(u,v)$$
The DC bin $[0,0]$ is forced to unity gain to prevent mean-brightness shift:
$$W_{\text{resp}}(0,0) = 1.0$$

> **Why frequency-dependent $K$ outperforms dual-band Canny blending:** Natural-image power spectra fall as $\sim 1/f^2$ while sensor noise is approximately white (flat power). A scalar $K$ cannot represent both regimes simultaneously. The previous dual-band approach used Canny edge masks to proxy for this, but still applied a flat $K$ within each band. The per-frequency approach directly matches the textbook SNR-inverse Wiener solution, yielding +1.54 dB PSNR average improvement across noise levels σ = 1–9.
