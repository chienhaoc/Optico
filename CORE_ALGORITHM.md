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
   where $\alpha = 0.75$ is the optical decay constant.

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

`box` above has a real, not merely theoretical, failure mode: a real-burst benchmark (`backend/benchmarks/kernel_bench.py`, two Sony A7C tripod bursts, scale=2.0) measured its coverage-hole grid pattern via `grid_periodicity_score` (FFT peak-to-floor ratio at the aliased grid frequency $1/S$) and found it 3-17× worse than the sinc-shaped alternatives below, confirming the coverage-hole fill above does not fully eliminate the underlying periodic ripple.

- **`lanczos2`** (default, changed 2026-07): windowed sinc kernel, $a=2$ (`_lanczos`, `DRIZZLE_LANCZOS_A`). Every HR pixel receives positive weight from at least one LR pixel — no structural zeros, hence no periodic coverage-hole grid. Combined with the grid-safe PSF cap (§5), this beat `box` on ringing, grid-periodicity, *and* leave-one-out fidelity simultaneously on both benchmarked real bursts — a genuine improvement, not a trade-off. Full data: `backend/benchmarks/reports/`.
- **`lanczos2_clamped`**: same footprint as `lanczos2` but negative sidelobe weights are zeroed (`clamp_negative=True`), intended to remove Gibbs-style ringing. Benchmarked but not selected as default: it reduces ringing slightly at matched PSF, but counter-intuitively has *higher* grid_periodicity than plain `lanczos2` at most tested PSF values — clamping the negative lobes does not uniformly improve coverage uniformity.
- **`box_supersample`**: accumulates with `box` at `DRIZZLE_SUPERSAMPLE_FACTOR`× the requested scale, then area-decimates (`cv2.INTER_AREA`) back down — the standard anti-aliasing strategy of pushing the periodic zeros above Nyquist before a proper low-pass decimation. Benchmarked and discarded: only reduced grid_periodicity by ~30-50% (vs. lanczos2's 3-17×), with no ringing benefit and added compute cost.

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

**Why JPEG is more affected:** JPEG 8px DCT block boundaries are co-aligned
across all burst frames. Drizzle stacking reinforces rather than averages them,
increasing the vertical boundary discontinuity strength beyond that of RAW
input. Wiener's low/mid-frequency amplification then magnifies the leakage
bands into clearly visible stripes.

**Sandbox measurement** (512×512 face scene with JPEG block residuals, n=8 runs):

| | Row-mean std | vs input (27.79) |
|---|---|---|
| Without taper | 30.76 | +10.7% ← visible banding |
| **With taper** | **27.66** | **−0.5% ← eliminated** |

### Noise Estimation
The global noise floor standard deviation $\sigma_{\text{noise}}$ is estimated from the Laplacian MAD:
$$\sigma_{\text{noise}} = \frac{1.4826 \cdot \text{MAD}(\nabla^2 I)}{\sqrt{20}}$$
The $\sqrt{20}$ factor corrects for the 3×3 Laplacian kernel's noise amplification.

### PSF Model
The optical PSF is modeled as a Gaussian with:
$$\sigma_{\text{PSF}} = \max(0.4,\ 0.4 \cdot S_{\text{final}})$$

**JPEG correction:** JPEG quantisation adds a blur PSF ($\sigma_{\text{JPEG}} \approx 0.3\text{–}0.5\ \text{px}$) on top of the optical PSF. The composite effective sigma is:
$$\sigma_{\text{eff}} = \sqrt{\sigma_{\text{optical}}^2 + \sigma_{\text{JPEG}}^2} \approx \sigma_{\text{optical}} \times 1.35$$
For JPEG input, `psf_sigma` is multiplied by `JPEG_PSF_SCALE_FACTOR`. This was lowered from `1.35` to `1.10` (2026-07): the 1.35 composite-blur estimate over-stated the true PSF, causing excessive inverse-filter gain near the cutoff frequency and ~3px-wide undershoot bands after high-contrast edges (sandbox: post-edge undershoot −8.74 → −5.61 ADU, PSNR +0.83 dB at high contrast). `1.10` still compensates for real JPEG quantisation blur while keeping overshoot below visual threshold.

**Manual override:** Use `--psf-override <sigma_hr>` to supply the PSF sigma directly **in HR pixels** — this is a deliberate convention (see `pipeline.py`'s module docstring): the value is used as-is and is *not* rescaled by `final_scale` downstream, unlike the auto-estimated path above. An explicit override also bypasses the grid-safe cap described next, so it remains a raw value for controlled experimentation (e.g. `backend/benchmarks/kernel_bench.py --psf-override`). Recommended for telephoto lenses, when default auto-estimation under- or over-corrects blur, or for benchmarking.

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
