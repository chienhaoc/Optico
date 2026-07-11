# Optico: Core Glossary & Theoretical Terminology

This document serves as the official glossary for the Optico MFSR (Multi-Frame Super Resolution) engine. It defines the advanced computational photography concepts, physical phenomena, and proprietary algorithms developed during the project's evolution.

## 1. Algorithmic Routing & Filtering

### 🌟 Cumulative Probability Decay (累積機率衰減模型)
A revolutionary frame-routing algorithm derived to measure the **Joint Survival Probability (聯合存活機率)** of high-frequency pixels across a stack of aligned images. Instead of using a fixed CC (Correlation Coefficient) threshold, it treats each frame's CC as an independent probability of structural integrity. By multiplying the CCs of sorted frames sequentially ($P = CC_1 \times CC_2 \times ... \times CC_n$), it precisely detects the exact frame where accumulated micro-errors become fatal to the optical structure, perfectly automating the cutoff point for the Elite Track.

### 📉 Knee Point Detection (拐點偵測 / Elbow Method)
An early evolutionary routing strategy attempted before the Cumulative Probability Decay model. It sought to find the "cliff" or sudden drop-off in CC scores between frames (e.g., a drop $> 2.5\%$). It was ultimately abandoned after empirical testing on `input2` proved that alignment decay in consecutive frame bursts is often a *smooth, continuous curve* rather than a stepped cliff. Algorithms hunting for a sudden "Knee" are blind to this smooth "Death by a thousand cuts", proving that cumulative probability is the only mathematically sound approach.

### 🎭 Hybrid Spatio-Luminance Routing (空間亮度混合路由)
*Optico v2.0 Roadmap.* An advanced dual-alignment strategy designed to break the "Detail vs. Noise" trade-off. It utilizes **Dense Optical Flow** purely on the Low-Frequency track to eliminate noise (sacrificing structural rigidity for pure luminance/color blending), while strictly applying **Rigid ECC Homography** and the Cumulative Probability Decay model on the High-Frequency track to preserve raw photon structures. The two are then fused via a Laplacian Pyramid.

### ⚓ Harmony Anchor (和諧定錨點)
The algorithm used in Phase 3 to select the ultimate Reference Frame. Instead of blindly picking the sharpest frame (which might be physically skewed or misaligned relative to the burst), it calculates the **Geometric Median** (the center of mass of all affine matrices) to find the most "harmonious" structural baseline, and then selects the sharpest frame near that center. This prevents outlier skewing in deep stacks.

## 2. Optical Phenomena & Physics

### 🦠 Micro-Blur Degradation (微模糊劣化)
The physical degradation of extremely high-frequency dark details (e.g., shadows, hair, textures) caused by sub-pixel alignment errors (Parallax, Rolling Shutter) when stacking multiple frames. Because these errors cannot be resolved by global affine/homography matrices, stacking too many frames inevitably "melts" the dark details. This phenomenon enforces the strict rule that for ultimate resolution, **Frame Quality > Frame Quantity**.

### ⚔️ Death by a Thousand Cuts (千刀萬剮 / 微差累積效應)
A colloquial term describing how stacking slightly misaligned frames (even with CC scores as high as 0.990 - 0.960) gradually erodes the underlying optical resolution. Every added frame introduces an independent micro-shift, progressively dragging the Retained Pixel Ratio and the Optical Blur Limit down.

### 🛑 SNR Alignment Blur Limit (光學極限模糊圈)
A mathematical boundary defined by the Spatial Sampling Theorem (Nyquist limit) and Cramer-Rao Lower Bound. It proves that the maximum upscale factor of a Drizzle operation cannot be arbitrarily set. It is strictly bounded by the structural noise (Signal-to-Noise Ratio of the alignment) calculated via the Dynamic Mask. If the Retained Pixel Ratio drops, this limit mathematically forces the engine to cap the upscale factor to prevent interpolative smearing.

## 3. Signal Processing

### 🎼 Dual-Track Frequency Fusion (雙頻軌道融合)
Optico's core architectural layout. It splits the MFSR process into two parallel tracks:
* **Elite Track (High-Frequency)**: Uses only a very small, perfectly aligned subset of frames to reconstruct raw photon edges.
* **Full Track (Low-Frequency)**: Uses the entire frame stack (with heavy Gaussian low-pass filtering) to construct a zero-noise color and illumination base.
The two are fused seamlessly, bypassing the traditional single-track limitations.

### 🎯 Frequency-Dependent Wiener Deconvolution (頻率相依維納反捲積)
The current Phase 9 filter, which replaced the Dual-Band Edge-Aware scheme below. Rather than a scalar or dual-band regularizer, it estimates a per-frequency map $K(f)$ directly from the image's own power spectrum (locating the white-noise plateau at high frequency). Natural-image spectra fall as $\sim 1/f^2$ while sensor noise is roughly flat — a single scalar $K$ cannot represent both regimes at once, but a per-frequency map does, directly matching the textbook SNR-inverse Wiener solution. Measured +1.54 dB average PSNR improvement over the dual-band approach across noise levels σ = 1–9.

### 🕸️ Grid-Safe PSF Sigma Cap (格柵安全 PSF 標準差上限)
A 2026-07 real-burst finding: Wiener deconvolution amplifies whatever residual grid artifact the Drizzle kernel (see the **Box-Overlap Drizzle Kernel** graveyard entry below) leaves behind, whenever the filter's passband reaches the grid's spatial frequency $f_{\text{grid}} = 1/S$. Since the Wiener cutoff frequency for a Gaussian PSF of sigma $\sigma$ is $f_c(\sigma) = \sqrt{\ln(1/K)}/(2\pi\sigma)$, solving $f_c(\sigma) = f_{\text{grid}}$ gives a theoretical upper bound $\sigma_{\text{cap}}$ beyond which the filter starts re-amplifying the grid. A theory-grounded sweep across two real tripod bursts confirmed the prediction closely: grid-periodicity collapsed sharply right at the predicted threshold, not gradually or at an arbitrary point — the kind of confirmation that justifies trusting the formula over further trial-and-error. Applied only to *auto-estimated* PSF sigma; an explicit `--psf-override` remains a raw, uncapped value for controlled experimentation.

## 4. Evolutionary Graveyard (失敗為成功之母)

The following algorithms were implemented, empirically tested, and ultimately abandoned. Documenting these failures is critical, as they each revealed a profound truth about optical physics that shaped the final engine.

### 🪦 Static CC Thresholding (靜態 CC 閥值)
* **What it was**: A simplistic routing logic that rigidly discarded any frame with a CC score below a fixed number (e.g., $CC < 0.85$).
* **Why it failed**: CC scores are relative to scene texture. A flat night sky might naturally yield very high CCs ($0.99+$), while a highly textured daylight scene might yield lower CCs ($0.90$) due to noise and micro-parallax. A static threshold either allowed massive micro-blur in night scenes or discarded perfectly viable frames in day scenes.
* **What it taught us**: Routing must be dynamically based on the *relative decay* of the specific frame stack, leading to the Cumulative Probability Decay model.

### 🪦 Arithmetic Mean Harmony Anchor (算術平均定錨)
* **What it was**: An early Phase 3 logic that calculated the arithmetic mean (average) of all spatial displacements to find the center Reference Frame.
* **Why it failed**: A single severely blurred or violently shifted outlier frame (e.g., a massive hand jolt) would heavily skew the mathematical average, causing the engine to anchor on a suboptimal, off-center frame.
* **What it taught us**: In unconstrained burst stacks, outlier rejection is paramount. It was replaced by the **Geometric Median (Robust Median)**, which isolates the true optical center of mass regardless of wild outliers.

### 🪦 Empirical Linear Scale Limits (經驗線性極限)
* **What it was**: Hardcoded multiplier rules (e.g., limiting upscale to $1.2x$ or $1.6x$) based purely on human guessing to prevent Drizzle grid artifacts.
* **Why it failed**: It lacked physical grounding. When alignment was perfect, $1.2x$ severely bottlenecked the potential resolution. When alignment was terrible, $1.6x$ caused catastrophic smearing and ringing.
* **What it taught us**: Resolution cannot be guessed; it is a strict physical derivative of the alignment error. It was replaced by the **SNR Alignment Blur Limit**, which mathematically couples the scale ceiling to the actual Retained Pixel Ratio.

### 🪦 Dual-Band Edge-Aware Wiener Deconvolution (雙頻邊緣感知維納反捲積)
* **What it was**: The original Phase 9 filter. Instead of a uniform deconvolution, it constructed two parallel restored frequencies ($K_{strong}$ for flat areas, $K_{weak}$ for harsh edges) and dynamically blended them in the spatial domain via a Canny/Sobel edge mask, targeting absolute energy conservation ($Variance\_Boost \approx 1.00x$).
* **Why it failed**: Not a failure of correctness, but of optimality — a synthetic ground-truth benchmark (noise σ = 1–9) found the **Frequency-Dependent Wiener Deconvolution** approach above consistently outperformed it (+1.54 dB average PSNR), because a flat $K$ within each spatial band still cannot represent how natural-image spectra ($\sim 1/f^2$) and sensor noise (flat) diverge across frequency, only across space.
* **What it taught us**: The right axis to regularize along is frequency, not screen-space edge proximity. Spatial blending was solving the right problem (edges need different treatment than flat regions) on the wrong axis.

### 🪦 Box-Overlap Drizzle Kernel — Second Revert (Box 疊圖核，二度淘汰)
* **What it was**: The 4-neighbour overlap Drizzle kernel (`kernel_mode='box'`), reinstated as the default in 2026-07 after a synthetic ground-truth benchmark showed it had zero coverage holes at the validated `pixfrac=0.7` while being ~4× sharper than `lanczos2`.
* **Why it failed (again)**: The synthetic validation didn't generalize. A real-burst benchmark (two Sony A7C tripod bursts, `backend/benchmarks/kernel_bench.py`) measured `box`'s structural coverage-hole grid pattern at 3–17× worse `grid_periodicity` than the sinc-shaped kernels, on real photographic content the synthetic test hadn't captured.
* **What it taught us**: A synthetic-only benchmark validated a real physical mechanism (coverage-hole structural zeros exist) but not its real-world *severity* — the whole reason `input3`/`input4` real bursts were set aside earlier in the project. `lanczos2` was restored as the default, this time paired with the **Grid-Safe PSF Sigma Cap** above so the gain in coverage uniformity isn't erased by Phase 9 re-amplifying the residual grid.

### 🪦 Box-Supersample Anti-Aliasing (Box 超取樣反鋸齒)
* **What it was**: A candidate fix for `box`'s grid artifact: accumulate at 2× the requested scale using plain `box`, then area-decimate (`cv2.INTER_AREA`) back down — pushing the periodic coverage zeros above Nyquist before a proper low-pass decimation, the standard signal-processing anti-aliasing strategy.
* **Why it failed**: Real-burst benchmarking showed it only reduced `grid_periodicity` by ~30–50%, far short of `lanczos2`'s 3–17× reduction, while adding real compute/memory cost (a full supersampled intermediate per chunk).
* **What it taught us**: The grid artifact isn't purely a sampling-rate problem solvable by brute-force oversampling — `box`'s overlap kernel has *structural* zeros (some HR pixels get exactly zero weight from any LR frame), which a 2× supersample doesn't push far enough above Nyquist to fully separate from the signal band. The sinc-shaped kernels avoid this by construction (no HR pixel gets a zero weight in the first place), not by out-sampling it.
