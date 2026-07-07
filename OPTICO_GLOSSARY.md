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

### 🎛️ Dual-Band Edge-Aware Wiener Deconvolution (雙頻邊緣感知維納反捲積)
A highly specialized frequency-domain filter used in Phase 9. Instead of applying a uniform deconvolution (which causes ringing artifacts on edges and amplifies noise in flat areas), it constructs two parallel restored frequencies ($K_{strong}$ for flat areas, $K_{weak}$ for harsh edges) and dynamically blends them using a spatial Sobel gradient map. This achieves absolute energy conservation ($Variance\_Boost \approx 1.00x$).

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
