# Optico: Core Algorithm & Mathematics

This document details the mathematical models, formulations, and exact algorithmic steps implemented in the Optico Multi-Frame Super-Resolution (MFSR) engine.

---

## 1. Sub-pixel Registration & Dither circular statistics (`alignment.py`)

### Sub-pixel Translation Mapping
For burst photography (either handheld or tripod-mounted bursts), camera motion is modeled purely as rigid translation to prevent overfitting to high-frequency noise. Each target frame $I_i$ is mapped to the reference frame $I_{ref}$ by solving:
$$E(M_i) = \max \text{ECC}(I_{ref}, I_i(M_i))$$
where the transformation matrix is:
$$M_i = \begin{bmatrix} 1 & 0 & t_{x,i} \\ 0 & 1 & t_{y,i} \end{bmatrix}$$
To speed up convergence and stabilize registration against high-frequency noise, registration is solved on downscaled frames (scale factor $\gamma = 0.5$). The resolved translations are then rescaled back to the original resolution:
$$t_{x,\text{original}} = \frac{t_{x,\text{downscaled}}}{\gamma}$$
$$t_{y,\text{original}} = \frac{t_{y,\text{downscaled}}}{\gamma}$$

### 2D Torus Resultant Vector Length (Dither Quality)
To achieve sub-pixel super-resolution, sub-pixel displacements must cover the unit pixel area uniformly. Optico models this using 2D circular statistics on the unit torus.

For valid translation offsets, we extract the sub-pixel fractional parts:
$$f_{x,i} = t_{x,i} - \lfloor t_{x,i} \rfloor, \quad f_{y,i} = t_{y,i} - \lfloor t_{y,i} \rfloor$$
These are mapped to phase angles:
$$\theta_{x,i} = 2\pi f_{x,i}, \quad \theta_{y,i} = 2\pi f_{y,i}$$
The axis-wise mean resultant vector lengths are:
$$R_x = \sqrt{\left(\frac{1}{N}\sum_{i=1}^N \cos\theta_{x,i}\right)^2 + \left(\frac{1}{N}\sum_{i=1}^N \sin\theta_{x,i}\right)^2}$$
$$R_y = \sqrt{\left(\frac{1}{N}\sum_{i=1}^N \cos\theta_{y,i}\right)^2 + \left(\frac{1}{N}\sum_{i=1}^N \sin\theta_{y,i}\right)^2}$$
To preserve the coordinate correlation, the joint 2D resultant vector length is defined as:
$$R_{2D} = \sqrt{R_x \cdot R_y}$$
The final dither quality is:
$$Q_{\text{dither}} = 1 - R_{2D} \in [0.0, 1.0]$$
A value of $Q \to 1.0$ indicates uniform sub-pixel coverage, while $Q \to 0.0$ indicates stacked redundant sub-pixel alignment.

### Harmony Anchor (Geometric Median Selection)
To avoid outlier reference selection (e.g. choosing a frame skewed by massive camera shake), the reference frame is chosen by solving the geometric median of the translation coordinates:
$$\mathbf{t}_{\text{median}} = \arg\min_{\mathbf{x} \in \mathbb{R}^2} \sum_{i=1}^N \|\mathbf{t}_i - \mathbf{x}\|_2$$
We solve this using Weiszfeld's iterative algorithm:
$$\mathbf{x}^{(k+1)} = \frac{\sum_{i=1}^N \frac{\mathbf{t}_i}{\|\mathbf{t}_i - \mathbf{x}^{(k)}\|_2}}{\sum_{i=1}^N \frac{1}{\|\mathbf{t}_i - \mathbf{x}^{(k)}\|_2}}$$
After locating $\mathbf{t}_{\text{median}}$, candidate frames within the $50\text{th}$ percentile of distance to the median are collected. Among these, the frame with the highest Laplacian variance is chosen as the reference:
$$\text{Sharpness}(I) = \text{Var}(\text{Laplacian}(I))$$

---

## 2. Dynamic Foreground Masking (`masking.py`)

To prevent ghosting artifacts, we compute a normalized difference mask.
We model sensor noise locally using a Poisson-Gaussian model:
$$\sigma_{\text{noise}}(x,y) = \sqrt{a \cdot I_{\text{ref}}(x,y) + b}$$
where $a = 0.5$ represents the photon noise gain, and $b = 1.0$ represents read/system noise.
To prevent false-positive motion detection at high-frequency edges due to sub-pixel interpolation errors, we include the gradient magnitude in the denominator:
$$\text{denom}(x,y) = \sigma_{\text{noise}}(x,y) + c \cdot \|\nabla I_{\text{ref}}(x,y)\|_2 + 1.0$$
where $c = 0.3$. The normalized absolute difference for frame $i$ is:
$$D_{\text{norm}, i}(x,y) = \frac{|I_{i,\text{warped}}(x,y) - I_{\text{ref}}(x,y)|}{\text{denom}(x,y)}$$
Dual-thresholding is applied:
* Background motion: $D_{\text{norm}, i} > 1.5$ (dilated with a $7\times7$ kernel)
* Subject motion: $D_{\text{norm}, i} > 3.0$ (dilated with an $11\times11$ kernel, 2 iterations)
A soft weight map $W_i(x,y) \in [0, 1]$ is produced by Gaussian smoothing the combined binary motion mask.

---

## 3. Pre-flight Scale Bounding (`preflight.py`)

The theoretical resolution limit of the Drizzle stack is bounded by two physical constraints:
1. **Sampling Density Limit**: Based on the spatial sampling theorem, the density of clean, non-moving spatial data bounds the resolution. The density limit is:
   $$S_{\text{density}} = \sqrt{N \cdot R_{\text{global}}}$$
2. **Alignment Blur Limit (CRLB)**: Registration errors and concentrated sub-pixel coverage introduce alignment drift variance $\sigma_{\text{align}}^2 \propto \frac{1 - Q_{\text{dither}}}{Q_{\text{dither}}}$. Based on the Cramer-Rao Lower Bound (CRLB) of sub-pixel phase reconstruction, to prevent alignment drift blur, the upscale factor is capped by:
   $$S_{\text{blur}} = \alpha \sqrt{\frac{Q_{\text{dither}}}{1 - Q_{\text{dither}}}}$$
   where $Q_{\text{dither}}$ is the sub-pixel dither quality, and $\alpha = 0.75$ is the optical decay constant.

The final scale factor is bounded dynamically:
$$S_{\text{final}} = \min(S_{\text{target}}, S_{\text{density}}, S_{\text{blur}})$$

---

## 4. Vectorized Drizzle Stacking (`drizzle.py`)

Optico implements Variable-Pixel Linear Reconstruction (Drizzle) by mapping each input pixel onto the high-resolution grid.
To optimize performance, Optico projects the entire grid using a vectorized affine warp:
$$x_{\text{HR}} = S \cdot (x_{\text{LR}} - t_x), \quad y_{\text{HR}} = S \cdot (y_{\text{LR}} - t_y)$$
The shrunken pixel footprint is modeled using the pixel fraction factor $p = \text{pixfrac} \in [0.0, 1.0]$. The droplet area scales the warped mask weight:
$$W_{\text{drizzle}, i} = W_i \cdot p^2$$
For each horizontal chunk (strip), the high-resolution canvas is accumulated:
$$\text{Num}(x,y) = \sum_{i=1}^N W_{\text{drizzle}, i}(x,y) \cdot I_{i,\text{warped}}(x,y)$$
$$\text{Den}(x,y) = \sum_{i=1}^N W_{\text{drizzle}, i}(x,y)$$
The final normalized chunk pixel value is:
$$I_{\text{HR}}(x,y) = \frac{\text{Num}(x,y)}{\max(\text{Den}(x,y), 10^{-6})}$$

---

## 5. Adaptive Dual-Band Wiener Deconvolution (`deconvolution.py`)

### Corrected Noise MAD Formula
The global noise floor standard deviation $\sigma_{\text{noise}}$ is estimated from the Laplacian of the image. For a discrete 3x3 Laplacian operator with kernel $K_{\Delta}$, convolving i.i.d. Gaussian noise with variance $\sigma^2$ amplifies the output variance to $\sigma_{out}^2 = \sigma^2 \sum K(u,v)^2 = 20\sigma^2$. Thus, the standard deviation is amplified by $\sqrt{20}$.
To recover the original noise floor standard deviation $\sigma_{\text{noise}}$ from the Median Absolute Deviation (MAD) of the Laplacian image, we must scale by $1/\sqrt{20}$:
$$\sigma_{\text{noise}} = \frac{1.4826 \cdot \text{median}(|\text{Lap}(I) - \text{median}(\text{Lap}(I))|)}{\sqrt{20}}$$
The Wiener noise regularization parameter is:
$$K_{\text{est}} = \text{clamp}(\sigma_{\text{noise}}^2, 0.001, 0.08)$$

### Point Spread Function (PSF) Circulant Padding
The Optical Transfer Function $H(u,v)$ is computed from the Gaussian PSF kernel:
$$h(x,y) = \frac{1}{2\pi \sigma_{\text{PSF}}^2} \exp\left(-\frac{x^2+y^2}{2\sigma_{\text{PSF}}^2}\right)$$
where $\sigma_{\text{PSF}} = \max(0.6, 0.4 \cdot S_{\text{final}})$.
Instead of generating an image-sized PSF array directly, we generate a small kernel of size $2\lceil 3\sigma_{\text{PSF}} \rceil + 1$, and embed it into the corners of an empty image-sized canvas using wrap-around indices (circulant padding). This ensures that the center of the PSF aligns with $(0,0)$ when computing the FFT:
$$H(u,v) = \mathcal{F}\{h_{\text{padded}}(x,y)\}$$

### Dual-Band Spatial Blending
Two deconvolution regularizers are defined:
$$K_{\text{strong}} = \max(2 \cdot K_{\text{est}}, 0.01), \quad K_{\text{weak}} = \max(6 \cdot K_{\text{est}}, 0.03)$$
Two parallel restorations are computed in the frequency domain:
$$\hat{F}_{\text{strong}}(u,v) = \left( \frac{H^*(u,v)}{|H(u,v)|^2 + K_{\text{strong}}} \right) G(u,v)$$
$$\hat{F}_{\text{weak}}(u,v) = \left( \frac{H^*(u,v)}{|H(u,v)|^2 + K_{\text{weak}}} \right) G(u,v)$$
We transform both back to the spatial domain:
$$I_{\text{strong}}(x,y) = \mathcal{F}^{-1}\{\hat{F}_{\text{strong}}(u,v)\}, \quad I_{\text{weak}}(x,y) = \mathcal{F}^{-1}\{\hat{F}_{\text{weak}}(u,v)\}$$
A soft Canny edge mask $M_{\text{edge}}(x,y) \in [0, 1]$ is generated. The final sharpened image is a spatial blend of the two restorations:
$$I_{\text{final}}(x,y) = M_{\text{edge}}(x,y) \cdot I_{\text{weak}}(x,y) + (1 - M_{\text{edge}}(x,y)) \cdot I_{\text{strong}}(x,y)$$
This keeps edges protected from ringing while fully sharpening flat texture regions.
