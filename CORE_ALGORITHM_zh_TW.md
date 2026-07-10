# Optico: 核心演算法與底層數學推導

本文件詳細記錄了驅動 Optico 多幀超解析 (MFSR) 引擎的具體數學模型、公式與演算法執行步驟。

---

## 0. JPEG vs RAW 來源偵測 (`pipeline.py`)

在所有處理開始之前，Optico 會自動偵測輸入是否為 JPEG 來源——讀取每張照片的前 2 bytes，檢查是否為 JPEG SOI 標記（`0xFF 0xD8`）。結果存入 `config.jpeg_input`，並向下游三個 Phase 傳遞：

| Phase | RAW / PNG | JPEG |
|---|---|---|
| Phase 2 — ECC 對齊 | `gauss_filt_size = 5` | `gauss_filt_size = 7` |
| Phase 8 — Drizzle | coverage-hole 填補（通用） | 同左 |
| Phase 9 — 反捲積 | PSF σ = `0.4·S`，HF fraction = 0.75 | PSF σ ×1.35，HF fraction = 0.60 |

可透過 `config.jpeg_input = True/False` 或 CLI 旗標 `--jpeg` / `--raw` 強制覆蓋。

---

## 1. 亞像素配准與抖動品質 (`alignment.py`)

### 亞像素平移映射
在連拍攝影中（包含手持連拍或腳架微移連拍），相機的相對位移主要以剛性平移為主，為防範高頻雜訊過擬合，影像平移參數以剛性平移來建模。每一幀 $I_i$ 與參考幀 $I_{ref}$ 的相對對齊矩陣透過最大化增強相關係數 (ECC) 來求解：
$$E(M_i) = \max \text{ECC}(I_{ref}, I_i(M_i))$$
其中平移矩陣為：
$$M_i = \begin{bmatrix} 1 & 0 & t_{x,i} \\ 0 & 1 & t_{y,i} \end{bmatrix}$$
為加速收斂並降低噪聲干擾，配准是在降采樣的影像上進行（縮放因子 $\gamma = 0.5$），隨後將計算得到的平移量放大回原始解析度：
$$t_{x,\text{original}} = \frac{t_{x,\text{downscaled}}}{\gamma}, \quad t_{y,\text{original}} = \frac{t_{y,\text{downscaled}}}{\gamma}$$

**JPEG 特別處理：** JPEG 的 8×8 DCT 分塊邊界會產生假高頻梯度，ECC 有可能將這些 inter-block 不連續誤判為真實的次像素位移。JPEG 輸入時，ECC 的前置高斯濾波核心從 5 → 7 px（`JPEG_ECC_GAUSS_FILT_SIZE`），以平滑 block 邊緣後再進行配准。

### N_eff 熵值抖動品質
次像素偏移必須均勻覆蓋一個像素的內部空間。Optico 使用 4×4 次像素直方圖的 Shannon 熵來量化覆蓋品質：
$$H = -\sum_{k} p_k \log_2 p_k$$
$$N_{\text{eff}} = 2^H \in [1.0,\ 16.0]$$
$N_{\text{eff}}$ 越高代表次像素位置越獨立均勻；Pre-flight 以 $\sqrt{N_{\text{eff}}}$ 作為抖動貢獻計算倍率上限。

> **歷史說明：** 舊版使用 2D Rayleigh resultant 加小樣本偏誤校正（$\pi/4N$ 基準）。該方法在 8 個測試分佈中有 6 個回傳 $Q=1.0$（相當於關閉 pre-flight 分支）。$N_{\text{eff}}$ 熵值度量數值穩定且物理意義清楚。

### 和諧定錨 (Harmony Anchor)
為了防止錨定在嚴重晃動或模糊的異常幀，參考幀是藉由求解所有位移坐標的幾何中位數來決定的：
$$\mathbf{t}_{\text{median}} = \arg\min_{\mathbf{x}} \sum_{i=1}^N \|\mathbf{t}_i - \mathbf{x}\|_2$$
使用 Weiszfeld 迭代法求解。取得中位數後，篩選距離中位數最近的前 50% 候選幀，並選擇 Laplacian 變異數最大（最清晰）的幀作為參考幀：
$$\text{Sharpness}(I) = \text{Var}(\nabla^2 I)$$

---

## 2. 動態前景運動遮罩 (`masking.py`)

為了防範運動殘影與鬼影，系統計算了歸一化幀差遮罩。
局部感測器雜訊採用泊松-高斯混合模型進行估計：
$$\sigma_{\text{noise}}(x,y) = \sqrt{a \cdot I_{\text{ref}}(x,y) + b}, \quad a=0.5,\ b=1.0$$
為避免亞像素插值誤差在邊緣處引發運動誤判，分母中加入梯度強度項：
$$\text{denom}(x,y) = \sigma_{\text{noise}}(x,y) + 0.3 \cdot \|\nabla I_{\text{ref}}(x,y)\|_2 + 1.0$$
$$D_{\text{norm}, i}(x,y) = \frac{|I_{i,\text{warped}}(x,y) - I_{\text{ref}}(x,y)|}{\text{denom}(x,y)}$$
運動判斷採用雙閾值：
- 背景微動：$D_{\text{norm}} > 1.5$（7×7 膨脹）
- 主體運動：$D_{\text{norm}} > 3.0$（11×11 膨脹，迭代 2 次）

結合後的二值化遮罩經高斯模糊平滑後，輸出 $[0.0, 1.0]$ 的連續軟性權重圖 $W_i(x,y)$。

---

## 3. Pre-flight 安全倍率限制 (`preflight.py`)

Drizzle 重構的最高安全倍率由兩個物理極限共同制約：
1. **空間取樣密度極限：**
   $$S_{\text{density}} = \sqrt{N \cdot R_{\text{global}}}$$
2. **對齊衰減模糊極限 (CRLB)**：以 $N_{\text{eff}}$ 作為抖動品質度量：
   $$S_{\text{blur}} = \alpha \cdot \sqrt{N_{\text{eff}}}, \quad \alpha = 0.75$$

最終生效的放大倍率：
$$S_{\text{final}} = \min(S_{\text{target}},\ S_{\text{density}},\ S_{\text{blur}})$$

---

## 4. 向量化 Drizzle 疊加 (`drizzle.py`)

Optico 實作 Variable-Pixel Linear Reconstruction（Fruchter & Hook 2002），適配手持連拍攝影。

每個 LR 像素透過仿射變換投影至 HR 網格：
$$x_{\text{HR}} = S \cdot (x_{\text{LR}} - t_x), \quad y_{\text{HR}} = S \cdot (y_{\text{LR}} - t_y)$$

液滴半徑為：
$$r_{\text{drop}} = \frac{p \cdot S}{2}, \quad p = \text{pixfrac} \in [0,1]$$

分塊累加：
$$\text{Num}(x,y) = \sum_{i=1}^N \text{overlap}_i(x,y) \cdot W_i(x,y) \cdot I_i(x,y)$$
$$\text{Den}(x,y) = \sum_{i=1}^N \text{overlap}_i(x,y) \cdot W_i(x,y)$$
$$I_{\text{HR}}(x,y) = \frac{\text{Num}(x,y)}{\max(\text{Den}(x,y),\ 10^{-6})}$$

### Coverage-Hole 填補

反向 4-鄰域 overlap 核心有一個結構性盲點：當最近的 LR 像素中心投影位置距 HR 像素中心超過 $r_{\text{drop}}$ 時，overlap = 0。若所有幀的次像素偏移碰巧都落在同一個死角，這些 HR 像素在每幀中都是 under-covered，歸一化後放大數值雜訊，形成肉眼可見的格子狀亮暗紋。

**修復方式：** 累加完成、歸一化之前，對滿足下列條件的 HR 像素標記為 coverage hole（$\tau = 0.35$）：
$$\text{Den}(x,y) < \tau \cdot \text{median}(\text{Den})$$
並以 3×3 Gaussian 鄰域平均填補其 numerator 與 denominator：
$$\text{Den}_{\text{filled}}(x,y) = (\text{uniform\_filter}_{3\times3} * \text{Den})(x,y)$$
此操作等同於從周圍覆蓋良好的像素做雙線性插值。僅 hole 像素被修改，正常像素完全不受影響。

---

## 5. 頻率相依 Wiener 反捲積 (`deconvolution.py`)

### 噪聲估計
Drizzle 輸出的底噪標準差 $\sigma_{\text{noise}}$ 使用 Laplacian MAD 動態估計：
$$\sigma_{\text{noise}} = \frac{1.4826 \cdot \text{MAD}(\nabla^2 I)}{\sqrt{20}}$$
$\sqrt{20}$ 係數補正 3×3 Laplacian 算子的噪聲放大效應。

### PSF 模型
光學 PSF 建模為高斯函數：
$$\sigma_{\text{PSF}} = \max(0.6,\ 0.4 \cdot S_{\text{final}})$$

**JPEG 補正：** JPEG 量化會在光學 PSF 之外疊加一個量化模糊 PSF（$\sigma_{\text{JPEG}} \approx 0.3\text{–}0.5\ \text{px}$），複合有效 sigma 為：
$$\sigma_{\text{eff}} = \sqrt{\sigma_{\text{optical}}^2 + \sigma_{\text{JPEG}}^2} \approx \sigma_{\text{optical}} \times 1.35$$
JPEG 輸入時，`psf_sigma` 乘以 `JPEG_PSF_SCALE_FACTOR = 1.35`。

### 頻率相依正則化 $K(f)$

與舊版標量 $K$ 不同，Optico 從影像自身功率頻譜估計逐頻率正則化映射：

1. **定位噪聲基準面：** 從 $f_{\text{lo}} \times f_{\text{Nyquist}}$ 開始向外掃描徑向功率頻譜。找到 median power 梯度低於 DC 區功率 5% 的平台區，取該平台的 median 為 $N_{\text{floor}}$。

   > **JPEG 修復：** JPEG 輸入時，掃描起點從 $0.75$ 降至 $0.60 \times f_{\text{Nyquist}}$（`JPEG_NOISE_FLOOR_HIGH_FREQ_FRACTION`）。JPEG DCT 量化在約 0.55–0.65 × Nyquist 處造成硬性頻率截止；若從預設的 0.75 起掃，平台偵測器會把 JPEG 截止帶誤判為白噪聲基準，將 $N_{\text{floor}}$ 高估 2–5 倍，導致整體頻率的 $K(f)$ 過大，高頻細節過度壓制，輸出模糊。

2. **逐頻率信號功率：**
   $$S(f) = \max\bigl(P(f) - N_{\text{floor}},\ N_{\text{floor}} \cdot 0.005\bigr)$$

3. **逐頻率 $K$：**
   $$K(f) = \text{clip}\!\left(\frac{N_{\text{floor}}}{S(f)},\ 10^{-4},\ 200\right)$$

### Wiener 濾波器與 DC 增益保護
$$\hat{F}(u,v) = \frac{H^*(u,v)}{|H(u,v)|^2 + K(u,v)} \cdot G(u,v)$$
DC bin $[0,0]$ 強制設為 unity gain，防止平均亮度偏移：
$$W_{\text{resp}}(0,0) = 1.0$$

> **為何頻率相依 $K$ 優於舊版雙頻段 Canny 混合：** 自然影像功率頻譜以 $\sim 1/f^2$ 衰減，而感測器噪聲近似白噪聲（平坦功率）。單一標量 $K$ 無法同時處理兩個頻段。舊版雙頻段方案用 Canny 邊緣遮罩做代理，但每個頻段內部仍套用平坦的 $K$。逐頻率方案直接符合教科書 Wiener SNR-inverse 解，平均 PSNR 提升 +1.54 dB（噪聲 σ = 1–9 全範圍）。
