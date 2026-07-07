# Optico: 核心演算法與底層數學推導

本文件詳細記錄了驅動 Optico 多幀超解析 (MFSR) 引擎的具體數學模型、公式與演算法執行步驟。

---

## 1. 亞像素配准與抖動圓統計 (`alignment.py`)

### 亞像素平移映射
在連拍攝影中（包含手持連拍或腳架微移連拍），相機的相對位移主要以剛性平移為主，為防範高頻雜訊過擬合，影像平移參數以剛性平移來建模。每一幀 $I_i$ 與參考幀 $I_{ref}$ 的相對對齊矩陣透過最大化增強相關係數 (ECC) 來求解：
$$E(M_i) = \max \text{ECC}(I_{ref}, I_i(M_i))$$
其中平移矩陣為：
$$M_i = \begin{bmatrix} 1 & 0 & t_{x,i} \\ 0 & 1 & t_{y,i} \end{bmatrix}$$
為加速收斂並降低噪聲干擾，配准是在降采樣的影像上進行（縮放因子 $\gamma = 0.5$），隨後將計算得到的平移量放大回原始解析度：
$$t_{x,\text{original}} = \frac{t_{x,\text{downscaled}}}{\gamma}$$
$$t_{y,\text{original}} = \frac{t_{y,\text{downscaled}}}{\gamma}$$

### 2D 圓統計 resultant vector 估計 (Dither Quality)
為了在重構時還原亞像素細節，抖動偏移量必須均勻覆蓋一個像素的內部空間。Optico 使用 2D 環面上的圓統計來量化這種分佈。

我們提取所有平移偏移量的小數部分：
$$f_{x,i} = t_{x,i} - \lfloor t_{x,i} \rfloor, \quad f_{y,i} = t_{y,i} - \lfloor t_{y,i} \rfloor$$
並將其映射至單位圓上的相位角：
$$\theta_{x,i} = 2\pi f_{x,i}, \quad \theta_{y,i} = 2\pi f_{y,i}$$
各軸的平均合成向量長度為：
$$R_x = \sqrt{\left(\frac{1}{N}\sum_{i=1}^N \cos\theta_{x,i}\right)^2 + \left(\frac{1}{N}\sum_{i=1}^N \sin\theta_{x,i}\right)^2}$$
$$R_y = \sqrt{\left(\frac{1}{N}\sum_{i=1}^N \cos\theta_{y,i}\right)^2 + \left(\frac{1}{N}\sum_{i=1}^N \sin\theta_{y,i}\right)^2}$$
為了捕捉坐標間的相關性，聯合 2D 合成向量長度定義為：
$$R_{2D} = \sqrt{R_x \cdot R_y}$$
最終的抖動品質得分為：
$$Q_{\text{dither}} = 1 - R_{2D} \in [0.0, 1.0]$$
當 $Q_{\text{dither}} \to 1.0$ 時代表次像素抖動非常均勻，有利於超解析重建；$Q_{\text{dither}} \to 0.0$ 則代表所有抖動的次像素位置高度重合，無法提供額外細節。

### 和諧定錨 (Harmony Anchor)
為了防止錨定在嚴重晃動或模糊的異常幀，參考幀是藉由求解所有位移坐標的 **幾何中位數 (Geometric Median)** 來決定的：
$$\mathbf{t}_{\text{median}} = \arg\min_{\mathbf{x} \in \mathbb{R}^2} \sum_{i=1}^N \|\mathbf{t}_i - \mathbf{x}\|_2$$
我們使用 Weiszfeld 迭代法求解：
$$\mathbf{x}^{(k+1)} = \frac{\sum_{i=1}^N \frac{\mathbf{t}_i}{\|\mathbf{t}_i - \mathbf{x}^{(k)}\|_2}}{\sum_{i=1}^N \frac{1}{\|\mathbf{t}_i - \mathbf{x}^{(k)}\|_2}}$$
取得中位數 $\mathbf{t}_{\text{median}}$ 後，篩選出距離中位數最近的前 50% 候選幀，並在其中選擇 Laplacian 變異數最大（最清晰）的幀作為參考幀：
$$\text{Sharpness}(I) = \text{Var}(\text{Laplacian}(I))$$

---

## 2. 動態前景運動遮罩 (`masking.py`)

為了防範運動殘影與鬼影，系統計算了歸一化幀差遮罩。
局部感測器雜訊採用泊松-高斯混合模型進行估計：
$$\sigma_{\text{noise}}(x,y) = \sqrt{a \cdot I_{\text{ref}}(x,y) + b}$$
其中 $a = 0.5$ 代表光子雜訊增益，$b = 1.0$ 代表系統與讀取底噪。
為了避免亞像素插值誤差在影像邊緣處引發運動誤判，我們在分母中加入梯度強度項：
$$\text{denom}(x,y) = \sigma_{\text{noise}}(x,y) + c \cdot \|\nabla I_{\text{ref}}(x,y)\|_2 + 1.0$$
其中 $c = 0.3$。第 $i$ 幀的歸一化絕對差值圖為：
$$D_{\text{norm}, i}(x,y) = \frac{|I_{i,\text{warped}}(x,y) - I_{\text{ref}}(x,y)|}{\text{denom}(x,y)}$$
運動判斷採用雙閾值：
* 背景微動：$D_{\text{norm}, i} > 1.5$ (以 $7\times7$ 結構元素進行膨脹)
* 主體運動：$D_{\text{norm}, i} > 3.0$ (以 $11\times11$ 結構元素膨脹，迭代 2 次)
結合後的二值化遮罩經高斯模糊平滑後，輸出 $[0.0, 1.0]$ 的連續軟性權重圖 $W_i(x,y)$。

---

## 3. Pre-flight 安全倍率限制 (`preflight.py`)

Drizzle 重構的最高安全倍率由兩個物理極限共同制約：
1. **空間取樣密度極限**：依據取樣定理，乾淨且無運動的空間取樣數據密度限制了解析度。取樣密度極限為：
   $$S_{\text{density}} = \sqrt{N \cdot R_{\text{global}}}$$
2. **對齊衰減模糊極限 (CRLB)**：配准誤差與高度集中的次像素相位覆蓋會引入對齊漂移變異數 $\sigma_{\text{align}}^2 \propto \frac{1 - Q_{\text{dither}}}{Q_{\text{dither}}}$。根據次像素相位重建的 Cramer-Rao Lower Bound (CRLB) 限制，為避免點擴散函數 (PSF) 散焦與對齊漂移模糊，倍率上限被限制為：
   $$S_{\text{blur}} = \alpha \sqrt{\frac{Q_{\text{dither}}}{1 - Q_{\text{dither}}}}$$
   其中 $Q_{\text{dither}}$ 為次像素抖動品質，而 $\alpha = 0.75$ 為光學衰減常數。

最終生效的放大倍率被限制在物理安全天花板之下：
$$S_{\text{final}} = \min(S_{\text{target}}, S_{\text{density}}, S_{\text{blur}})$$

---

## 4. 向量化 Drizzle 疊加 (`drizzle.py`)

Optico 通過將輸入像素映射到高解析度網格來實現 Variable-Pixel Linear Reconstruction (Drizzle)。
為了優化效能，我們利用仿射變換將輸入影像及遮罩整體投影至 HR 網格：
$$x_{\text{HR}} = S \cdot (x_{\text{LR}} - t_x), \quad y_{\text{HR}} = S \cdot (y_{\text{LR}} - t_y)$$
微點收縮率為 $p = \text{pixfrac} \in [0.0, 1.0]$，疊加時的權重依微點面積進行縮放：
$$W_{\text{drizzle}, i} = W_i \cdot p^2$$
HR 畫布以水平條帶為單位進行分塊累加：
$$\text{Num}(x,y) = \sum_{i=1}^N W_{\text{drizzle}, i}(x,y) \cdot I_{i,\text{warped}}(x,y)$$
$$\text{Den}(x,y) = \sum_{i=1}^N W_{\text{drizzle}, i}(x,y)$$
最後，每個分塊進行歸一化輸出：
$$I_{\text{HR}}(x,y) = \frac{\text{Num}(x,y)}{\max(\text{Den}(x,y), 10^{-6})}$$

---

## 5. 自適應雙頻段 Wiener 反捲積 (`deconvolution.py`)

### 修正後的噪聲 MAD 估計
Drizzle 輸出影像的底噪標準差 $\sigma_{\text{noise}}$ 是通過 Laplacian 梯度的中位數絕對偏差 (MAD) 動態估計的。對於離散的 3x3 Laplacian 算子（卷積核為 $K_{\Delta}$），當輸入變異數為 $\sigma^2$ 的獨立同分佈高斯噪聲時，卷積輸出的噪聲變異數會被放大為 $\sigma_{out}^2 = \sigma^2 \sum K(u,v)^2 = 20\sigma^2$。因此，噪聲標準差被放大了 $\sqrt{20}$ 倍。
為了從 Laplacian 影像的 MAD 值還原真實的影像底噪標準差 $\sigma_{\text{noise}}$，必須除以該核心的噪聲放大係數 $\sqrt{20}$：
$$\sigma_{\text{noise}} = \frac{1.4826 \cdot \text{median}(|\text{Lap}(I) - \text{median}(\text{Lap}(I))|)}{\sqrt{20}}$$
頻域維納濾波的正則化係數估計值為：
$$K_{\text{est}} = \text{clamp}(\sigma_{\text{noise}}^2, 0.001, 0.08)$$

### PSF 圓周卷積填充 (Circulant Padding)
系統將 PSF 建模為高斯點擴散函數：
$$h(x,y) = \frac{1}{2\pi \sigma_{\text{PSF}}^2} \exp\left(-\frac{x^2+y^2}{2\sigma_{\text{PSF}}^2}\right)$$
其中 $\sigma_{\text{PSF}} = \max(0.6, 0.4 \cdot S_{\text{final}})$。
為避免在大矩陣上進行無謂的高斯計算，我們先生成一個尺寸僅為 $2\lceil 3\sigma_{\text{PSF}} \rceil + 1$ 的小型實體核心，隨後將其各頂點透過取模運算 (modulo) 映射嵌入至影像尺寸的畫布角落。這種 Circulant Embedding 確保了 PSF 的能量重心恰好對齊在 $(0,0)$，做 FFT 運算時無須再進行 `fftshift`。
$$H(u,v) = \mathcal{F}\{h_{\text{padded}}(x,y)\}$$

### 空間域雙頻混合 (Spatial Blending)
我們定義了兩種正則化強度：
$$K_{\text{strong}} = \max(2 \cdot K_{\text{est}}, 0.01), \quad K_{\text{weak}} = \max(6 \cdot K_{\text{est}}, 0.03)$$
在頻域並行計算兩種重建強度：
$$\hat{F}_{\text{strong}}(u,v) = \left( \frac{H^*(u,v)}{|H(u,v)|^2 + K_{\text{strong}}} \right) G(u,v)$$
$$\hat{F}_{\text{weak}}(u,v) = \left( \frac{H^*(u,v)}{|H(u,v)|^2 + K_{\text{weak}}} \right) G(u,v)$$
轉回空間域後：
$$I_{\text{strong}}(x,y) = \mathcal{F}^{-1}\{\hat{F}_{\text{strong}}(u,v)\}, \quad I_{\text{weak}}(x,y) = \mathcal{F}^{-1}\{\hat{F}_{\text{weak}}(u,v)\}$$
最後，使用 Canny 邊緣檢測並經高斯模糊生成邊緣遮罩 $M_{\text{edge}}(x,y) \in [0, 1]$，在空間域進行邊緣感知混合：
$$I_{\text{final}}(x,y) = M_{\text{edge}}(x,y) \cdot I_{\text{weak}}(x,y) + (1 - M_{\text{edge}}(x,y)) \cdot I_{\text{strong}}(x,y)$$
在保護高對比度邊緣不產生白邊與振鈴的同時，最大限度地恢復平坦紋理區的清晰度。
