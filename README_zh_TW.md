# Optico 📸

**多幀超解析 (MFSR) 攝影引擎 — 純粹光學數據，追求極致真實。**

Optico 是一套頂級的計算攝影引擎，專為從連拍照片（例如 4 到 8 張，包括腳架微移連拍或手持連拍）中榨取極致高畫素細節與純淨度而生。生成式 AI 在創意與修圖領域表現卓越，而 Optico 則專注於另一條道路：嚴格遵守光學物理，只還原真實存在的次像素 (Sub-pixel) 資訊，保留最純粹的攝影本質。使用腳架搭配微位移抖動（如感光元件位移或微小震動）進行連拍，是實現次像素超解析度重建最理想的條件。

---

## 🌟 核心架構

Optico 引擎建立在三大光學物理支柱之上：

### 1. 雙向預估防護網 (嚴謹光學箝制)
盲目地對連拍影像進行超高倍率放大，必然導致災難性的「對齊衰減模糊 (Alignment Drift Blur)」。Optico 採用了極先進的前置預估遮罩分析，提前算出「全域保留率 ($R_{global}$)」。
基於 **空間取樣定理 (Nyquist Limit)** 與 **Cramer-Rao Lower Bound (CRLB)**，引擎會計算影像對齊的絕對訊噪比 (SNR)，並透過嚴謹的數學公式算出「安全倍率天花板」，徹底淘汰盲目的經驗猜測。
* **靜態場景**：放行至空間密度的絕對極限 (例如 2.5x - 2.8x)。
* **動態場景**：精準箝制倍率 (例如 1.3x - 1.8x)，將多餘張數完美轉化為無損降噪。

### 2. 自適應雙頻譜邊緣感知反捲積
寫死的銳化參數是光學的死穴。Optico 透過修正後的 Laplacian MAD 動態偵測 Drizzle 疊加後的真實物理底噪，接著在傅立葉頻域 (Fourier Domain) 中進行雙軌拆分，並於空間域進行軟體融合：
* **平坦區**：注入重火力反捲積（低 K 值）以榨出紋理。
* **邊緣區**：啟動嚴密防護網（高 K 值），徹底消滅白邊 (Ringing) 與十字網格瑕疵。

### 3. 水平記憶體切割 (Active Memory Chunking)
超大畫素的 Drizzle 疊加運算（例如將 8 張 2,400 萬畫素照片疊加成 1.5 億畫素）往往會引發 Swap Thrashing 導致電腦卡死。
Optico 將目標高解析度畫布進行水平條狀切割，將分塊處理與垃圾回收結合，將峰值記憶體死死鎖定在 3GB 以內，免疫 OOM (Out of Memory) 災難。

---

## 🛠️ 程式碼結構

Optico 目前已完全模組化在 `backend/` 目錄下：
* [__init__.py](file:///C:/Users/chchen/Optico_git/backend/__init__.py): 引擎版本宣告與主要 API 匯出。
* [constants.py](file:///C:/Users/chchen/Optico_git/backend/constants.py): 配置類別 (`OpticoConfig`) 與所有光學常數的集中地。
* [alignment.py](file:///C:/Users/chchen/Optico_git/backend/alignment.py): 亞像素 ECC 影像配准、幾何中位數參考幀選擇 (Harmony Anchor)、以及 2D 圓統計分析。
* [masking.py](file:///C:/Users/chchen/Optico_git/backend/masking.py): 基於泊松-高斯雜訊模型的雙閥值動態運動遮罩計算。
* [preflight.py](file:///C:/Users/chchen/Optico_git/backend/preflight.py): 基於奈奎斯特極限與 CRLB 的安全解析度天花板計算。
* [drizzle.py](file:///C:/Users/chchen/Optico_git/backend/drizzle.py): 水平記憶體分塊的向量化 Variable-Pixel Linear Reconstruction 疊加核心。
* [deconvolution.py](file:///C:/Users/chchen/Optico_git/backend/deconvolution.py): 頻域雙頻 Wiener 反捲積與空間域邊緣感知混合。
* [pipeline.py](file:///C:/Users/chchen/Optico_git/backend/pipeline.py): Pipeline 流程協調與命令列介面 (CLI) 進入點。

---

## ⚙️ 環境建置與安裝

Optico 需要 Python 3.10+ 及標準的科學計算套件。

1. 安裝相依套件：
   ```bash
   pip install -r requirements.txt
   ```

2. 驗證模組是否可正常匯入：
   ```bash
   python -c "import backend; print(backend.__version__)"
   ```

---

## 🚀 使用指南

### 命令列介面 (CLI)

直接從命令列執行端到端 Pipeline：

```bash
# 基本執行：使用預設值（2.0x 放大，輸出至 optico_output.png）處理連拍目錄
python -m backend.pipeline --input path/to/burst_folder --output output.png

# 進階執行：請求 3.0x 放大，設定 pixfrac 0.6，分割成 12 個記憶體分塊，並顯示詳細日誌
python -m backend.pipeline --input path/to/burst_folder --scale 3.0 --pixfrac 0.6 --chunks 12 --verbose

# 執行但不啟用維納反捲積銳化
python -m backend.pipeline --input path/to/burst_folder --no-deconv
```

### Python API 調用範例

您可以直接將 Optico 整合至您自己的 Python 計算攝影工作流中：

```python
import cv2
from backend import OpticoConfig, run_pipeline

# 1. 載入連拍影像 (BGR uint8)
image_paths = ["frame0.png", "frame1.png", "frame2.png", "frame3.png"]
images = [cv2.imread(p) for p in image_paths]

# 2. 建立設定參數
config = OpticoConfig(
    target_scale=2.5,        # 期望放大倍率 (可能會被 Pre-flight 限制)
    pixfrac=0.7,             # Drizzle 微點收縮比例
    num_chunks=8,            # 記憶體分塊數
    skip_deconv=False        # 是否套用維納銳化
)

# 3. 執行超解析度 Pipeline
hr_result = run_pipeline(images, config=config, output_path="mfsr_result.png")
```

---

## 🛠️ 下一步：雙軌頻率融合 (Dual-Track Merging)
Optico 正朝向終極的頻率分離架構邁進：
* **高頻細節軌**：僅挑選對齊最完美的「菁英張數 (cc > 0.95)」進行疊加，推升解析度極限。
* **低頻降噪軌**：榨乾「所有張數」的數據來鋪平暗部雜訊，最後透過拉普拉斯金字塔 (Laplacian Pyramid) 達成純淨度與解析度的終極融合。

---
*為追求絕對的光學真實而生。*
