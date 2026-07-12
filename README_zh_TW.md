# Optico

**專為手持連拍設計的多幀超解析 (MFSR) 引擎。**

Optico 融合連拍序列中的多張照片，透過次像素配准、動態遮罩、Drizzle 疊加與自適應 Wiener 反捲積，輸出單張高解析度影像。

---

## 快速開始

```bash
pip install -r requirements.txt
python -m backend.pipeline --input ./burst --output result.png
```

## CLI 參數一覽

| 旗標 | 預設值 | 說明 |
|---|---|---|
| `--input` / `-i` | *(必填)* | 連拍影像資料夾 |
| `--output` / `-o` | *(自動生成)* | 輸出路徑 |
| `--scale` / `-s` | `0.0` (自動) | 目標放大倍率 (0.0 = 依據 N_eff 抖動品質自動解算) |
| `--pixfrac` | `0.7` | Drizzle 像素分率（0–1）|
| `--chunks` | `8` | Drizzle 記憶體分塊數 |
| `--no-deconv` | 關 | 跳過 Wiener 反捲積 |
| `--psf-base` | `0.63` | Wiener 反捲積 PSF 縮放常數（廣角 17mm 設 0.35，長焦 50mm 設 0.63） |
| `--psf-override` | None | 手動指定 HR 去模糊 PSF 標準差（會跳過縮放與安全上限） |
| `--align-scale` | 自動 | 覆蓋 ECC 縮放因子 |
| `--jpeg` | 自動 | 強制 JPEG 模式 |
| `--raw` | 自動 | 強制 RAW/PNG 模式 |
| `--no-cache` | 關 | 停用 Drizzle 快取 |
| `--cache-dir` | `~/.optico_cache` | 自訂快取目錄 |
| `--verbose` / `-v` | 關 | 開啟 debug 日誌 |

### JPEG vs RAW 物理焦距 PSF 定錨

Optico 透過讀取檔案 header（JPEG SOI 標記 `0xFF 0xD8`）自動偵測輸入格式。JPEG 輸入時自動啟用下列優化：

- **Phase 2 對齊：** ECC 高斯濾波核心從 5 → 7 px，壓制 8×8 DCT inter-block 邊界。
- **Phase 8.0 數據端預加重：** 在 Drizzle 疊加投影前先對原始 LR 影格套用殘差高通濾波器（`alpha=0.55`），還原 JPEG 被量化丟失的高頻細節。

- **Phase 9 反捲積：** 避開不穩定的空間噪訊-對比估計（JPEG 降噪會嚴重扭曲噪訊估算導致過度銳化），改為由用戶在執行時透過參數手動將 `psf_base` 定錨至鏡頭物理焦距：
  - **焦距 <= 28mm (17mm 超廣角，小人臉)** $\to$ `--psf-base 0.35`（保護小人臉五官不扭曲）。
  - **焦距 = 45mm** $\to$ `--psf-base 0.57`。
  - **焦距 = 50mm (50mm 中長焦，人臉大)** $\to$ `--psf-base 0.63`（發揮最高解像力與銳度）。

使用 `--jpeg` 或 `--raw` 可強制覆蓋自動偵測結果。

### Drizzle 核心選擇

`kernel_mode`（預設 **`lanczos4`**，2026-07 變更）決定 Phase 8 的累加核心。`lanczos4` 結合 Phase 9 自動估計 PSF sigma 的格柵安全上限，同時在振鈴、格柵週期性、leave-one-out 保真度三項指標上均有最佳表現。


---

## Pipeline Phase 說明

| Phase | 模組 | 描述 |
|---|---|---|
| 0 | `pipeline.py` | JPEG vs RAW 來源偵測 |
| 1 | `pipeline.py` | 載入連拍影像 |
| 2 | `alignment.py` | 粗略 ECC 次像素配准 |
| 3 | `alignment.py` | Harmony Anchor 參考幀選取 |
| 4 | `alignment.py` | 精細 ECC 配准 |
| 5 | `alignment.py` | N_eff 熵值抖動品質 |
| 6 | `masking.py` | 動態前景運動遮罩 |
| 7 | `preflight.py` | Pre-flight 安全倍率限制 |
| 8 | `drizzle.py` | Drizzle 疊加 + coverage-hole 填補 |
| 9 | `deconvolution.py` | 頻率相依 Wiener 反捲積 |
| 10 | `pipeline.py` | 最終輸出 |

---

## Drizzle 快取

Phase 2–8 為確定性計算，結果快取於 `~/.optico_cache`（以輸入檔案 SHA-256 + config 為 key）。相同連拍與設定的後續執行可直接跳至 Phase 9。使用 `--no-cache` 強制重新計算。

---

## 參數設定

所有參數集中於 `backend/constants.py` 的 `OpticoConfig` dataclass：

```python
OpticoConfig(
    target_scale=2.0,        # 放大倍率
    pixfrac=0.7,             # Drizzle 液滴大小
    jpeg_input=None,         # None = 自動偵測
    psf_override=None,       # 手動指定 PSF sigma
    skip_deconv=False,
)
```

---

## 參考文獻

- Fruchter & Hook (2002). *Drizzle: A Method for the Linear Reconstruction of Undersampled Images.* PASP 114.
- Wiener, N. (1949). *Extrapolation, Interpolation, and Smoothing of Stationary Time Series.*
