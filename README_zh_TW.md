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
| `--output` / `-o` | `optico_output.png` | 輸出路徑 |
| `--scale` / `-s` | `2.0` | 目標放大倍率 |
| `--pixfrac` | `0.7` | Drizzle 像素分率（0–1）|
| `--chunks` | `8` | Drizzle 記憶體分塊數 |
| `--no-deconv` | 關 | 跳過 Wiener 反捲積 |
| `--align-scale` | 自動 | 覆蓋 ECC 縮放因子 |
| `--jpeg` | 自動 | 強制 JPEG 模式 |
| `--raw` | 自動 | 強制 RAW/PNG 模式 |
| `--no-cache` | 關 | 停用 Drizzle 快取 |
| `--cache-dir` | `~/.optico_cache` | 自訂快取目錄 |
| `--verbose` / `-v` | 關 | 開啟 debug 日誌 |

### JPEG vs RAW 處理差異

Optico 透過讀取檔案 header（JPEG SOI 標記 `0xFF 0xD8`）自動偵測輸入格式。JPEG 輸入時自動啟用三項調整：

- **Phase 2 對齊：** ECC 高斯濾波核心從 5 → 7 px，壓制 8×8 DCT inter-block 邊界假梯度，避免次像素偏移估計偏差。
- **Phase 8 Drizzle：** coverage-hole 填補對所有輸入通用。
- **Phase 9 反捲積：** PSF sigma ×1.10（複合光學 + JPEG 量化模糊），之後再套用格柵安全上限公式（見下方）；噪聲基準面掃描起點從 0.75 → 0.60 × Nyquist，避免 JPEG 頻率截止帶被誤判為白噪聲，高估噪聲基準導致高頻細節過度壓制。

使用 `--jpeg` 或 `--raw` 可強制覆蓋自動偵測結果。

### Drizzle 核心選擇

`kernel_mode`（預設 **`lanczos2`**）決定 Phase 8 的累加核心。真實連拍照片基準測試（`backend/benchmarks/kernel_bench.py`）發現舊預設 `box` 的 coverage-hole 格柵瑕疵在真實腳架連拍上確實嚴重存在；`lanczos2` 搭配 Phase 9 自動估計 PSF sigma 的格柵安全上限（避免 Wiener 濾波器放大同一個格柵頻率），同時在振鈴、格柵週期性、leave-one-out 保真度三項指標勝過 `box`——詳見 [CORE_ALGORITHM_zh_TW.md](CORE_ALGORITHM_zh_TW.md) 第 4–5 節與 `backend/benchmarks/reports/` 完整數據。`box`、`lanczos2_clamped`、`box_supersample` 仍可透過 `config.kernel_mode` 選用以供比較。

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
