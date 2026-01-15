# GEMINI_E4.md (重構版)

## 專案總覽

本專案 (E4_PI_NCA) 的核心目標是**開發一種用於流體動力學模擬的物理資訊神經細胞自動機 (Physics-Informed Neural Cellular Automata, PI-NCA)**。靈感啟發自可成長神經細胞自動機 (Growing NCA)，我們試圖將物理定律（以偏微分方程的形式）作為一種歸納偏置 (inductive bias) 嵌入到 NCA 的學習規則中，期望模型不僅能學習「如何演化」，更能理解「為何如此演化」。

專案採用 PyTorch 作為主要開發框架，並以 OpenFOAM 模擬的流場資料作為初始的真值 (Ground Truth)。

---

## 研究脈絡與檔案結構

為了清晰地反映研究的演進過程，`E4_PI_NCA/notebooks` 目錄下的檔案已按照以下邏輯重新編號與命名。整個研究路徑分為主要路徑、備用路徑與平行研究三部分。

### 1. 核心研究路徑 (CFD-PI-NCA)

這是專案的主線，目標是實現最終的 PI-NCA 模型。

- **第一階段：資料處理 (`E4-1.x`)**

  - **簡述**: 處理作為真值的 OpenFOAM 模擬資料。
  - **重點**:
    - `E4-1.0_Data_Visualization.ipynb`: 視覺化檢查，確保資料品質。
    - `E4-1.1_Data_Preprocessing.ipynb`: 將資料轉換為模型可用的格式。

- **第二階段：基準模型建立 (`E4-2.x`)**

  - **簡述**: 將 GNCA 思想應用於 CFD 資料，建立一個純資料驅動的 NCA 模型作為效能基準。
  - **重點**:
    - `E4-2.0_CFD_GrowthNCA.ipynb`: 實現並訓練一個僅基於 MSE 損失的 CFD-NCA。

- **第三階段：物理資訊整合 (`E4-3.x`)**
  - **簡述**: **本專案的核心貢獻**。將 PINN 的思想整合進 NCA，使模型學習物理定律。
  - **重點**:
    - `E4-3.0_PINN_Concept_Test.ipynb`: 獨立驗證使用物理殘差作為損失函數的可行性。
    - `E4-3.1_CFD_PhysicsInformed_NCA.ipynb`: **最終模型**，將資料損失與物理損失結合，進行端到端訓練。

### 2. 備用資料集路徑

在研究過程中，由於發現原始 CFD 資料存在一些潛在問題，我們轉向公開的 UrbanTales 資料集來驗證模型的穩健性。

- **第四階段：UrbanTales 資料集實驗 (`E4-4.x`)**
  - **簡述**: 在一個新的、更標準化的資料集上重複 NCA 實驗。
  - **重點**:
    - `E4-4.0_..._UrbanTales.ipynb`: 資料預處理。
    - `E4-4.1_..._UrbanTales_GrowthNCA.ipynb`: 在 UrbanTales 上訓練 NCA 模型。

### 3. 平行對照研究

為了從非機器學習的視角進行比較，我們獨立進行了 LBM 的研究。

- **第五階段：格子波茲曼方法 (`E4-5.x`)**
  - **簡述**: 使用傳統的計算流體力學方法 LBM 進行模擬。
  - **重點**:
    - `E4-5.0_LBM_Simulation.ipynb`: 實現一個基礎的 LBM 模擬器，其結果可作為 NCA 模型的對照組，或用於未來生成更高品質的訓練資料。

---

## 如何建置與執行

要運行本專案，您需要一個已安裝以下函式庫的 Python 環境:

- PyTorch
- NumPy
- Matplotlib
- tqdm
- ipynbname

### 執行核心實驗

本專案最核心的實驗是物理資訊 NCA 的實現，您可以直接運行相關的 Jupyter Notebook：

1.  開啟 Jupyter Notebook 或 JupyterLab。
2.  導航至 `E4_PI_NCA/notebooks/` 目錄。
3.  打開 `E4-3.1_CFD_PhysicsInformed_NCA.ipynb`。
4.  依照筆記本中的儲存格順序執行，即可開始模型的訓練與評估。
