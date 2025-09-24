# GEMINI.md

## Project Overview

This project explores the use of Neural Cellular Automata (NCA) for simulating fluid dynamics. It appears to be a research project with a series of experiments, each building upon the previous one. The project uses TensorFlow and Keras for building and training the NCA models. The experiments are conducted in Jupyter Notebooks, which allows for interactive development and visualization of the results.

The project also involves pre-processing of Computational Fluid Dynamics (CFD) data using ParaView. A Python script is provided to automate this process, converting `.foam` files into NumPy arrays that can be used for training the NCA models.

## Building and Running

The project is primarily run through Jupyter Notebooks. To run the experiments, you will need a Python environment with the following dependencies installed:

- TensorFlow
- Keras
- NumPy
- Matplotlib
- OpenCV
- MoviePy

You will also need to have ParaView installed to run the data pre-processing script.

### Running the Experiments

1.  **Pre-process the CFD data:**

    - Make sure you have your CFD case files (in `.foam` format) in the `00_RAW_FloorPLan_CFD_CASE` directory.
    - Run the `ParaviewCasePreprocess.py` script to convert the `.foam` files into NumPy arrays. The script will save the output in the `dataset/FP_case_tensor` directory.
    - `python ParaviewCasePreprocess.py`

2.  **Run the Jupyter Notebooks:**
    - Navigate to the directory of the experiment you want to run (e.g., `E1_basicGNCA/notebooks`).
    - Open the Jupyter Notebook file (e.g., `E1_2-baseline_NCA.ipynb`).
    - Run the cells in the notebook to train the NCA model and visualize the results.

## Development Conventions

- The project is written in Python.
- The code is organized into modules, with shared utilities in the `core_utils` directory.
- Each experiment is self-contained in its own directory.
- The use of Jupyter Notebooks with clear headings and comments is encouraged for documenting the research process.
- The project uses a consistent naming convention for files and directories.

把 E4*PI_NCA 資料夾中所有.ipynb 的 notebook 依照開發的實驗過程順序重新命名為，並且取名為簡易可以讀懂有意義的敘述，檔案用英文重新命名。E4-{number}*{description}

1.先讀取 GEMINI_E4.md 理解所有 E4_PI_NCA 在做甚麼。
2.E4_PI_NCA 資料夾中所有.ipynb 的 notebooks 中，第一最開頭地方新增一個 md cell 記錄這的筆記本相較上一個 或是 主要的改進部分。
/clear

# for E4_PI_NCA/

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
