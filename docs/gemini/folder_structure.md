# 專案資料夾結構 (`folder_structure.md`)

本文件旨在說明 `NCA_CFD` 專案的資料夾結構及其檔案分類方式，以利專案成員理解程式碼組織邏輯。

```
NCA_CFD/
├── .gitignore             # Git 版本控制忽略檔案設定
├── GEMINI.md              # 專案總覽與使用說明
├── requirements.txt       # Python 專案相依套件列表
├── setup.py               # Python 套件安裝設定
├── data/                  # 資料存放區
│   ├── raw/               # 原始資料，例如 LBM 模擬的未經處理結果
│   │   └── (LBM 模擬的原始結果，例如速度場、壓力場等)
│   └── processed/         # 處理後的資料，例如 NCA 訓練專用的資料集
│       └── (經過處理後，適合 NCA 訓練的資料集)
├── docs/                  # 專案文件
│   └── (專案相關的說明文件、理論背景、設計規範等)
├── notebooks/             # Jupyter Notebook 檔案
│   └── (用於資料分析、結果視覺化、模型探索與實驗的互動式筆記本)
├── src/                   # 專案原始碼
│   ├── nca/               # 神經細胞自動機 (NCA) 相關模組
│   │   ├── __init__.py    # 將此目錄標記為 Python 套件
│   │   ├── model.py       # NCA 模型定義，包含 BaseTextureNCA 類別等
│   │   ├── trainer.py     # NCA 訓練邏輯，包含 NCATrainer 類別等
│   │   ├── loss.py        # NCA 訓練中使用的損失函數，例如 VGGLoss, MSELoss 等
│   │   └── ...            # 其他 NCA 相關程式碼
│   ├── lbm/               # Lattice Boltzmann Method (LBM) 相關模組
│   │   ├── __init__.py    # 將此目錄標記為 Python 套件
│   │   ├── solver.py      # LBM 核心求解器實作
│   │   ├── boundary.py    # 邊界條件處理函數或類別
│   │   └── utils.py       # LBM 相關的輔助工具函數
│   │   └── ...            # 其他 LBM 相關程式碼
│   └── utils/             # 共用工具模組
│       ├── __init__.py    # 將此目錄標記為 Python 套件
│       ├── EarlyStopping.py # 早停機制實作
│       ├── OmniConfig.py  # 配置管理工具
│       ├── utils.py       # 其他通用的輔助函數，如圖片處理、隨機種子設定等
│       └── visualization.py # 共用的視覺化工具函數
├── scripts/               # 可執行腳本
│   ├── run_lbm.py         # 執行 LBM 模擬的命令列腳本
│   ├── train_nca.py       # 執行 NCA 模型訓練的命令列腳本
│   └── evaluate_nca.py    # 評估 NCA 模型效能的命令列腳本
└── tests/                 # 測試檔案
    ├── nca/               # NCA 模組的單元測試或整合測試
    │   └── ...
    └── lbm/               # LBM 模組的單元測試或整合測試
        └── ...
```

### 檔案歸類原則：

*   **`src/` 核心邏輯:** 存放所有與 NCA 和 LBM 演算法實作相關的 Python 原始碼。為了保持模組化和高內聚低耦合，NCA 和 LBM 各自擁有一個子資料夾。
*   **`data/` 資料管理:** 將原始資料 (`raw`) 和為訓練準備的資料 (`processed`) 分開，明確資料處理的階段。
*   **`scripts/` 自動化執行:** 提供獨立的 Python 腳本，作為執行專案主要流程（如模擬、訓練、評估）的命令列介面，方便自動化和重複執行。
*   **`notebooks/` 探索與實驗:** 專為互動式分析、快速原型開發和結果展示而設計，不應包含生產環境的核心程式碼。
*   **`docs/` 文件記錄:** 存放專案的設計文檔、原理說明、API 文件等，保持專案知識的完整性。
*   **`tests/` 品質保證:** 包含各模組的測試程式碼，確保程式碼的正確性和穩定性。

此結構旨在提高專案的可讀性、可維護性和可擴展性，讓您在開發 LBM 模組時能更有效率地與現有的 NCA 模組協同工作。
