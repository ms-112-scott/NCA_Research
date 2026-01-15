## CAADRIA 論文架構（10 頁）

### 1. Introduction （1–1.5 頁）

**目的：** 問題脈絡與研究動機

- 建築模擬中 CFD 的運算代價與資料難以泛化的問題
- 近期 AI-based 模擬（如 NCA、PINN）的潛力與限制
- 前期研究成果簡述：自製 CFD 資料集 + NCA 模型
- 本研究動機：

  - 使用**公開 CFD dataset** 檢驗泛化能力
  - 引入**物理知識約束（Physics-Informed constraints）** 提升可解釋性
  - 構建**跨解析度學習框架（coarse → mid → fine grid）**

- 研究問題：

  1. 如何以 NCA 框架表達建築流場的多尺度結構？
  2. 如何在不依賴特定模擬公式下仍保留物理合理性？
  3. 對建築設計流程有何潛在影響？

---

### 2. Related Works （1–1.5 頁）

**目的：** 建立研究定位與對比背景

- **AI-based CFD approaches：** surrogate modeling, data-driven turbulence, PINN
- **Neural Cellular Automata：** 生成式自組織模擬（Growing NCA, Fluid NCA）
- **Building performance & digital design：** 以 NCA/AI 融合數位模擬的潛能
- 差異化定位：

  - 現有研究多重視「模擬準確度」
  - 本研究轉向探討「**資料泛化性 + 建築應用解釋性**」

---

### 3. Methodology （2–2.5 頁）

**目的：** 描述研究實作的核心方法與實驗流程

#### 3.1 Dataset and Multi-Resolution Grids

- 使用公開建築風場 CFD dataset（描述內容與案例）
- 定義三層解析度（coarse / mid / fine）與其對應的學習階段
- 保持統一的 loss 與 data pipeline

#### 3.2 Model Architecture

- 基於 PINCA 框架的 Neural Cellular Automata
- 模型結構簡述：

  - 狀態通道組成（velocity, pressure, divergence, etc.）
  - 區域更新規則（local convolutional rules）
  - 搭配 coarse→fine grid refinement

#### 3.3 Physics-informed Constraints

- 基本流體守恆規則（散度、動量）以 soft constraint 方式引入
- 以「可解釋性」而非「嚴格公式」為重點
- 對比 purely data-driven NCA 模型的限制

---

### 4. Experiments and Results （2–2.5 頁）

**目的：** 呈現主要發現與比較分析

#### 4.1 Training and Evaluation Protocol

- 統一 pipeline、loss 組成、訓練階段
- 泛化測試：不同幾何或開口配置的風場

#### 4.2 Result Visualization

- 以圖像對比方式呈現流場結果（非數學表格）
- 可視化散度與流線結構的穩定性
- Coarse→Fine 的局部細化效果

#### 4.3 Discussion of Generalization and Interpretability

- 模型如何在不同場景維持流場拓撲
- 各種物理約束對模型穩定性與收斂的影響
- 對比 baseline data-driven 模型

---

### 5. Architectural Implications （1.5 頁）

**目的：** 將技術成果轉化為建築理論／應用討論

- PINCA 作為一種「**行為生成模型**」的潛力
- 在早期設計階段提供即時流場預估
- 對多模態設計（熱、風、行為模擬）的延伸可能
- 對建築師而言的價值：

  - 從 CFD 黑盒 → 可互動、可學習的 field automata
  - 強化數位建築在「生成性模擬」層面的討論

---

### 6. Conclusion and Future Work （0.5–1 頁）

**目的：** 研究貢獻與提出未來方向

- 主要貢獻：

  1. 建立一個具物理啟發的 NCA 模型框架
  2. 驗證公開 CFD 資料集上的泛化性
  3. 提出多解析度的學習與生成策略

- 未來方向：

  - 與 real-time generative design system 整合
  - 探索風-熱-行為耦合的自組織模擬
  - 結合 GNCA 作為自適應生成引擎

---

### 7. Reference

---

## 論文重點策略

| 面向     | 寫作語氣                                   | 強調重點                         |
| -------- | ------------------------------------------ | -------------------------------- |
| 建築導向 | “design-relevant simulation paradigm”      | 聚焦建築師如何重新思考模擬過程   |
| 技術導向 | “physics-informed yet data-flexible model” | 不追求公式推導，而強調整合概念   |
| 理論連結 | “from simulation to generative behavior”   | 將流場模擬視為一種行為生成的基礎 |

---

---

---

# TODO

1. 資料處理

   - 資料物理意義 建立 pipeline
   - 物理空間 -> 數值空間
     - 無因次化 -> normalize

2. 模型基礎測試

   - patch 測試
   - pinn loss 物理約束
   - rowout 測試
   - PI-NCA 模型架構 (gnca-style, unet-style, VAE-style, GAN-style)

3. 模型標準
   - RMSE
   - SSIM
