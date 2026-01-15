這是一份針對 eCAADe/CAADRIA 投稿標準優化的完整論文架構規劃。

這份大綱將重點放在**「從技術細節轉向設計賦能」**，並整合了您剛才對於 **"State-Continuous (狀態連續性)"** 與 **"Physics NCA (物理類神經元自動機)"** 的修正意見。

---

### **論文標題建議 (Working Titles)**

- **選項 A (強調互動性):** _Fluid Interaction: A State-Continuous m-NCA Framework for Real-Time Multi-Physics Design Exploration_
- **選項 B (強調解決痛點):** _Beyond Reset: Enabling Continuous Design-Evaluation Loops with Modular Neural Cellular Automata_

---

### **章節詳細規劃 (Detailed Outline)**

#### **1. Introduction (緒論)**

- **1.1 Context (背景):**
- 早期建築設計階段需要快速的環境性能反饋（Environmental Feedback）。
- 設計師需要的是「趨勢探索（Trend Discovery）」而非最終的工程驗證。

- **1.2 The Problem Gap (問題陳述 - 修正重點):**
- **The "Stop-and-Wait" Conflict:** 現有的 DL 代理模型（如 cGAN, CNN）雖然運算快，但本質上是「離散的快照預測（Discrete Snapshot Prediction）」。
- **The Cognitive Break:** 每次修改幾何都需要 "Computational Reset"（全場重算）。這種斷裂感切斷了設計師對於「因果關係（Causality）」的直覺理解，阻礙了設計思考的流暢度（Design Flow）。

- **1.3 Proposed Solution (本研究解方):**
- 提出基於 **Modular NCA (m-NCA)** 的 **State-Continuous** 模擬框架。
- 利用 NCA 的「時間遞歸（Recurrent）」與「局部更新（Local Update）」特性，實現模擬場的「動態癒合（Self-healing）」。

- **1.4 Contributions (貢獻):**
- 技術面：建立一個基於 Taichi 的多物理場耦合 NCA 引擎。
- 應用面：提出 "Physics Mixing Console" 的互動概念，允許即時調節物理權重。

#### **2. Related Works (文獻回顧)**

- **2.1 CFD & Optimization in Architecture:**
- 傳統數值模擬（CFD）精準但過於緩慢，僅適用於後期的驗證階段（Validation Phase）。

- **2.2 Deep Learning Surrogates (DL 代理模型):**
- 回顧 CNN/GAN 在建築物理預測的應用（引用 Guo et al., Westermann et al.）。
- **批判：** 這些模型多為 "Static Mapping"（輸入 A -> 輸出 B），缺乏時間維度的連續互動。

- **2.3 Cellular Automata in Architecture (建築中的細胞自動機 - 修正重點):**
- **Phase 1: Classical CA:** 早期用於城市生長、空間語法（Space Syntax），規則是人工定義的（Hand-crafted rules）。
- **Phase 2: Generative NCA:** 近年用於 3D 體素生成、結構生長（引用相關 ACADIA 論文）。
- **Phase 3: Physics-based NCA (Positioning):** 定位本研究於此最新領域（引用 Mordvintsev, Niklasson, 並提及 **Lu & Hou (2025)** 作為對照）。
- **差異化：** 區別於單一任務的 NCA，本研究強調 **"Modular" (模組化)** 與 **"Human-in-the-loop" (人機協作)** 的控制權。

#### **3. Methodology: The m-NCA Engine (系統架構)**

- _本章節重點在於解釋「系統如何運作」，使用圖表 (System Diagram) 輔助。_
- **3.1 m-NCA Architecture (模型架構):**
- 解釋 NCA 的基本原理：Perception (感知) -> Update Rule (神經網絡更新) -> Stochastic Update (隨機更新)。
- **Key Concept: Depth through Time:** 解釋如何用時間的迭代來換取空間的物理複雜度。

- **3.2 Modular Design (模組化設計):**
- 如何訓練不同的 NCA Cell 來分別模擬「平流（Advection）」、「擴散（Diffusion）」、「浮力（Buoyancy）」。
- 展示通道（Channels）的分配：例如 RGBA 通道分別代表 速度向量、溫度、障礙物標記。

- **3.3 Implementation Details (實作細節):**
- 使用 **Taichi Lang** 進行並行運算加速（這是達到 Real-time 60FPS 的關鍵）。
- 邊界條件（Boundary Conditions）的處理方式。

#### **4. Workflow: The Physics Mixing Console (互動工作流)**

- _本章節是 eCAADe 審稿人最看重的「工具應用」部分。_
- **4.1 The Metaphor (隱喻):**
- 將模擬環境比喻為一個 **"Mixing Console" (混音台)**。設計師像 DJ 一樣，不是在等待計算結果，而是在「操縱」物理場。

- **4.2 User Interaction Loop (使用者互動迴圈):**
- **Input:** 繪製障礙物、熱源、風口（幾何操作）。
- **Process (The Self-Healing):** 當幾何改變時，舊的流場不會消失，而是受到新幾何的干擾（Perturbation）並自動適應繞行。
- **Steering:** 使用者即時調整 Slider（例如增強風速、改變熱擴散係數），觀察場的變化。

#### **5. Experiments & Evaluation (實驗與評估)**

- **5.1 Performance Benchmark (效能測試):**
- **Metric:** FPS (Frame Per Second) 與 Grid Resolution 的關係圖。
- **Metric:** Convergence Speed (幾何變動後，流場恢復穩定的秒數)。
- 證明此工具在 Consumer-grade GPU 上可達到即時互動。

- **5.2 Design Scenarios (設計情境案例):**
- **Case 1: Indoor Ventilation (室內通風):** 展示移動隔間牆時，氣流死角（Dead Zone）的即時變化。
- **Case 2: Micro-climate (微氣候):** 展示建築量體配置對風廊的影響。
- _重點：_ 使用 **Time-series Screenshots (時序截圖)** 來展示 "Continuous" 的流動過程，而不僅僅是最後一張圖。

#### **6. Discussion (討論)**

- **6.1 Cognitive Continuity (認知連續性):**
- 回應 Introduction 的問題。討論「不重置」如何幫助設計師建立對物理現象的直覺（Intuition building）。

- **6.2 Limitations (限制):**
- **定性非定量：** 承認數值精確度不如 ANSYS，但對於早期設計決策已足夠。
- **解析度限制：** 目前受限於顯存的網格大小。

#### **7. Conclusion (結論)**

- 總結 m-NCA 框架如何彌補設計與模擬之間的時間斷裂。
- 展望未來：結合 3D 體素（Voxels）或更多物理模組（如聲學、光學）。

---

### **寫作小叮嚀 (Tips for Writing)**

1. **圖表優先 (Visuals First):**

- **Figure 1 (Teaser Image):** 必須是一張強烈對比圖。左邊是傳統流程（斷裂的箭頭），右邊是你的流程（連續的循環）。
- **架構圖:** 清楚標示 Input (User) -> Blackbox (m-NCA) -> Output (Screen)。

2. **關鍵字置入 (Keywords):**

- 在摘要和內文中多次提及：_Human-in-the-loop, Real-time Feedback, Early-stage Design, Surrogate Modeling, Design Cognition._

3. **避坑指南:**

- 不要過度強調你的 Loss Function 有多低（那是 CS 會議在乎的）。
- 要強調你的工具**「多好用」**、**「多快」**、以及**「如何改變設計師的決策」**。

這個架構已經非常完整，您可以直接依照這個骨架開始填充內容。如果有需要針對特定章節（例如 Methodology 的數學描述或是 Introduction 的開場白）進行英文擬稿，請隨時告訴我！
