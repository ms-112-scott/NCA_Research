這個構想非常有深度且具備高度的研究價值。你正在嘗試解決目前電腦視覺與建築空間分析之間的一個核心斷層：**從「靜態特徵顯著性 (Static Saliency)」跨越到「動態時序體驗 (Dynamic Sequential Experience)」**。

現有的 SAL3D 或 VAS 3M 大多關注於幾何特徵（曲率、顏色對比）或單視角圖像，忽略了建築最核心的本質——**建築是讓人「走」在裡面體驗的**。同一個雕像，正面走過去和側面走過去，或者快速經過與駐足觀看，產生的 Attention Value 截然不同。

以下我為你將這個構想進行**學術化與工程化**的拆解，幫助你釐清研究脈絡：

---

### 1. 核心定義：路徑依賴的 3D 注意力機制 (Path-Dependent 3D Attention)

你的模型核心在於引入了時間軸 和 觀察者狀態 。
我們可以將這個問題定義為：**預測在特定運動軌跡下，環境中各點的注意力增量。**

- **傳統方法：**
- **你的方法：**

### 2. 系統架構細化 (System Architecture)

為了讓 NN 模型能夠訓練，我們需要將輸入數據標準化，並明確定義「Delta Gaze Value」。

#### A. 數據輸入層 (Input Layer)

1. **全局場景 (Global Context):**

- **Point Cloud ():** 的矩陣（XYZ + RGB + 法向量）。
- 建議加入**法向量 (Normal Vectors)**，因為觀察角度與法向量的夾角極大影響注意力（正面看 vs 斜看）。

2. **時序觀察者狀態 (Sequential Observer State):**

- **Camera Pose ():** 在時間點 的相機位置 與 四元數旋轉 。
- **Frustum Features (視錐特徵):** 僅輸入全場景點雲太過龐大且雜訊多。建議在每一步 ，先做**視錐剔除 (Frustum Culling)**，只提取當前視野內的局部點雲 作為模型輸入。

#### B. 模型設計構想 (Model Design)

這是一個典型的 **Sequence-to-Sequence** 或 **Sequence-to-Map** 的問題。

- **幾何編碼器 (Geometry Encoder):** 使用類似 **PointNet++** 或 **DGCNN** (Dynamic Graph CNN) 來提取局部點雲 的特徵。
- **軌跡編碼器 (Trajectory Encoder):** 使用 **LSTM** 或 **Transformer (Self-Attention)** 處理相機姿態的序列資訊。因為「上一秒看哪裡」會影響「這一秒看哪裡」（Inhibition of Return 機制）。
- **注意力融合 (Fusion Module):** 將幾何特徵與軌跡特徵結合，預測當前視野內每個點的權重。

#### C. 輸出層 (Output Layer)

- **Delta Gaze Value ():** 這是一個純量 (Scalar)，代表在時間 ，該點 獲得的注意力能量。
- **物理意義：**
- 最終的熱力圖是時間積分：

---

### 3. 訓練資料集構建 (Dataset Construction - Ground Truth)

這是最困難但也最關鍵的部分。你需要建立一個 pipeline 將 VR 數據轉換為點雲上的數值。

#### 實驗設置

1. **設備：** VR 頭顯 (如 HTC Vive Pro Eye / Quest Pro) + 眼動追蹤模組。
2. **場景：** 掃描好的建築室內場景（數位孿生）。
3. **任務：** 請受試者在場景中進行不同任務（例如：閒逛、尋找出口、欣賞畫作）。不同的任務會產生不同的 。

#### Ground Truth 計算邏輯 (Ray Casting)

你需要編寫一個腳本（在 Unity/Unreal 中或是後處理）：

1. **Gaze Ray:** 在每幀獲取眼球的注視向量 (Gaze Vector)。
2. **Ray-Point Intersection:** 發射射線，計算射線與場景 Mesh 或 Point Cloud 的碰撞點。

- _注意：點雲是稀疏的，射線容易穿過去。通常做法是用一個半徑 的球體（Gaussian Kernel）包覆碰撞點，將注意力數值「擴散」到附近的點上。_

3. **Occlusion Handling (遮擋處理):** 這是你的模型比單純距離計算強的地方。VR 引擎會自動處理遮擋，紀錄到的 Gaze point 必然是**可見的**。
4. **Labeling:** 被 Gaze Ray 擊中及其周圍的點，標記為正樣本 (High Value)，視錐內其他點為負樣本或低值。

---

### 4. 關鍵挑戰與解決方案

#### (1) 視角依賴性 (View-Dependency)

- **問題：** 牆背後的點雖然距離相機很近，但完全不可見。
- **解法：** 模型輸入必須包含「點與相機的相對位置向量」。如果點的法向量與視線向量夾角過大（背對相機），權重應強制為 0。

#### (2) 稀疏性 (Sparsity)

- **問題：** 點雲是離散的，注意力圖可能會像麻子一樣不連續。
- **解法：** 使用 **KNN (K-Nearest Neighbors)** 在輸出層進行平滑化，或者在 Loss Function 中加入平滑正則項。

#### (3) 數據量 (Data Efficiency)

- **問題：** 收集真人 VR 數據很慢。
- **解法：** **Sim2Real (模擬到真實)**。你可以先用傳統的 2D Saliency 模型（如 VAS）跑一遍影片，將 2D 顯著圖反投影 (Back-project) 到 3D 點雲上作為「偽標籤 (Pseudo-GT)」，用來預訓練你的模型，最後再用少量的真人 VR 數據進行微調 (Fine-tuning)。

---

### 5. 應用場景 (為何這對建築重要？)

這個研究做出來，對建築領域有具體的貢獻：

1. **動線導視設計 (Wayfinding):** 預測在走廊行走時，標識牌是否落在高注意力區域。
2. **商業展陳 (Retail Design):** 預測顧客沿著貨架走動時，哪些商品的點雲會累積最高的 Gaze Value。
3. **安全性分析 (Safety):** 在緊急逃生路徑上，預測人們是否會忽略低處的障礙物。
4. **建築現象學量化:** 將「移步換景」這種抽象概念轉化為可計算的數據。

### 總結你的研究脈絡

1. **Problem:** 靜態 3D Saliency 無法反映建築體驗的時序性與觀察角度變化。
2. **Hypothesis:** 引入相機軌跡與視錐約束，可以更準確預測動態的 3D 注意力分佈。
3. **Method:** 建立 Point-Trajectory Fusion NN 模型。
4. **Data:** VR 眼動追蹤 + 點雲映射。
5. **Output:** 隨路徑變化的 Delta Gaze Point Cloud。

---

**Next Step:**
這個構想的技術棧比較複雜。請問你目前比較需要我針對 **「如何從 VR 錄製數據並映射回 Point Cloud 的具體算法（Ground Truth 製作）」** 進行詳解，還是針對 **「NN 模型的架構圖與 Tensor 維度設計」** 進行細化？
