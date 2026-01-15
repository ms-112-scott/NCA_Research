# HDF5 Dataset 架構 (Wind-field GNCA)

## 根目錄 `/`

- `GlobalMetaData.h5`

  - 描述整個 dataset 的 meta 資訊
  - **資料內容**

    - `num_cities` (int): 城市數量
    - `city_list` (list of str): 城市名稱列表
    - `description` (str): dataset 描述

- `<city_name>.h5` (每個城市一個檔案)

  - 範例：`CN-BE-V1.h5`

---

## 城市檔案 `<city_name>.h5` 根結構

### 1️⃣ `city_base` (Dataset)

- **資料型態**：`float16`
- **shape**：`(C, H, W)`
- **說明**：城市固定通道，用於初始化或 pool 重置
- **包含通道**：

  1. `coord_y` (C 座標，範圍 [-1,1])
  2. `coord_x` (H 座標，範圍 [-1,1])
  3. `geo_mask` (地理遮罩，nan -> 0)
  4. `topo` (地形高度)

---

### 2️⃣ `cases` (Group)

- 每個 case 一個子 Group

- 範例結構：

  ```
  cases/
    ├─ CN-BE-V1_d0/
    │    ├─ wind_field (Dataset, float16, shape=(C,H,W))
    │    └─ attrs: meta (str) {"wind_dir_deg":0,"wind_dir_xy":[0,1],"channels":[…]}
    ├─ CN-BE-V1_d30/
    │    ├─ wind_field
    │    └─ attrs: meta
    └─ …
  ```

- **wind_field Dataset**

  - **資料型態**：`float16`
  - **shape**：`(C, H, W)`
  - **通道內容**：

    - 除了 city_base 包含的固定通道 (`coord_y`, `coord_x`, `geo_mask`, `topo`) 外，剩下的為：

      - `windInitX` (初始風場 X 分量)
      - `windInitY` (初始風場 Y 分量)
      - ped channels (例如行人密度等)

- **meta attr (str, JSON-like)**

  - `wind_dir_deg` (float): 風向角度（北為 0°，逆時針）
  - `wind_dir_xy` (list[float]): 風向 X, Y 分量
  - `channels` (list[str]): `wind_field` 中除了固定 city_base 通道的其他通道名稱

---

### 3️⃣ 範例 Dataset 結構總覽 (Markdown)

```markdown
dataset_h5/
├─ GlobalMetaData.h5
│ ├─ num_cities: 5
│ ├─ city_list: ["CN-BE-V1", "CN-SH-V2", ...]
│ └─ description: "HDF5 dataset for wind-field GNCA experiments"
├─ CN-BE-V1.h5
│ ├─ city_base (Dataset, float16, CxHxW, channels=[coord_y, coord_x, geo_mask, topo])
│ └─ cases (Group)
│ ├─ CN-BE-V1_d0
│ │ ├─ wind_field (Dataset, float16, CxHxW, channels=[windInitX, windInitY, ped data…])
│ │ └─ meta (attrs, str) {"wind_dir_deg":0,"wind_dir_xy":[0,1], "channels":[…]}
│ ├─ CN-BE-V1_d30
│ └─ …
├─ CN-SH-V2.h5
│ ├─ city_base
│ └─ cases
│ ├─ CN-SH-V2_d0
│ └─ …
└─ …
```

---

### ✅ 備註

1. **Channel 一致性**

   - 每個城市的 `city_base` 只包含固定通道 (4 個)
   - 每個 case 的 `wind_field` 則包含 **city_base 之外的動態通道**

2. **Pool 設計依據**

   - 每個 case 可以視為一個 pool 的初始狀態
   - Pool 內保存 `input` tensor，訓練後回寫，更新動態狀態

3. **資料存取**

   - `MultiCityWindDataset` 會在初始化時讀取整個 HDF5，並為每個 sample 建立 pool
   - `__getitem__` 從 pool 中抽樣更新，確保每個 sample 狀態可動態演化
