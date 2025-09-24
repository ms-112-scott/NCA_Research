# E4-4.2 模型後續優化與實驗方向建議

觀察 Loss 曲線，可以看到 Loss 雖然穩定下降，但在後期（約 500 epochs 後）下降趨於平緩，且仍有一定程度的震盪。這說明模型的能力尚未完全發揮，我們可以從以下幾個方向進行調整，目標是**進一步降低 Loss**、**加速收斂**並**提升泛化能力**。

---

## 1. Loss 函數與權重優化

Loss 函數是引導模型學習方向的關鍵，目前的設計雖然有效，但可以更精細化。

### 1.1. 調整 Loss 權重策略

目前的 `custom_loss` 中，您透過除以一個小常數（如 `1e-2`）來放大特定 loss 的影響力。

```python
# Current
def custom_loss(x: torch.Tensor, y: torch.Tensor) -> dict:
    return {
        "mse_loss": W_MSE * data_mse_loss(x, y)/1e-2,
        "obstacle_loss": W_OBSTACLE * obstacle_loss(x)/1e-4,
        "Uvel_loss": W_UVEL * Uvel_loss(x)/1e-2,
    }
```

**建議：**

- **移除硬編碼的縮放因子**：直接調整 `W_MSE`, `W_OBSTACLE`, `W_UVEL` 的值會更直觀且易於管理。例如，將 `W_MSE` 設為 `1.0`，`W_OBSTACLE` 設為 `300.0`，`W_UVEL` 設為 `20.0`，這樣更容易比較不同 loss 的相對重要性。
- **動態權重調整 (Loss Annealing)**：在訓練初期，模型需要先學會生成基本的數據分佈，因此 `mse_loss` 應該佔主導。隨著訓練進行，再逐步提高物理約束 loss 的權重。

  **範例策略**：

  ```python
  def get_physics_weight(epoch, total_epochs, max_weight=1.0, start_epoch=500):
      if epoch < start_epoch:
          return 0.0
      # 從 start_epoch 開始，權重線性增長到 max_weight
      return max_weight * min(1.0, (epoch - start_epoch) / (total_epochs - start_epoch))

  # 在 training loop 中
  w_physics = get_physics_weight(epoch, TOTAL_EPOCHS, max_weight=W_OBSTACLE)
  loss_dict["obstacle_loss"] = w_physics * obstacle_loss(x)
  ```

### 1.2. 引入新的物理約束：散度損失 (Divergence Loss)

您的程式碼中已經定義了 `divergence_loss`，但並未在 `custom_loss` 中使用。對於不可壓縮流體，速度場的散度應為零 (`∇ · u = 0`)，這是一個非常強的物理先驗知識。

**建議：**
將 `divergence_loss` 加入到總 loss 中。這會迫使模型學習到物理上更合理的流場，有助於提升泛化能力。

```python
# 1. 在 global_params 中新增權重
W_DIV = 1.0 # 可調整

# 2. 更新 custom_loss
def custom_loss(x: torch.Tensor, y: torch.Tensor) -> dict:
    u, v = x[:, 4:5, ...], x[:, 5:6, ...]
    return {
        "mse_loss": W_MSE * data_mse_loss(x, y),
        "obstacle_loss": W_OBSTACLE * obstacle_loss(x),
        "Uvel_loss": W_UVEL * Uvel_loss(x),
        "div_loss": W_DIV * divergence_loss(u, v) # 新增項目
    }
```

---

## 2. 模型結構與更新規則

### 2.1. 增加模型容量

目前的 `CAModel` 結構相對簡單（`num_hidden_layers=1`）。如果 Loss 持續無法下降，可能是模型容量不足以捕捉複雜的物理現象。

**建議：**

- **加深網絡**：嘗試增加 `num_hidden_layers` 到 2 或 3。
  ```python
  # in CAModel.build_rule_block
  for _ in range(num_hidden_layers):
      layers.append(nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=1))
      layers.append(nn.Tanh())
      layers.append(nn.Conv2d(hidden_dim*2, hidden_dim, kernel_size=1))
      layers.append(nn.Tanh())
  ```
- **加寬網絡**：增加 `hidden_dim`，例如从 `128` 增加到 `192` 或 `256`。

### 2.2. 嘗試不同的激活函數

`Tanh` 激活函數在值域兩端會饱和，可能導致梯度消失。

**建議：**
可以嘗試替換為 `GeLU` 或 `SiLU` (Swish)，這些是現代神經網絡中更常用的激活函數，有助於改善梯度流。

```python
# layers = [nn.Conv2d(in_channels, hidden_dim, kernel_size=1), nn.Tanh()]
layers = [nn.Conv2d(in_channels, hidden_dim, kernel_size=1), nn.GELU()]
```

### 2.3. 引入隨機性更新

目前的細胞狀態更新是確定性的。一些研究表明，引入隨機性可以幫助模型探索更廣闊的解空間，避免陷入局部最優。

**建議：**
在 `forward_pass` 中，讓一小部分的細胞隨機"死亡"（不進行更新）。

```python
def forward_pass(self, x: torch.Tensor) -> torch.Tensor:
    # ...
    dx = self.rule_block(y)

    # Stochastic update: 隨機丢棄 5% 的更新
    stochastic_mask = (torch.rand(x[:, :1, :, :].shape) > 0.05).float().to(DEVICE)

    # 結合 alive mask 和 stochastic mask
    update_mask = x[:, 0:1, :, :] * stochastic_mask

    updated = x + dx * update_mask
    # ...
```

---

## 3. 訓練策略與泛化能力

### 3.1. 優化器與學習率策略

- **L2 正則化 (Weight Decay)**：為了防止過擬合，可以在 `Adam` 優化器中加入 `weight_decay`。這相當於對模型權重進行 L2 懲罰，有助於提升泛化性。
  ```python
  optimizer = Optimizer.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
  ```
- **更平滑的學習率排程**：`StepLR` 會造成學習率的階梯式下降。可以改用 `CosineAnnealingLR`，它能讓學習率更平滑地下降，有助於模型在訓練後期更好地收斂到最優解。
  ```python
  # from torch.optim.lr_scheduler import CosineAnnealingLR
  lr_sched = CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-6)
  ```

### 3.2. 數據增強 (Data Augmentation)

目前的 `create_epoch_pool` 只使用了隨機裁剪。為了讓模型學習到旋轉、翻轉不變的物理特徵，可以加入更多數據增強。

**建議：**
在 `create_epoch_pool` 中，對裁剪出的 `sub` 張量進行隨機的：

- **90 度整數倍旋轉** (`torch.rot90`)
- **水平或垂直翻轉** (`torch.flip`)

這會極大地豐富訓練數據，有效提升模型的泛化能力。

---

## 4. 實驗與評估

- **更豐富的評估指標**：除了 L2 Error，可以在 `evaluate_one_epoch` 中加入對物理指標的監控，例如**平均散度**。觀察模型在驗證集上的物理一致性。
- **視覺化分析**：除了觀看最終的模擬影片，也可以生成 **誤差圖 (Y_batch - X_pred)** 的影片，觀察模型在哪些區域、哪些物理量上誤差較大，從而獲得下一步優化的靈感。

---

祝您實驗順利！
