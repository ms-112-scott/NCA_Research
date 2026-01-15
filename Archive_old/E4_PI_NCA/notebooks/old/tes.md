## 🧩 主程式流程

- **Main (Batch Run Loop, 共 50 次)**

  - `resolve_list_options(CONFIG, OPTIONS_PATHS)`
  - `model = CAModel(cfg)`

    - `build_rule_block()`
    - `forward(n_times)`

      - `forward_pass()`

        - `perception()`

          - `perchannel_conv()`

        - `rule_block (Sequential Conv2d + Tanh)`

  - `optimizer = Optimizer.Adam(model.parameters(), lr)`
  - `lr_sched = torch.optim.lr_scheduler.StepLR()`
  - `model_path = get_output_path()`
  - `run_training(cfg, model, optimizer, loss_fn=pinn_loss, metric_fn=metric_fn, lr_sched, output_path)`

    - （主訓練流程，見下）

---

## 🧠 `run_training()` 主訓練流程

- **初始化**

  - `npz_dict = np.load(config["dataset"]["dataset_npz_path"])`
  - `init_dataset_and_loader(config, npz_dict)`

    - `NCA_Dataset()`
    - `random_split(dataset)`
    - `DataLoader(train/val/test)`

  - `EarlyStopper(config)`

- **Epoch Loop**

  - `for epoch in trange(total_epochs):`

    - **訓練階段**

      - `train_one_epoch(config, epoch, model, optimizer, loss_fn, train_dataset, train_loader)`

        - `reset_nth_hidden_channels()`
        - `get_rollout_times()`
        - `model.forward()`
          ↳ `forward_pass()` → `perception()` → `perchannel_conv()`
        - `train_dataset.dataset.update_x_pool()`
        - `loss_fn(config, x_pred, y_batch, x_batch_reset)`
        - `optimizer.zero_grad()`
        - `total_loss.backward()`
        - `optimizer.step()`

    - **驗證階段**

      - `evaluate_one_epoch(config, epoch, model, loss_fn, val_dataset, val_loader, metric_fn)`

        - `model.forward()`
          ↳ `forward_pass()` → `perception()` → `perchannel_conv()`
        - `loss_fn(config, x_pred, y_batch, x_batch)`
        - `metric_fn(x_pred, y_batch)`

    - **視覺化與紀錄**

      - `viz_loss(train_loss_log, eval_loss_log)`
      - `print_loss_dict(train_loss_dict, eval_loss_dict)`
      - `viz_batch_channels(train_batch_dict)`
      - `viz_pool(train_dataset.dataset.x_pool)`
      - `viz_pool(train_dataset.dataset.y_pool)`

    - **模型儲存**

      - `if (epoch + 1) % save_interval == 0:`
        → `save_checkpoint(model, optimizer, epoch, path)`

    - **Early Stopping**

      - `early_stopper.step(train_loss)`
      - `check_tensor_nan_inf()`
      - `gc.collect()`
      - `torch.cuda.empty_cache()`

- **訓練結束後**

  - `viz_loss(..., save_path)`
  - `save_checkpoint(model, optimizer, total_epochs, "model_Final.pth")`

---

## 🧬 模型定義：`CAModel`

- `__init__()`

  - 建立參數：channels, hidden_dim, kernel_count, num_hidden_layers
  - 呼叫 `build_rule_block()`

- `build_rule_block(in_channels, hidden_dim, out_channels, num_hidden_layers)`

  - 建立多層 `Conv2d + Tanh` 區塊

- `perchannel_conv(x, filters)`

  - 對每個 channel 執行 depthwise convolution

- `perception(x)`

  - 建立感知濾波器：

    - identity
    - sobel_x, sobel_y
    - laplacian
    - LBM kernel

  - 呼叫 `perchannel_conv(x, filters)`

- `forward_pass(x)`

  - 呼叫 `perception(x)`
  - 通過 `rule_block`
  - 更新狀態 `x + dx * mask`

- `forward(x, n_times)`

  - 重複多次呼叫 `forward_pass()`

---

## 🧠 訓練與驗證輔助

- `reset_nth_hidden_channels(x, init_batch_count, channel_start)`

  - 清空部分 hidden channels

- `train_one_epoch(...)`

  - 單 epoch 訓練循環
  - 含資料池更新與梯度反傳

- `evaluate_one_epoch(...)`

  - 單 epoch 驗證流程
  - 不進行梯度更新

- `EarlyStopper`

  - 屬性：`patience`, `min_delta`, `counter`, `best_loss`
  - 方法：`step(loss)` 判斷是否早停

- `save_checkpoint(model, optimizer, epoch, path)`

  - 儲存模型與 optimizer 狀態

---

## 📈 Metrics

- `metric_fn(pred, target)`

  - 計算：

    - L1 誤差
    - L2 誤差
    - 相對誤差 (relative error)

  - 僅針對風場與湍流通道計算

---

## 🖼️ 視覺化後處理

- `show_all_png(root_dir)`

  - 遍歷資料夾並顯示所有 PNG 結果圖
  - 使用 `matplotlib` 顯示

---

## 🎬 模型測試（影片輸出）

- 建立 `output_dir`
- `Y_batch = create_epoch_pool(mode="eval")`
- `X_batch = init_X(Y_batch)`
- `load_model = CAModel(...)`
- `load_model.load_state_dict(...)`
- 逐步迭代 `rollout_steps = 50`

  - 每步輸出一張 PNG
  - 最後合成 `output.mp4`
