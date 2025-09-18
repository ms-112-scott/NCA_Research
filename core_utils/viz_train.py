import os
from datetime import datetime
from typing import Any, Optional, Sequence, Union, Dict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from core_utils.utils_image import imshow, tile2d
from core_utils.utils_io import imwrite
from E4_PI_NCA.utils.helper import channels_to_rgb, to_HWC


# ===============================================================
# region viz_pool
def viz_pool(pool: torch.Tensor, step_i: int, output_path: str = "train_log") -> None:
    """
    將 pool (BHWC tensor) 中前 49 張圖像拼貼，並加上四邊漸變淡出效果，
    最後輸出圖片存檔 (train_log/%04d_pool.jpg)。
    """
    count = int(np.floor(np.sqrt(pool.shape[0])))
    count = np.min([count, 7])
    subpool = torch.cat([pool[: np.square(count) - count], pool[-count:]])
    tiled_pool: np.ndarray = tile2d(subpool, n=count)  # (H, W, 3)

    # imwrite(f"{output_path}/{step_i:04d}_pool.jpg", (tiled_pool * 255).astype(np.uint8))
    imshow(tiled_pool)
    plt.imshow(tiled_pool, cmap="jet")


# ===============================================================
# region viz_batch
def viz_batch(batch_list, step_i, output_path="train_log"):
    """
    將訓練批次前後狀態左右合併、上下堆疊後輸出並顯示
    """
    lists = []
    for i in range(len(batch_list)):
        lists.append(np.hstack(batch_list[i][..., :3]))
    vis = np.vstack(lists)
    imwrite(f"{output_path}/batches_{step_i:04d}.jpg", vis)
    print("batch (before/after):")
    imshow(vis)


# ===============================================================
# region moving_average
def moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    """
    計算移動平均 (Moving Average)

    參數
    ----------
    arr : np.ndarray
        輸入的一維數列
    window : int
        平滑視窗大小

    回傳
    ----------
    np.ndarray
        經過移動平均處理後的數列
    """
    return np.convolve(arr, np.ones(window) / window, mode="valid")


# ===============================================================
# region viz_loss
def viz_loss(
    train_loss_log: Union[Sequence[float], np.ndarray],
    eval_loss_log: Optional[Union[Sequence[float], np.ndarray]] = None,
    log_scale: bool = True,
    window: int = 50,
    save_path: Optional[str] = None,
) -> None:
    train_loss = np.asarray(train_loss_log, dtype=np.float32)
    if log_scale:
        train_loss = np.log10(train_loss + 1e-8)

    x_train = np.arange(len(train_loss))

    plt.figure(figsize=(10, 5))
    plt.title("Loss history (log10)" if log_scale else "Loss history")

    # 訓練 loss
    plt.plot(x_train, train_loss, ".", alpha=0.1, label="Train (raw)", color="blue")

    if len(train_loss) >= window:  # ✅ 只有長度足夠時才畫移動平均
        ma_train = moving_average(train_loss, window)
        valid_x_train = x_train[window - 1 :]
        std_train = np.array(
            [
                train_loss[i - window + 1 : i + 1].std()
                for i in range(window - 1, len(train_loss))
            ]
        )

        plt.plot(
            valid_x_train, ma_train, "-", color="blue", label=f"Train MA (w={window})"
        )
        plt.fill_between(
            valid_x_train,
            ma_train - std_train,
            ma_train + std_train,
            color="blue",
            alpha=0.2,
            label="Train ±1 std",
        )

    # === 驗證 loss (同樣檢查長度) ===
    if eval_loss_log is not None:
        eval_loss = np.asarray(eval_loss_log, dtype=np.float32)
        if log_scale:
            eval_loss = np.log10(eval_loss + 1e-8)
        x_eval = np.arange(len(eval_loss))
        plt.plot(x_eval, eval_loss, ".", alpha=0.3, label="Eval (raw)", color="orange")

        if len(eval_loss) >= window:
            ma_eval = moving_average(eval_loss, window)
            valid_x_eval = x_eval[window - 1 :]
            plt.plot(
                valid_x_eval,
                ma_eval,
                "-",
                color="orange",
                label=f"Eval MA (w={window})",
            )

    plt.xlabel("Epoch")
    plt.ylabel("Log Loss" if log_scale else "Loss")
    plt.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)

        # 以當前訓練 epoch 數命名檔案
        epoch_num = len(train_loss_log)
        fig_path = os.path.join(save_path, f"loss_epoch_{epoch_num}.png")

        plt.savefig(fig_path, dpi=150)

        np.savez(
            os.path.join(save_path, f"losses_{epoch_num}.npz"),
            train_loss=train_loss,
            eval_loss=eval_loss,
        )

    plt.show()


# ===============================================================
# region viz_epochs
def viz_epochs(batch_dict: Dict[str, torch.Tensor], max_channels: int = 9) -> None:
    """
    從 batch_dict 隨機挑選一個樣本，將前 max_channels 個通道可視化並拼接成圖。

    參數
    ----
    batch_dict : Dict[str, torch.Tensor]
        儲存多個 batch tensor 的字典，假設每個 tensor 的 shape 為 (B, C, H, W)
        - B: batch size
        - C: channel 數
        - H, W: 高與寬
    max_channels : int, default=9
        最多要視覺化的通道數量

    回傳
    ----
    None
        直接呼叫 plot_HW3 繪製圖像
    """
    # 從第一個 batch tensor 取得 batch size
    max_batch = next(iter(batch_dict.values())).shape[0]

    # 隨機挑選一個 sample index
    random_idx = np.random.randint(0, max_batch)

    out = []

    # 遍歷所有 batch_tensor
    for batch_tensor in batch_dict.values():
        # 取出一個樣本並轉換為 HWC，僅保留前 max_channels
        hwc = to_HWC(batch_tensor[random_idx])[..., :max_channels]

        # 將多通道轉換成 RGB（假設 channels_to_rgb 有這功能）
        chw3 = channels_to_rgb(hwc)
        chw3 = torch.from_numpy(chw3)

        # F.pad 的 pad 格式為 (W_left, W_right, H_top, H_bottom, C_front, C_back)
        chw3 = F.pad(chw3, (0, 0, 5, 0, 5, 0), value=1.0)

        # 拆成 max_channels 張圖，水平拼接
        chw3 = [chw3[i] for i in range(max_channels)]
        out.append(torch.hstack(chw3))

    # 倒序排列後再垂直拼接
    out.reverse()
    out = torch.vstack(out)

    # 繪製最終圖像
    out = out.detach().cpu().numpy()
    fig, ax = plt.subplots()
    ax.imshow(out, aspect="equal", origin="lower")
    ax.axis("off")
    plt.show()
