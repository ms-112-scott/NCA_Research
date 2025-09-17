import os
from datetime import datetime
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
import matplotlib.pyplot as plt

from core_utils.utils_image import imshow, tile2d
from core_utils.utils_io import imwrite


# ===============================================================
# region viz_pool
def viz_pool(pool: torch.Tensor, step_i: int, output_path: str = "train_log") -> None:
    """
    將 pool (BHWC tensor) 中前 49 張圖像拼貼，並加上四邊漸變淡出效果，
    最後輸出圖片存檔 (train_log/%04d_pool.jpg)。
    """

    # 轉換為 numpy

    tiled_pool: np.ndarray = tile2d(pool[:49])  # (H, W, 3)
    H, W, C = tiled_pool.shape

    fade = np.linspace(1.0, 0.0, 72)

    # 左邊淡出
    tiled_pool[:, :72, :] *= fade[None, :, None]
    # 右邊淡出
    tiled_pool[:, -72:, :] *= fade[None, ::-1, None]
    # 上方淡出
    tiled_pool[:72, :, :] *= fade[:, None, None]
    # 下方淡出
    tiled_pool[-72:, :, :] *= fade[::-1, None, None]

    imwrite(f"{output_path}/{step_i:04d}_pool.jpg", (tiled_pool * 255).astype(np.uint8))
    imshow(tiled_pool)


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
        print(f"[✔] Loss curve saved to: {fig_path}")

    plt.show()
