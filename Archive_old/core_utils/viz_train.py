import os
from datetime import datetime
from typing import Any, Optional, Sequence, Union, Dict, Tuple
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from core_utils.utils_image import imshow, tile2d
from core_utils.utils_io import imwrite
from E4_PI_NCA.utils.helper import channels_to_rgb, to_HWC


# ===============================================================
# region viz_pool
def viz_pool(
    pool,
    step_i: int = None,
    output_path: str = None,
    title: str = None,
    show_channel: list[int] = [6, 7, 8],
    show_all: bool = False,
) -> None:
    """
    將 pool (BCHW tensor 或 np.ndarray) 圖像拼貼。
    **使用 channel 2 作為 0/1 mask (0=白色, 1=保留原色)。**

    - 如果 show_channel 為 3-channel (e.g., [6,7,8]), 則顯示 RGB。
    - 如果 show_channel 為 1-channel (e.g., [6]), 則使用 'jet' colormap。
    - **Mask (白色) 會在 colormap 之後才套用。**

    參數:
        pool        : BCHW tensor 或 np.ndarray。 (假設 channel 2 是 0/1 mask)
        step_i      : 當前步數 / 迭代數，用於檔名 (可為 None)
        output_path : 儲存圖片的資料夾路徑 (可選)
        title       : 圖像標題 (可選)
        show_channel: list[int]，指定要顯示的資料通道 (0-index)。
        show_all    : bool, 是否顯示整個 pool，預設 False
    """

    # -------------------------
    # 1️⃣ 判斷型別並轉成 BHWC
    # -------------------------
    if isinstance(pool, np.ndarray):
        pool_bhwc = np.transpose(pool, (0, 2, 3, 1))  # BCHW -> BHWC
    elif torch.is_tensor(pool):
        pool_bhwc = pool.permute(0, 2, 3, 1).detach().cpu().numpy()  # BCHW -> BHWC
    else:
        raise TypeError(f"pool 必須是 torch.Tensor 或 np.ndarray，但收到 {type(pool)}")

    B, H, W, C = pool_bhwc.shape

    # -------------------------
    # 2️⃣ 提取 Mask 並選擇顯示通道
    # -------------------------
    if C > 2:
        mask_bhwc = pool_bhwc[..., 2:3]  # (B, H, W, 1)
    else:
        mask_bhwc = np.ones((B, H, W, 1), dtype=pool_bhwc.dtype)

    current_show_channel = show_channel
    if current_show_channel is None:
        current_show_channel = [6, 7, 8]

    current_show_channel = [c for c in current_show_channel if c < C]

    if not current_show_channel:
        current_show_channel = [c for c in [0, 1] if c < C]
    if not current_show_channel and C > 0:
        current_show_channel = [0]

    if not current_show_channel:  # C=0
        display_data_bhwc = np.zeros((B, H, W, 1), dtype=pool_bhwc.dtype)
    else:
        display_data_bhwc = pool_bhwc[..., current_show_channel]  # (B, H, W, C_display)

    # -------------------------
    # 3️⃣ 選取要顯示的圖像 (包含 mask)
    # -------------------------
    if show_all:
        subpool_data = display_data_bhwc
        subpool_mask = mask_bhwc
    else:
        ten_percent = max(1, B // 10)
        # Data
        front_data = display_data_bhwc[:ten_percent]
        back_data = display_data_bhwc[-ten_percent:]
        subpool_data = np.concatenate([front_data, back_data], axis=0)
        # Mask
        front_mask = mask_bhwc[:ten_percent]
        back_mask = mask_bhwc[-ten_percent:]
        subpool_mask = np.concatenate([front_mask, back_mask], axis=0)

    # -------------------------
    # 4️⃣ 拼貼
    # -------------------------
    tiled_pool = tile2d(
        subpool_data,
    )  # (H_tiled, W_tiled, C_display)
    tiled_mask = tile2d(
        subpool_mask,
    )  # (H_tiled, W_tiled, 1)

    # -------------------------
    # 5️⃣ 核心: 處理 Colormap 和 Mask (套用 jet 之後才上白色)
    # -------------------------
    C_display = tiled_pool.shape[-1]
    white_color = np.array([1.0, 1.0, 1.0])  # 定義白色

    # 我們需要一個最終的 RGB 圖像用於顯示和儲存
    final_image_rgb = None

    if C_display == 1:
        # ----- 1-Channel (Jet) Case -----
        tiled_pool_1ch = tiled_pool.squeeze(-1)  # (H, W)
        tiled_mask_1ch = tiled_mask.squeeze(-1)  # (H, W)

        # a. 找出 "非 mask 區域" 的 vmin, vmax
        data_pixels = tiled_pool_1ch[tiled_mask_1ch == 1]
        vmin = data_pixels.min().item() if data_pixels.size > 0 else 0.0
        vmax = data_pixels.max().item() if data_pixels.size > 0 else 1.0

        # b. 手動正規化並套用 'jet' colormap
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap("jet")

        # 'cmap' 會回傳 (H, W, 4) RGBA 圖像
        image_jet_rgba = cmap(norm(tiled_pool_1ch))
        image_jet_rgb = image_jet_rgba[..., :3]  # 取 RGB, (H, W, 3)

        # c. 套用 mask: 0=white, 1=jet color
        #    需要將 (H, W) 的 mask 擴展到 (H, W, 3) 來做 np.where
        mask_rgb = np.expand_dims(tiled_mask_1ch, axis=-1)
        final_image_rgb = np.where(mask_rgb == 0, white_color, image_jet_rgb)

    elif C_display == 3:
        # ----- 3-Channel (RGB) Case -----
        # 邏輯同上, 只是 "原始" 圖像就是 RGB
        final_image_rgb = np.where(tiled_mask == 0, white_color, tiled_pool)

    else:
        # ----- Fallback (e.g., 2 channels) -----
        # 舊邏輯: 先套 mask (1.0), 再用 cmap
        # (這會導致 1.0 變紅色, 但這是邊界情況)
        print(f"Warning: 顯示 {C_display} 個通道, mask 可能無法正確顯示為白色。")
        tiled_pool_masked = np.where(tiled_mask == 0, 1.0, tiled_pool)

        # 儲存和顯示
        if output_path is not None:
            fname = f"{step_i:04d}_pool.jpg" if step_i is not None else "pool.jpg"
            imwrite(
                f"{output_path}/{fname}", (tiled_pool_masked * 255).astype(np.uint8)
            )

        vmin = tiled_pool_masked.min().item()
        vmax = tiled_pool_masked.max().item()
        plt.imshow(tiled_pool_masked, vmin=vmin, vmax=vmax, cmap="jet")  # 原始 cmap
        plt.axis("off")
        if title is not None:
            plt.title(title)
        plt.show()
        return  # 提早結束

    # -------------------------
    # 6️⃣ 儲存 & 顯示 (for 1-ch and 3-ch cases)
    # -------------------------

    # 確保 final_image_rgb 在 [0, 1] 範圍內 (np.where 可能引入 > 1.0)
    final_image_rgb = np.clip(final_image_rgb, 0.0, 1.0)

    # 儲存
    if output_path is not None:
        fname = f"{step_i:04d}_pool.jpg" if step_i is not None else "pool.jpg"
        imwrite(f"{output_path}/{fname}", (final_image_rgb * 255).astype(np.uint8))

    # 顯示 (現在 final_image_rgb 是一個 float [0,1] 的 RGB 圖像)
    plt.imshow(final_image_rgb)  # 不再需要 vmin, vmax, cmap
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.show()


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


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """簡單移動平均"""
    return np.convolve(x, np.ones(w) / w, mode="valid")


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

    # 訓練 loss 原始點
    plt.plot(x_train, train_loss, ".", alpha=0.1, label="Train (raw)", color="blue")

    # 動態調整 window (如果 epoch 太少)
    if len(train_loss) > 0:
        win = min(window, len(train_loss))
        ma_train = moving_average(train_loss, win)
        valid_x_train = x_train[win - 1 :]
        std_train = np.array(
            [
                train_loss[i - win + 1 : i + 1].std()
                for i in range(win - 1, len(train_loss))
            ]
        )

        plt.plot(
            valid_x_train, ma_train, "-", color="blue", label=f"Train MA (w={win})"
        )
        plt.fill_between(
            valid_x_train,
            ma_train - std_train,
            ma_train + std_train,
            color="blue",
            alpha=0.2,
            label="Train ±1 std",
        )

    # === 驗證 loss (同樣處理 window 動態) ===
    if eval_loss_log is not None:
        eval_loss = np.asarray(eval_loss_log, dtype=np.float32)
        if log_scale:
            eval_loss = np.log10(eval_loss + 1e-8)

        x_eval = np.arange(len(eval_loss))
        plt.plot(x_eval, eval_loss, ".", alpha=0.3, label="Eval (raw)", color="orange")

        if len(eval_loss) > 0:
            win_eval = min(window, len(eval_loss))
            ma_eval = moving_average(eval_loss, win_eval)
            valid_x_eval = x_eval[win_eval - 1 :]
            plt.plot(
                valid_x_eval,
                ma_eval,
                "-",
                color="orange",
                label=f"Eval MA (w={win_eval})",
            )

    plt.xlabel("Epoch")
    plt.ylabel("Log Loss" if log_scale else "Loss")
    plt.legend()
    plt.tight_layout()

    # 儲存檔案
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        epoch_num = len(train_loss_log)
        fig_path = os.path.join(save_path, f"loss_epoch_{epoch_num}.png")
        plt.savefig(fig_path, dpi=150)
        np.savez(
            os.path.join(save_path, f"losses_{epoch_num}.npz"),
            train_loss=train_loss,
            eval_loss=eval_loss if eval_loss_log is not None else None,
        )

    plt.show()


# ===============================================================
# region viz_batch_channels
def viz_batch_channels(
    batch_dict: Dict[str, torch.Tensor], show_channels: Tuple[int, int] = (0, 9)
) -> None:
    """
    從 batch_dict 隨機挑選一個樣本，將指定區間 [min_channel, max_channel) 的通道可視化並拼接成圖。

    參數
    ----
    batch_dict : Dict[str, torch.Tensor]
        儲存多個 batch tensor 的字典，假設每個 tensor 的 shape 為 (B, C, H, W)
        - B: batch size
        - C: channel 數
        - H, W: 高與寬
    show_channels : Tuple[int, int], default=(0,9)
        要視覺化的通道範圍 [min, max)，左閉右開

    回傳
    ----
    None
    """
    # 從第一個 batch tensor 取得 batch size
    max_batch = next(iter(batch_dict.values())).shape[0]

    # 隨機挑選一個 sample index
    random_idx = np.random.randint(0, max_batch)

    out = []

    # 遍歷所有 batch_tensor
    for batch_tensor in batch_dict.values():
        # 取出一個樣本並轉換為 HWC
        hwc = to_HWC(batch_tensor[random_idx])

        # 只取指定通道範圍
        hwc = hwc[..., show_channels[0] : show_channels[1]]

        # 將多通道轉換成 RGB（假設 channels_to_rgb 有這功能）
        chw3 = channels_to_rgb(hwc)
        chw3 = torch.from_numpy(chw3)

        # F.pad 的 pad 格式為 (W_left, W_right, H_top, H_bottom, C_front, C_back)
        chw3 = F.pad(chw3, (0, 0, 5, 0, 5, 0), value=1.0)

        # 按照實際切片的通道數拼接
        num_channels = show_channels[1] - show_channels[0]
        chw3 = [chw3[i] for i in range(num_channels)]
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


# ===============================================================
# region viz_batch_samples
def viz_batch_samples(
    batch_dict: Dict[str, torch.Tensor], channel_start: int = 0, max_batch: int = None
) -> None:
    """
    顯示 batch_dict 中每個 tensor 的 batch，row = tensor key, col = batch index。
    - 每個 batch 取連續 3 個通道做 RGB
    - 每個 tensor 對應一行
    - 限制每個 tensor 顯示的最大 batch 數量 max_batch
    """
    out_rows = []

    for batch_tensor in batch_dict.values():
        B, C, H, W = batch_tensor.shape
        if max_batch is None:
            B_show = B
        else:
            B_show = min(B, max_batch)  # 限制 batch 數量
        row_imgs = []

        for b in range(B_show):
            sample = batch_tensor[b]  # (C,H,W)
            ch_start = max(0, channel_start)
            ch_end = min(C, ch_start + 3)
            rgb_tensor = sample[ch_start:ch_end]  # (3,H,W) 或少於3通道

            # 補足3通道
            if rgb_tensor.shape[0] < 3:
                pad_channels = 3 - rgb_tensor.shape[0]
                rgb_tensor = F.pad(rgb_tensor, (0, 0, 0, 0, 0, pad_channels))

            # HWC
            rgb_hwc = rgb_tensor.permute(1, 2, 0).detach().cpu().numpy()
            # 可選 pad 分隔每張圖
            rgb_hwc = np.pad(rgb_hwc, ((0, 0), (5, 0), (0, 0)), constant_values=1.0)
            row_imgs.append(rgb_hwc)

        # 水平拼接同一 tensor 的所有 batch
        row = np.hstack(row_imgs)
        out_rows.append(row)

    # 垂直堆疊所有 tensor
    out_rows = out_rows[::-1]
    out_img = np.vstack(out_rows)

    # 繪圖
    plt.figure(figsize=(12, 4))
    plt.imshow(out_img, aspect="equal", origin="lower")
    plt.axis("off")
    plt.show()
