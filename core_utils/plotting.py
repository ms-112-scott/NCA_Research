import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Microsoft JhengHei']  # 設定中文字型
plt.rcParams['axes.unicode_minus'] = False            # 避免負號顯示錯誤

import torch

from typing import Optional, Union, List
import numpy as np
import math
##---------------------------------------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------------------
#region plt_HWC_split_channels
def plt_HWC_split_channels(
    image: Union[np.ndarray, torch.Tensor],
    histogram: bool = False,
    max_channels: Optional[int] = None
) -> None:
    """
    顯示多通道影像的每個通道分圖，並可選擇是否繪製直方圖。

    Args:
        image: shape 為 (H, W, C) 的 torch.Tensor 或 np.ndarray
        histogram: 若為 True，顯示每個通道的像素分佈直方圖
        max_channels: 若指定，僅顯示前 max_channels 個通道
    """

    # 將 tensor 轉為 numpy，若是 GPU tensor 先搬到 CPU
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    # flip vertically（上下翻轉）
    image = image[::-1, :, :]

    # 限制像素值範圍 [0, 1]
    image = np.clip(image, 0.0, 1.0)

    # 確定要顯示的通道數量
    num_channels = image.shape[2] if max_channels is None else min(max_channels, image.shape[2])

    # 指定每個通道的 colormap
    channel_cmaps = ['Reds', 'Greens', 'Blues', 'gray', 'gray']
    if num_channels > len(channel_cmaps):
        channel_cmaps += ['gray'] * (num_channels - len(channel_cmaps))

    # 決定 subplot 行列數
    nrows = 2 if histogram else 1
    ncols = num_channels + 1  # 原圖 + 每個通道

    # 建立 subplot
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2 * ncols, 2 * nrows))

    # 若只有一行，統一 axes 維度
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    # === 顯示原始影像（只取前3通道當作RGB） ===
    axes[0][0].imshow(image[:, :, :3])
    axes[0][0].set_title('原始影像')
    axes[0][0].axis('off')

    # === 顯示每個通道分圖 ===
    for i in range(num_channels):
        channel_data = image[:, :, i]
        cmap = channel_cmaps[i % len(channel_cmaps)]

        # 顯示單通道圖
        axes[0][i + 1].imshow(channel_data, cmap=cmap)
        axes[0][i + 1].set_title(f'通道 {i}')
        axes[0][i + 1].axis('off')

        # 若需要直方圖
        if histogram:
            hist, bins = np.histogram(channel_data.flatten(), bins=32, range=(0, 1))
            bin_centers = (bins[:-1] + bins[1:]) / 2
            axes[1][i + 1].plot(bin_centers, hist, color='black')
            axes[1][i + 1].set_title(f'通道 {i} 分佈')
            axes[1][i + 1].set_xlim(0, 1)
            axes[1][i + 1].axis('off')

    # 原圖下方留空
    if histogram:
        axes[1][0].axis('off')

    plt.tight_layout()
    plt.show()
##---------------------------------------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------------------
#region plt_CFD_channels
from typing import Optional, List, Union
import torch
import numpy as np
import matplotlib.pyplot as plt

def plt_CFD_channels(
    tensor: Union[np.ndarray, torch.Tensor],
    cols: int = 6,
    histogram: bool = False,
    channel_max: int = 18,
    title: Optional[List[str]] = None,
    figsize: float = 2.0
) -> None:
    """
    顯示 CFD 模擬多通道影像，每個通道單獨分圖，可選擇繪製 histogram。
    
    Args:
        tensor: shape (H, W, C) 的 torch.Tensor 或 np.ndarray
        cols: 每行 subplot 的列數
        histogram: 是否顯示 histogram
        channel_max: 顯示最大通道數
        title: 每個 channel 對應標題
        figsize: 每個 subplot 尺寸比例
    """
    
    # 內建預設 channel 標題
    default_title = [
        "X 方向速度", "Y 方向速度", "Z 方向速度", "流速大小", "壓力 p",
        "湍流動能 k", "渦動黏滯係數 νₜ", "湍流能耗率 ε",
        "X 標準化座標", "Y 標準化座標", "Z 標準化座標", "遮罩"
    ]
    
    # 轉 numpy，如果是 torch tensor
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    
    h, w, c = tensor.shape
    c = min(c, channel_max)  # 限制最多顯示的通道數
    
    rows_per_group = 2 if histogram else 1
    groups = (c + cols - 1) // cols  # ceil division
    
    total_rows = 1 + groups * rows_per_group  # 原圖 + 每組通道
    fig, axes = plt.subplots(total_rows, cols, figsize=(cols * figsize, total_rows * figsize))
    
    # 統一 axes 維度
    if total_rows == 1:
        axes = np.expand_dims(axes, 0)
    elif cols == 1:
        axes = np.expand_dims(axes, 1)
    
    # === 原圖位置，第一行第一列 ===
    for ax in axes[0]:
        ax.axis('off')
    axes[0][0].imshow(tensor[..., :3])  # 前 3 通道當作 RGB
    axes[0][0].set_title("Uxyz embed CFD img", fontsize=8)
    
    # === 顯示每個 channel 分圖 ===
    for i in range(c):
        group_idx = i // cols
        col_idx = i % cols
        
        # 決定標題
        if title is None:
            plt_title = default_title[i] if i < len(default_title) else f"hidden Ch {i - len(default_title)}"
        else:
            plt_title = title[i] if i < len(title) else f"hidden Ch {i - len(title)}"
        
        row_img = 1 + group_idx * rows_per_group
        ax_img = axes[row_img][col_idx]
        
        # 取得通道資料及顏色刻度
        channel_data = tensor[..., i]
        vmin, vmax = np.min(channel_data), np.max(channel_data)
        
        # 顯示通道圖
        ax_img.imshow(channel_data, cmap='jet', vmin=vmin, vmax=vmax)
        ax_img.set_title(plt_title, fontsize=figsize * 3)
        ax_img.axis('off')
        
        # 畫 histogram
        if histogram:
            row_hist = row_img + 1
            ax_hist = axes[row_hist][col_idx]
            ax_hist.hist(channel_data.ravel(), bins=50, color='gray')
            ax_hist.set_title(f"Hist {i}", fontsize=figsize * 3)
            ax_hist.axis('off')
    
    plt.tight_layout()
    plt.show()
##---------------------------------------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------------------
#region plt_random_cfd_slice
def plt_random_cfd_slice(data, show_info=True):
    """
    顯示隨機 timestep、Z 切片、channel 的 CFD 2D heatmap。
    
    Args:
        data (np.ndarray): CFD 資料 (T, Z, Y, X, C)
        show_info (bool): 是否輸出索引與欄位資訊
    """
    T, Z, Y, X, C = data.shape

    # 隨機選擇維度索引
    t = random.randint(0, T - 1)
    z = random.randint(0, Z - 1)
    c = random.randint(0, C - 1)

    # 對應欄位名稱
    columns = ["X", "Y", "Z", "Xnorm", "Ynorm", "Znorm", "Ux", "Uy", "Uz", "Umag", "p", "k"]
    channel_name = columns[c] if c < len(columns) else f"Channel {c}"

    # 擷取切片資料
    slice_2d = data[t, z, :, :, c]

    # 繪製圖像
    plt.figure(figsize=(8, 6))
    im = plt.imshow(slice_2d, cmap='viridis', origin='lower', aspect='auto')
    plt.colorbar(im, label=channel_name)
    plt.title(f"Timestep: {t}, Z-slice: {z}, Channel: {channel_name} (Index {c})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()

    if show_info:
        print(f"[Info] Data shape: {data.shape}")
        print(f"→ Timestep index = {t}")
        print(f"→ Z index        = {z}")
        print(f"→ Channel index  = {c} ({channel_name})")