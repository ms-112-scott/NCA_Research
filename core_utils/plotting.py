import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Microsoft JhengHei']  # 設定中文字型
plt.rcParams['axes.unicode_minus'] = False            # 避免負號顯示錯誤

import numpy as np
import tensorflow as tf
import math
##---------------------------------------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------------------
#region plt_split_channels
def plt_split_channels(image, histogram=False, max_channels=None):
    """
    顯示多通道影像的每個通道分圖，並可選擇是否繪製直方圖。

    Args:
        image: shape 為 (H, W, C) 的 tensor 或 ndarray
        histogram: 若為 True，顯示每個通道的像素分佈直方圖
    """

    # 如果是 TensorFlow tensor，取出前一張並轉換為 numpy
    if isinstance(image, tf.Tensor):
        image = image.numpy()

    image = np.clip(image, 0, 1)  # 限制像素值範圍為 [0, 1]

    if max_channels is None:
        num_channels = image.shape[2]  # 通道數量
    else:
        num_channels = max_channels

    # 指定每個通道的 colormap（不足會用 gray 補齊）
    channel_cmaps = ['Reds', 'Greens', 'Blues', 'gray', 'gray']
    channel_cmaps += ['gray'] * (num_channels - len(channel_cmaps))

    # 根據是否要畫直方圖，決定 subplot 行數與列數
    nrows = 2 if histogram else 1
    ncols = num_channels + 1  # +1 是原圖

    # 建立 subplot 圖表
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2 * ncols, 2 * nrows))

    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)  # 統一 axes 維度：[row][col]

    # === 顯示原始影像（只取前3通道當作RGB） ===
    axes[0][0].imshow(image[:, :, :3])
    axes[0][0].set_title('原始影像')
    axes[0][0].axis('off')

    # === 每個通道單獨顯示 ===
    for i in range(num_channels):
        channel_data = image[:, :, i]  # 單一通道資料
        cmap = channel_cmaps[i % len(channel_cmaps)]  # 對應 colormap

        # 顯示 channel 圖像
        axes[0][i + 1].imshow(channel_data, cmap=cmap)
        axes[0][i + 1].set_title(f'通道 {i}')
        axes[0][i + 1].axis('off')

        # 若需要，繪製 histogram
        if histogram:
            hist, bins = np.histogram(channel_data.flatten(), bins=32, range=(0, 1))
            bin_centers = (bins[:-1] + bins[1:]) / 2
            axes[1][i + 1].plot(bin_centers, hist, color='black')
            axes[1][i + 1].set_title(f'通道 {i} 分佈')
            axes[1][i + 1].set_xlim(0, 1)
            axes[1][i + 1].axis('off')

    # 若有直方圖，原圖下方位置留白
    if histogram:
        axes[1][0].axis('off')

    plt.tight_layout()
    plt.show()

##---------------------------------------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------------------
#region plt_CFD_channels
def plt_CFD_channels(tensor, cols=6, histogram=False, channel_max=18, title=None, figsize=2):
    default_title = [
        "X 方向速度", "Y 方向速度", "Z 方向速度", "流速大小", "壓力 p", "湍流動能 k", "渦動黏滯係數 νₜ", "湍流能耗率 ε",
        "X 標準化座標", "Y 標準化座標", "Z 標準化座標", "遮罩",
    ]

    # Tensor shape: (H, W, C)
    h, w, c = tensor.shape
    c = min(c, channel_max)
    rows_per_group = 2 if histogram else 1
    groups = (c + cols - 1) // cols  # ceil division

    total_rows = 1 + groups * rows_per_group
    fig, axes = plt.subplots(total_rows, cols, figsize=(cols * figsize, total_rows * figsize))

    if total_rows == 1:
        axes = np.expand_dims(axes, 0)

    # Original image (first row, first column)
    for ax in axes[0]:
        ax.axis('off')
    axes[0][0].imshow(tensor[..., :3])
    axes[0][0].set_title("Uxyz embed CFD img", fontsize=8)

    # Channel plots and histograms
    for i in range(c):
        group_idx = i // cols
        col_idx = i % cols

        if title is None:
            if i >= len(default_title):
                plt_title = f"hidden Ch {i - len(default_title)}"
            else:
                plt_title = default_title[i]
        else:
            if i >= len(title):
                plt_title = f"hidden Ch {i - len(title)}"
            else:
                plt_title = title[i]

        row_img = 1 + group_idx * rows_per_group
        ax_img = axes[row_img][col_idx]

        # 取得該 channel 最小最大值做顏色刻度
        # 若 tensor 是 tf.Tensor，先轉 numpy
        channel_data = tensor[..., i]
        if isinstance(channel_data, tf.Tensor):
            channel_data = channel_data.numpy()
        vmin = np.min(channel_data)
        vmax = np.max(channel_data)

        ax_img.imshow(channel_data, cmap='jet', vmin=vmin, vmax=vmax)
        ax_img.set_title(plt_title, fontsize=figsize*3)
        ax_img.axis('off')

        if histogram:
            row_hist = row_img + 1
            ax_hist = axes[row_hist][col_idx]
            ax_hist.hist(channel_data.ravel(), bins=50, color='gray')
            ax_hist.set_title(f"Hist {i}", fontsize=figsize*3)
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