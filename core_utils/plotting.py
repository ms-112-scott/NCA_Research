import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Microsoft JhengHei']  # 設定中文字型
plt.rcParams['axes.unicode_minus'] = False            # 避免負號顯示錯誤

import numpy as np
import tensorflow as tf


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
