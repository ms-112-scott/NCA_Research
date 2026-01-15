import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Microsoft JhengHei"]  # 設定中文字型
plt.rcParams["axes.unicode_minus"] = False  # 避免負號顯示錯誤

import torch
from typing import Optional, List, Union
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, List
import numpy as np
import math
import random

# end region


##---------------------------------------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------------------
# region plt_HWC_split_channels
def plt_HWC_split_channels(
    image: Union[np.ndarray, torch.Tensor],
    histogram: bool = False,
    max_channels: Optional[int] = None,
    original_img_channels: List[int] = [0, 1, 2],
    channel_names: Optional[List[str]] = None,
    IfReturn: bool = False,
) -> None:
    """
    顯示多通道影像的每個通道分圖，並可選擇是否繪製直方圖。
    並在每個 channel subplot 旁邊加上 scale label (colorbar)。
    若 channel 數值全部相同，則在影像中央顯示該數字。
    """

    # 將 tensor 轉為 numpy
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    # 確定要顯示的通道數量
    num_channels = (
        image.shape[2] if max_channels is None else min(max_channels, image.shape[2])
    )

    # 準備通道名稱
    if channel_names is None:
        names = [
            "coord_y",
            "coord_x",
            "geo_mask",
            "topo",
            "windInitX",
            "windInitY",
            "uped",
            "vped",
            "Uped",
            "TKEped",
            "Tuwped",
            "dwdz",
        ]
    else:
        names = channel_names[:num_channels]
        if len(names) < num_channels:
            names += [f"hidden{i}" for i in range(len(names), num_channels)]

    # 決定 subplot 行列數
    nrows = 2 if histogram else 1
    ncols = num_channels + 1  # 原圖 + 每個通道

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2 * ncols, 2 * nrows))

    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    # === 顯示原始影像（只取前3通道當作RGB） ===
    original_img = image[:, :, original_img_channels]
    axes[0][0].imshow(original_img)  # 避免 index error，取前3通道
    axes[0][0].set_title("原始影像")
    axes[0][0].axis("off")

    # === 顯示每個通道 ===
    for i in range(num_channels):
        channel_data = image[:, :, i]
        max_val = np.max(channel_data)
        min_val = np.min(channel_data)

        im = axes[0][i + 1].imshow(channel_data, cmap="jet", vmin=min_val, vmax=max_val)
        axes[0][i + 1].set_title(names[i])
        axes[0][i + 1].axis("off")

        # 如果數值全相同 → 顯示數字
        if np.all(channel_data == channel_data.flat[0]):
            axes[0][i + 1].text(
                channel_data.shape[1] // 2,
                channel_data.shape[0] // 2,
                f"{channel_data.flat[0]:.3f}",
                color="white",
                ha="center",
                va="center",
                fontsize=10,
                weight="bold",
                bbox=dict(facecolor="black", alpha=0.6, boxstyle="round,pad=0.2"),
            )

        # 在每個 channel subplot 加 colorbar
        plt.colorbar(im, ax=axes[0][i + 1], fraction=0.046, pad=0.04)

        if histogram:
            hist, bins = np.histogram(channel_data.flatten(), bins=32, range=(0, 1))
            bin_centers = (bins[:-1] + bins[1:]) / 2
            axes[1][i + 1].plot(bin_centers, hist, color="black")
            axes[1][i + 1].set_title(f"{names[i]} 分佈")
            axes[1][i + 1].set_xlim(0, 1)
            axes[1][i + 1].axis("off")

    if histogram:
        axes[1][0].axis("off")

    plt.tight_layout()
    plt.show()


# end region


##---------------------------------------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------------------
# region plot_HW3
def plot_HW3(t: torch.Tensor, title: str = None):
    """
    繪製 shape = (H, W, 3) 的 tensor 圖像
    - t: torch.Tensor, shape (H, W, 3)，值域建議 [0,1] 或 [0,255]
    """
    if t.ndim != 3 or t.shape[-1] != 3:
        raise ValueError(f"輸入必須是 (H, W, 3)，目前是 {t.shape}")

    img = t.detach().cpu().numpy()

    # 如果數值大於1，假設是 0~255 範圍，轉成 0~1
    if img.max() > 1.0:
        img = img / 255.0

    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.show()


# end region


##---------------------------------------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------------------
# region plt_CFD_channels
def plt_CFD_channels(
    tensor: Union[np.ndarray, torch.Tensor],
    cols: int = 6,
    histogram: bool = False,
    channel_max: int = 18,
    title: Optional[List[str]] = None,
    figsize: float = 2.0,
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
        "X 方向速度",
        "Y 方向速度",
        "Z 方向速度",
        "流速大小",
        "壓力 p",
        "湍流動能 k",
        "渦動黏滯係數 νₜ",
        "湍流能耗率 ε",
        "X 標準化座標",
        "Y 標準化座標",
        "Z 標準化座標",
        "遮罩",
    ]

    # 轉 numpy，如果是 torch tensor
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    h, w, c = tensor.shape
    c = min(c, channel_max)  # 限制最多顯示的通道數

    rows_per_group = 2 if histogram else 1
    groups = (c + cols - 1) // cols  # ceil division

    total_rows = 1 + groups * rows_per_group  # 原圖 + 每組通道
    fig, axes = plt.subplots(
        total_rows, cols, figsize=(cols * figsize, total_rows * figsize)
    )

    # 統一 axes 維度
    if total_rows == 1:
        axes = np.expand_dims(axes, 0)
    elif cols == 1:
        axes = np.expand_dims(axes, 1)

    # === 原圖位置，第一行第一列 ===
    for ax in axes[0]:
        ax.axis("off")
    axes[0][0].imshow(tensor[..., :3])  # 前 3 通道當作 RGB
    axes[0][0].set_title("Uxyz embed CFD img", fontsize=8)

    # === 顯示每個 channel 分圖 ===
    for i in range(c):
        group_idx = i // cols
        col_idx = i % cols

        # 決定標題
        if title is None:
            plt_title = (
                default_title[i]
                if i < len(default_title)
                else f"hidden Ch {i - len(default_title)}"
            )
        else:
            plt_title = title[i] if i < len(title) else f"hidden Ch {i - len(title)}"

        row_img = 1 + group_idx * rows_per_group
        ax_img = axes[row_img][col_idx]

        # 取得通道資料及顏色刻度
        channel_data = tensor[..., i]
        vmin, vmax = np.min(channel_data), np.max(channel_data)

        # 顯示通道圖
        ax_img.imshow(channel_data, cmap="jet", vmin=vmin, vmax=vmax)
        ax_img.set_title(plt_title, fontsize=figsize * 3)
        ax_img.axis("off")

        # 畫 histogram
        if histogram:
            row_hist = row_img + 1
            ax_hist = axes[row_hist][col_idx]
            ax_hist.hist(channel_data.ravel(), bins=50, color="gray")
            ax_hist.set_title(f"Hist {i}", fontsize=figsize * 3)
            ax_hist.axis("off")

    plt.tight_layout()
    plt.show()


# end region


##---------------------------------------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------------------
# region plt_random_cfd_slice
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
    columns = [
        "X",
        "Y",
        "Z",
        "Xnorm",
        "Ynorm",
        "Znorm",
        "Ux",
        "Uy",
        "Uz",
        "Umag",
        "p",
        "k",
    ]
    channel_name = columns[c] if c < len(columns) else f"Channel {c}"

    # 擷取切片資料
    slice_2d = data[t, z, :, :, c]

    # 繪製圖像
    plt.figure(figsize=(8, 6))
    im = plt.imshow(slice_2d, cmap="viridis", origin="lower", aspect="auto")
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


# end region


##---------------------------------------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------------------
# region plt_acc_over_time
def plt_acc_over_time(
    acc_list: List[np.ndarray],
    title: str = "Accuracy over time",
    ylabel: str = "Metric",
    smooth_window: int = 1,
    log_scale: bool = True,
):
    """
    繪製多條 metric 隨時間變化折線圖。
    - 前面部分淡化 (alpha=0.05)
    - 後 10 條線漸層
    - 最後一條線加粗且用紅色
    """
    plt.figure(figsize=(8, 5))

    n = len(acc_list)
    n_fade = max(0, n - 10)  # 前面需要淡化的數量
    for i, arr in enumerate(acc_list):
        if isinstance(arr, list):
            arr = np.array(arr)

        # 判斷線條顏色和粗細
        if i == n - 1:
            color = "red"
            linewidth = 2.0
            alpha = 1.0
        elif i < n_fade:
            color = "tab:blue"
            linewidth = 1.0
            alpha = 0.005
        else:
            # 後 10 條線漸層
            idx = i - n_fade
            alpha = 0.3 + 0.7 * (idx / 9)  # 0.3~1.0
            color = "tab:blue"
            linewidth = 1.0

        # 原始線
        plt.plot(
            range(1, len(arr) + 1),
            arr,
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            label=f"Run {i+1}" if i == n - 1 else None,
        )

        # 平滑線
        if smooth_window > 1 and len(arr) >= smooth_window:
            smooth_values = np.convolve(
                arr, np.ones(smooth_window) / smooth_window, mode="valid"
            )
            plt.plot(
                range(1, len(smooth_values) + 1),
                smooth_values,
                color=color,
                alpha=alpha,
                linestyle="--",
                linewidth=linewidth,
            )

    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    if log_scale:
        plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.show()


# end region


##---------------------------------------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------------------
# region plt_metrics_dict


def moving_average(x, w=10):
    """動態平均 (rolling average)"""
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plt_metrics_dict(metrics_log, window=10):
    metric_names = [
        "data_mae_score",
        "structural_ssim_score",
        "physical_p_k_neg_score",
        "physical_p_flux_score",
        "physical_l_u_bound_score",
        "physical_continuity_score",
        "COMPOSITE_SCORE",
    ]

    # 取 metrics
    data = {k: np.array([m[k] for m in metrics_log]) for k in metric_names}

    # Normalize
    norm_data = {}
    for k, v in data.items():
        minv, maxv = v.min(), v.max()
        norm_data[k] = (
            np.zeros_like(v) if maxv - minv < 1e-12 else (v - minv) / (maxv - minv)
        )

    plt.figure(figsize=(10, 6))

    for k in metric_names:
        y = norm_data[k]
        y_smooth = moving_average(y, w=window)
        smooth_epochs = np.arange(len(y_smooth))
        plt.plot(smooth_epochs, y_smooth, linewidth=2, label=k)

    plt.title(f"訓練指標趨勢 (Moving Average = {window})")
    plt.xlabel("Epoch")
    plt.ylabel("標準化值 (0~1)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
