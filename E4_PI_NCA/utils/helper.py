import numpy as np
import torch
from typing import Union, List, Tuple
from scipy.ndimage import generic_filter
import datetime
from pathlib import Path
import os
import inspect
import ipynbname
import matplotlib.pyplot as plt
from matplotlib import cm


# ===========================================================================================
# region to_HWC
def to_HWC(arr: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    將輸入 3D 或 4D array / tensor 轉成 channel-last 格式。
    3D: (C, H, W) -> (H, W, C)
    4D: (B, C, H, W) -> (B, H, W, C)

    Parameters
    ----------
    arr : np.ndarray 或 torch.Tensor
        shape = (C,H,W) 或 (B,C,H,W)

    Returns
    -------
    np.ndarray 或 torch.Tensor
        channel-last array/tensor，與輸入同類型
    """
    if isinstance(arr, np.ndarray):
        if arr.ndim == 3:
            return np.transpose(arr, (1, 2, 0))
        elif arr.ndim == 4:
            return np.transpose(arr, (0, 2, 3, 1))
        else:
            raise ValueError(f"輸入 np.ndarray 維度 {arr.ndim} 不支援, 只支援 3D 或 4D")
    elif isinstance(arr, torch.Tensor):
        if arr.ndim == 3:
            return arr.permute(1, 2, 0)
        elif arr.ndim == 4:
            return arr.permute(0, 2, 3, 1)
        else:
            raise ValueError(
                f"輸入 torch.Tensor 維度 {arr.ndim} 不支援, 只支援 3D 或 4D"
            )
    else:
        raise TypeError(
            f"輸入類型 {type(arr)} 不支援, 只支援 np.ndarray 或 torch.Tensor"
        )


# ===========================================================================================
# region print_tensor_stats
def print_tensor_stats(x: torch.Tensor, name: str = "Tensor") -> None:
    """
    列印 3D 或 4D tensor 每個 channel的 min/max

    Parameters
    ----------
    x : torch.Tensor
        shape = (C,H,W) 或 (B,C,H,W)
    name : str
        tensor 名稱
    """
    if x.ndim == 3:
        C = x.shape[0]
        print(f"{name} (C,H,W) shape = {x.shape}")
        for c in range(C):
            print(
                f"  channel {c}: min={x[c].min().item():.6f}, max={x[c].max().item():.6f}"
            )
    elif x.ndim == 4:
        B, C, H, W = x.shape
        print(f"{name} (B,C,H,W) shape = {x.shape}")
        for c in range(C):
            ch_min = x[:, c, :, :].min().item()
            ch_max = x[:, c, :, :].max().item()
            print(f"  channel {c}: min={ch_min:.6f}, max={ch_max:.6f}")
    else:
        raise ValueError(f"{name} 不支援 ndim={x.ndim}, 只支援 3D 或 4D tensor")


# ===========================================================================================
# region split_cases
def split_cases(
    case_list: List[torch.Tensor],
    train_ratio: float = 0.7,
    eval_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    將一個 list of 3D tensors 隨機分為 train / eval / test 三個集合。

    參數
    ----------
    case_list : List[torch.Tensor]
        原始資料 list，每個 tensor shape=(C, H, W)
    train_ratio : float
        訓練集比例
    eval_ratio : float
        驗證集比例
    test_ratio : float
        測試集比例
    seed : int
        隨機種子，確保可重現

    回傳
    ----------
    Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]
        train_case_list, eval_case_list, test_case_list
    """
    if not np.isclose(train_ratio + eval_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + eval_ratio + test_ratio 必須等於 1.0")

    np.random.seed(seed)
    indices = np.random.permutation(len(case_list))

    n = len(case_list)
    n_train = int(n * train_ratio)
    n_eval = int(n * eval_ratio)
    # 剩下給 test
    n_test = n - n_train - n_eval

    train_cases = [case_list[i] for i in indices[:n_train]]
    eval_cases = [case_list[i] for i in indices[n_train : n_train + n_eval]]
    test_cases = [case_list[i] for i in indices[n_train + n_eval :]]

    return train_cases, eval_cases, test_cases


# ===========================================================================================
# region get_output_path
def get_output_path(Suffix: str = None) -> str:
    """
    建立輸出資料夾，會抓呼叫端的檔案名稱
    """
    notebook_path = ipynbname.path()

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if Suffix:
        path = f"../outputs/{notebook_path.stem}_{timestamp}/{Suffix}"
    else:
        path = f"../outputs/{notebook_path.stem}_{timestamp}"

    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)

    return str(output_path)


# ===========================================================================================
# region norm_CHW
def norm_CHW(arr: np.ndarray) -> np.ndarray:
    """
    對每個 channel 做 min-max normalization 到 [0,1]。
    輸入 arr 假設已經沒有 NaN。
    """
    arr_norm = np.empty_like(arr, dtype=np.float32)
    for c in range(arr.shape[0]):
        ch = arr[c]
        min_val, max_val = ch.min(), ch.max()
        if max_val > min_val:
            arr_norm[c] = (ch - min_val) / (max_val - min_val)
        else:
            arr_norm[c] = ch * 0.0
    return arr_norm


# ===========================================================================================
# region norm_CHW_select
def norm_CHW_select(arr: np.ndarray, channels: list[int]) -> np.ndarray:
    """
    對指定 channel 做 min-max normalization 到 [0,1]，其他 channel 保持不變。

    Parameters
    ----------
    arr : np.ndarray
        shape = (C, H, W)
    channels : list[int]
        要正規化的 channel index

    Returns
    -------
    np.ndarray
        shape = (C, H, W)
    """
    arr_norm = arr.copy().astype(np.float32)
    for c in channels:
        ch = arr[c]
        min_val, max_val = ch.min(), ch.max()
        if max_val > min_val:
            arr_norm[c] = (ch - min_val) / (max_val - min_val)
        else:
            arr_norm[c] = ch * 0.0
    return arr_norm


# ===========================================================================================
# region plot_HW3
def plot_HW3(hw3: Union[np.ndarray, torch.Tensor], show_axis: bool = False) -> None:
    """
    Plot a H x W x 3 RGB image (hw3), keeping x/y scale equal,
    optionally hiding axes.

    Parameters
    ----------
    hw3 : np.ndarray or torch.Tensor
        H x W x 3 RGB image with float values [0,1].
    show_axis : bool
        是否顯示座標軸。
    """
    if isinstance(hw3, torch.Tensor):
        hw3 = hw3.detach().cpu().numpy()

    fig, ax = plt.subplots()
    ax.imshow(hw3, aspect="equal", origin="lower")

    if not show_axis:
        ax.axis("off")

    plt.show()


##---------------------------------------------------------------------------------------------------------------------------
# region channels_to_rgb
def channels_to_rgb(
    image: Union[np.ndarray, torch.Tensor], cmap: str = "jet"
) -> np.ndarray:
    """
    將 H x W x C 的多通道影像，每個 channel 用 jet colormap 映射成 RGB，
    回傳 B x H x W x 3，其中 B=C。

    Parameters
    ----------
    image : np.ndarray 或 torch.Tensor
        H x W x C 影像

    Returns
    -------
    bhw3 : np.ndarray
        B x H x W x 3 RGB 映射後結果
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    # 限制像素值範圍 [0, 1]
    image = np.clip(image, 0.0, 1.0)

    H, W, C = image.shape
    bhw3 = np.zeros((C, H, W, 3), dtype=np.float32)

    cmap = cm.get_cmap(cmap)

    for i in range(C):
        channel = image[:, :, i]

        # 將 channel 映射成 RGB [0,1]
        rgb = cmap(channel)[:, :, :3]
        bhw3[i] = rgb

    return bhw3
