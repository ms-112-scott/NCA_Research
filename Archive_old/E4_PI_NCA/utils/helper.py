import numpy as np
import torch
from typing import Union, List, Tuple, Dict, Optional
from scipy.ndimage import generic_filter
import datetime
from pathlib import Path
import os
import inspect
import ipynbname
import matplotlib.pyplot as plt
from matplotlib import cm
import json
import random
import re


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
# region print_dict_stats
def print_dict_stats(d: dict, prefix: str = ""):
    """
    遞迴列印 dict tree 的統計資訊。

    Parameters
    ----------
    d : dict
        要列印的字典
    prefix : str
        用於遞迴縮排
    """
    for k, v in d.items():
        key_str = f"{prefix}{k}"
        if isinstance(v, dict):
            print(f"{key_str}: dict")
            print_dict_stats(v, prefix=prefix + "  ")
        elif isinstance(v, torch.Tensor):
            print(f"{key_str}: torch.Tensor, shape={tuple(v.shape)}, dtype={v.dtype}")
        elif isinstance(v, np.ndarray):
            print(f"{key_str}: np.ndarray, shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"{key_str}: {type(v)}")


# ===========================================================================================
# region print_tensor_stats
def print_tensor_stats(
    x, name: str = "Tensor", max_C: int = None, as_plot: bool = False
) -> None:
    """
    輸出 Tensor/Numpy channel-wise 統計資訊，支援表格或 boxplot。

    Parameters
    ----------
    x : torch.Tensor | np.ndarray
        shape = (C,H,W) 或 (B,C,H,W)
    name : str
        資料名稱
    max_C : int, optional
        最多顯示多少個 channel
    as_plot : bool, default=False
        True → 畫 boxplot
        False → 印出統計表
    """

    # ------------------------------------------------------------
    # 型態統一：轉成 numpy
    # ------------------------------------------------------------
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    elif not isinstance(x, np.ndarray):
        raise TypeError(f"{name} 必須是 torch.Tensor 或 np.ndarray")

    # ------------------------------------------------------------
    # 檢查維度並擷取 channel
    # ------------------------------------------------------------
    if x.ndim == 3:  # (C, H, W)
        C = x.shape[0]
        data = [x[c].flatten() for c in range(C)]
    elif x.ndim == 4:  # (B, C, H, W)
        B, C, H, W = x.shape
        data = [x[:, c, :, :].flatten() for c in range(C)]
    else:
        raise ValueError(f"{name} 維度錯誤：期望 3D 或 4D，但得到 ndim={x.ndim}")

    if max_C is not None:
        data = data[:max_C]

    # ------------------------------------------------------------
    # 輸出統計
    # ------------------------------------------------------------
    if as_plot:
        # --- boxplot ---
        plt.figure(figsize=(min(len(data), 6), 3))
        plt.boxplot(data, labels=[f"ch{c}" for c in range(len(data))], showfliers=False)
        plt.title(f"{name} Channel-wise Distribution")
        plt.ylabel("Value")
        plt.xlabel("Channel")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()

    else:
        # --- 表格輸出 ---
        print(f"{name} Channel-wise stats (共 {len(data)} 個 channel):")
        header = f"{'ch':<5} {'min':>10} {'q1':>10} {'mean':>10} {'q3':>10} {'max':>10}"
        print(header)
        print("-" * len(header))
        for i, arr in enumerate(data):
            q1, q3 = np.percentile(arr, [25, 75])
            print(
                f"{i:<5} {arr.min():>10.6f} {q1:>10.6f} {arr.mean():>10.6f} {q3:>10.6f} {arr.max():>10.6f}"
            )


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


##---------------------------------------------------------------------------------------------------------------------------
# region print_loss_dict
def print_loss_dict(
    train_loss_dict: Optional[Dict[str, torch.Tensor]] = None,
    eval_loss_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    """
    安全列印訓練與驗證的 loss 值 (小數點後四位)

    參數
    ----
    train_loss_dict : Optional[Dict[str, torch.Tensor]]
        訓練過程中的 loss 字典 (key: loss 名稱, value: loss tensor)
    eval_loss_dict : Optional[Dict[str, torch.Tensor]], default=None
        驗證過程中的 loss 字典 (若為 None 則不輸出)

    回傳
    ----
    None
    """

    def safe_print_dict(loss_dict, title: str):
        if not loss_dict:
            print(f"{title}: 無可用資料 (None 或空字典)")
            return
        print(f"{title}:")
        for name, value in loss_dict.items():
            try:
                if value is None:
                    print(f"  {name}: None", end=" | ")
                elif isinstance(value, torch.Tensor):
                    print(f"  {name}: {value.item():.4f}", end=" | ")
                else:
                    print(f"  {name}: {float(value):.4f}", end=" | ")
            except Exception as e:
                print(f"  {name}: [無法解析: {type(value)}] ({e})", end=" | ")
        print("\n")

    print("\n========== Loss Summary ==========")
    safe_print_dict(train_loss_dict, "Train Losses")
    if eval_loss_dict is not None:
        safe_print_dict(eval_loss_dict, "Eval Losses")
    print("==================================\n")


##---------------------------------------------------------------------------------------------------------------------------
# region sort_pool_by_mse
def sort_pool_by_mse(
    X_pool: torch.Tensor,  # shape = (N, C, H, W)
    Y_pool: torch.Tensor,  # shape = (N, C, H, W)
    channel_start: int = 4,
    channel_end: int = 9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    直接依 X_pool 與 Y_pool 的 MSE 排序 (小到大)

    參數
    ----------
    X_pool : torch.Tensor
        輸入池, shape = (N, C, H, W)
    Y_pool : torch.Tensor
        目標池, shape = (N, C, H, W)
    channel_start : int
        計算 MSE 的起始 channel
    channel_end : int
        計算 MSE 的結束 channel (不包含)

    回傳
    ----------
    X_sorted, Y_sorted : torch.Tensor
        依 MSE 排序後的池
    """
    # 選取指定 channel

    X_sel = X_pool[:, channel_start:channel_end, :, :]
    Y_sel = Y_pool[:, channel_start:channel_end, :, :]

    # 計算每個 sample 的 MSE (展平後對每個 sample mean)
    N = X_sel.shape[0]
    mse_per_sample = ((X_sel - Y_sel) ** 2).view(N, -1).mean(dim=1)

    # 依 MSE 排序
    sorted_idx = torch.argsort(mse_per_sample)  # 小到大

    X_sorted = X_pool[sorted_idx]
    Y_sorted = Y_pool[sorted_idx]

    return X_sorted, Y_sorted


##---------------------------------------------------------------------------------------------------------------------------
# region log_globals
def log_globals(
    scope: dict,
    log_dir: str = "train_log",
    log_file: str = "globals_log.txt",
    exclude_vars: list[str] = None,
) -> None:
    import os

    if exclude_vars is None:
        exclude_vars = ["TRAIN_CASES", "EVAL_CASES", "TEST_CASES", "F", "HTML"]

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    uppercase_vars = [
        name
        for name in scope
        if name.isupper() and name not in exclude_vars and not name.startswith("_")
    ]

    with open(log_path, "w", encoding="utf-8") as f:
        for name in uppercase_vars:
            value = scope[name]
            if hasattr(value, "shape"):
                f.write(f"{name}: shape = {value.shape}\n")
            else:
                f.write(f"{name} = {value}\n")

    print(f"全域變數已寫入 {log_path}")


##---------------------------------------------------------------------------------------------------------------------------
# region check_tensor_nan_inf
import torch


def check_tensor_nan_inf(obj, name="tensor"):
    """
    檢查 tensor 或巢狀結構中是否含有 NaN / Inf，並分開警告

    Parameters
    ----------
    obj : torch.Tensor, list, dict
        要檢查的對象，可以是 tensor 或巢狀結構
    name : str
        名稱，用於打印提示

    Returns
    -------
    has_invalid : bool
        True 表示有 NaN 或 Inf
    """
    has_invalid = False

    if isinstance(obj, torch.Tensor):
        if torch.isnan(obj).any():
            print(f"[Warning] {name} contains NaN")
            print("min:", obj.min().item())
            print("max:", obj.max().item())
            has_invalid = True
        if torch.isinf(obj).any():
            print(f"[Warning] {name} contains Inf")
            print("min:", obj.min().item())
            print("max:", obj.max().item())
            has_invalid = True

    elif isinstance(obj, dict):
        for k, v in obj.items():
            if check_tensor_nan_inf(v, f"{name}.{k}"):
                has_invalid = True

    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            if check_tensor_nan_inf(v, f"{name}[{i}]"):
                has_invalid = True

    else:
        # 非 tensor 的物件忽略
        pass

    return has_invalid


##---------------------------------------------------------------------------------------------------------------------------
# region minmax_scale_channelwise
def minmax_scale_channelwise(x):
    """
    對 BCHW 或 CHW 的資料進行 channel-wise min-max normalization，使每個 channel 都落在 [0,1]。

    Args:
        x : torch.Tensor | np.ndarray
            shape = (C,H,W) 或 (B,C,H,W)

    Returns:
        同型態的 normalized array (值域為 [0,1])
    """
    # ------------------------------------------------------------
    # 統一型態 → numpy
    # ------------------------------------------------------------
    is_torch = isinstance(x, torch.Tensor)
    if is_torch:
        x = x.detach().cpu().numpy()
    elif not isinstance(x, np.ndarray):
        raise TypeError("輸入必須是 torch.Tensor 或 np.ndarray")

    # ------------------------------------------------------------
    # 檢查維度
    # ------------------------------------------------------------
    if x.ndim == 3:  # (C,H,W)
        x = x[None, ...]  # 加一個 batch 維度 → (1,C,H,W)
        squeeze_back = True
    elif x.ndim == 4:  # (B,C,H,W)
        squeeze_back = False
    else:
        raise ValueError(f"不支援 ndim={x.ndim}, 只接受 (C,H,W) 或 (B,C,H,W)")

    B, C, H, W = x.shape
    x_scaled = np.zeros_like(x, dtype=np.float32)

    # ------------------------------------------------------------
    # 每個 channel 獨立縮放
    # ------------------------------------------------------------
    for c in range(C):
        ch_data = x[:, c, :, :]
        ch_min = np.nanmin(ch_data)
        ch_max = np.nanmax(ch_data)
        if np.isclose(ch_max, ch_min):
            # 常數通道 → 全為 0
            x_scaled[:, c, :, :] = 0.0
        else:
            x_scaled[:, c, :, :] = (ch_data - ch_min) / (ch_max - ch_min)

    # ------------------------------------------------------------
    # 移除 batch 維度 (若原本是 CHW)
    # ------------------------------------------------------------
    if squeeze_back:
        x_scaled = x_scaled[0]

    # ------------------------------------------------------------
    # 若原始輸入是 torch.Tensor → 轉回 torch
    # ------------------------------------------------------------
    if is_torch:
        x_scaled = torch.from_numpy(x_scaled)

    return x_scaled


##---------------------------------------------------------------------------------------------------------------------------
# region remove_empty_dirs
def remove_empty_dirs(root_dir: str) -> None:
    """
    遞迴刪除 root_dir 下的所有空資料夾 (沒有檔案，也沒有非空子資料夾)

    Parameters
    ----------
    root_dir : str
        要清理的根目錄
    """
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        # 如果這個資料夾沒有檔案，且底下子資料夾也都被刪光
        if not dirnames and not filenames:
            try:
                os.rmdir(dirpath)
                print(f"Removed empty folder: {dirpath}")
            except OSError as e:
                print(f"Skip {dirpath}, error: {e}")


##---------------------------------------------------------------------------------------------------------------------------
# region timed
import time


def timed(func):
    """
    Decorator: 計算函數執行時間
    """

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"[TIMER] {func.__name__} 執行時間: {end - start:.6f} 秒")
        return result

    return wrapper


##---------------------------------------------------------------------------------------------------------------------------
# region get_rollout_times
def get_rollout_times(epoch, max_epoch, min_n=1, max_n=8, scale=1.0):
    """
    scale > 1 → 增長更慢
    """
    ratio = max_n / min_n
    n = min_n * (ratio ** (epoch / (scale * max_epoch)))
    return min(max_n, int(round(n)))


##---------------------------------------------------------------------------------------------------------------------------
# region view_npz
def view_npz(npz_path_or_obj):
    """
    查看 npz 檔案內部結構與每個 array 的 shape / dtype。

    參數
    ----------
    npz_path_or_obj : str or np.lib.npyio.NpzFile
        npz 檔案路徑，或已經用 np.load() 打開的 npz 物件。
    """
    # 如果輸入是路徑，先 load
    if isinstance(npz_path_or_obj, str):
        data = np.load(npz_path_or_obj)
    else:
        data = npz_path_or_obj

    print("Keys in npz:", list(data.keys()))
    print("-" * 30)

    for key in data.keys():
        arr = data[key]
        print(f"Key: {key}")
        print(f"  Type: {type(arr)}")
        print(f"  Shape: {arr.shape}")
        print(f"  Dtype: {arr.dtype}")
        print("-" * 30)


##---------------------------------------------------------------------------------------------------------------------------
# region resolve_list_options
def resolve_list_options(config: dict, key_paths: list[tuple]) -> dict:
    new_config = json.loads(json.dumps(config))  # 深拷貝乾淨版本

    for path in key_paths:
        d = new_config
        for k in path[:-1]:
            d = d[k]
        last_key = path[-1]
        if isinstance(d.get(last_key), list):
            d[last_key] = random.choice(d[last_key])

    return new_config


##---------------------------------------------------------------------------------------------------------------------------
# region to_device
def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(to_device(v, device) for v in data)
    else:
        return data  # 非 tensor 原樣返回


##---------------------------------------------------------------------------------------------------------------------------
# region cleanup_checkpoints
def cleanup_checkpoints(directory: str, pattern: str, keep_num: int = 3):
    """
    清理指定目錄下的模型檢查點檔案，只保留最新的 N 個。

    Args:
        directory (str): 檢查點所在的目錄路徑。
        pattern (str): 檢查點檔案名稱的正規表達式模式。
                       (例如: 如果您的檔名是 'ca_model_step_100.pth',
                       模式可能像 'ca_model_step_(\d+)\.pth')
        keep_num (int): 欲保留的最新檢查點數量。
    """

    # 組合完整的正規表達式模式
    full_pattern = re.compile(pattern)

    # 儲存 (步驟編號, 完整檔案路徑) 的列表
    found_checkpoints = []

    # 遍歷目錄中的所有檔案
    for filename in os.listdir(directory):
        match = full_pattern.match(filename)
        if match:
            # 假設第一個捕獲組 (\d+) 是步數或週期編號
            step = int(match.group(1))
            file_path = os.path.join(directory, filename)
            found_checkpoints.append((step, file_path))

    # 根據步驟編號（第一個元素）進行排序，從舊到新
    found_checkpoints.sort(key=lambda x: x[0])

    # 計算需要刪除的舊檔案數量
    files_to_delete = len(found_checkpoints) - keep_num

    if files_to_delete > 0:
        # 取得需要刪除的檔案列表 (最舊的 files_to_delete 個)
        for _, path_to_delete in found_checkpoints[:files_to_delete]:
            os.remove(path_to_delete)
            print(f"✅ 已刪除舊檢查點：{path_to_delete}")

    if len(found_checkpoints) > keep_num:
        print(f"🗑️ 目前保留最新的 {keep_num} 個檢查點。")


##---------------------------------------------------------------------------------------------------------------------------
# region save_checkpoint
def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: str,
    cleanup_pattern: Optional[str] = "ca_model_step_(\d+)\.pth",
    keep_num: int = 3,
):
    """
    儲存模型檢查點，並可選地清理舊檔案。
    """
    # 1. 儲存當前檢查點
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    print(f"💾 已儲存新檢查點：{path}")

    # 2. 清理舊檢查點
    if cleanup_pattern:
        # 取得儲存目錄
        directory = os.path.dirname(path)
        if not directory:  # 如果 path 只是檔名，假設目錄是當前目錄
            directory = "."

        cleanup_checkpoints(directory, cleanup_pattern, keep_num)
