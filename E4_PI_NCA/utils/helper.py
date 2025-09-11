import numpy as np
import torch
from typing import Union, List, Tuple
from scipy.ndimage import generic_filter

#===========================================================================================
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
            return np.transpose(arr, (1,2,0))
        elif arr.ndim == 4:
            return np.transpose(arr, (0,2,3,1))
        else:
            raise ValueError(f"輸入 np.ndarray 維度 {arr.ndim} 不支援, 只支援 3D 或 4D")
    elif isinstance(arr, torch.Tensor):
        if arr.ndim == 3:
            return arr.permute(1,2,0)
        elif arr.ndim == 4:
            return arr.permute(0,2,3,1)
        else:
            raise ValueError(f"輸入 torch.Tensor 維度 {arr.ndim} 不支援, 只支援 3D 或 4D")
    else:
        raise TypeError(f"輸入類型 {type(arr)} 不支援, 只支援 np.ndarray 或 torch.Tensor")


#===========================================================================================
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
            print(f"  channel {c}: min={x[c].min().item():.6f}, max={x[c].max().item():.6f}")
    elif x.ndim == 4:
        B, C, H, W = x.shape
        print(f"{name} (B,C,H,W) shape = {x.shape}")
        for c in range(C):
            ch_min = x[:,c,:,:].min().item()
            ch_max = x[:,c,:,:].max().item()
            print(f"  channel {c}: min={ch_min:.6f}, max={ch_max:.6f}")
    else:
        raise ValueError(f"{name} 不支援 ndim={x.ndim}, 只支援 3D 或 4D tensor")


#===========================================================================================
# region split_cases
def split_cases(
    case_list: List[torch.Tensor],
    train_ratio: float = 0.7,
    eval_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42
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
    eval_cases = [case_list[i] for i in indices[n_train:n_train+n_eval]]
    test_cases = [case_list[i] for i in indices[n_train+n_eval:]]

    return train_cases, eval_cases, test_cases

