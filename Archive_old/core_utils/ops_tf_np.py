import torch
import torch.nn.functional as F
import numpy as np
import random
from typing import Tuple, List, Union, Optional


# ================================================================
# region RGBA utility
def to_rgba(x: torch.Tensor) -> torch.Tensor:
    """
    提取輸入張量的前四個 channel 作為 RGBA。

    參數
    ----------
    x : torch.Tensor
        shape = (..., 4+)，最後一維至少要有 RGBA 四個 channel

    回傳
    ----------
    torch.Tensor
        RGBA tensor，shape = (..., 4)
    """
    return x[..., :4]


def to_alpha(x: torch.Tensor) -> torch.Tensor:
    """
    提取 Alpha channel，並限制範圍在 [0, 1]。

    參數
    ----------
    x : torch.Tensor
        shape = (..., 4+)

    回傳
    ----------
    torch.Tensor
        Alpha channel，shape = (..., 1)
    """
    return torch.clamp(x[..., 3:4], 0.0, 1.0)


def to_rgb(x: torch.Tensor) -> torch.Tensor:
    """
    提取 RGB，假設輸入的 RGB 已經是 alpha-premultiplied。

    參數
    ----------
    x : torch.Tensor
        shape = (..., 4+)

    回傳
    ----------
    torch.Tensor
        RGB tensor，shape = (..., 3)
    """
    rgb, a = x[..., :3], to_alpha(x)
    return 1.0 - a + rgb


# ================================================================
# region crop_and_resize
def crop_and_resize(
    inputx: Union[np.ndarray, torch.Tensor],
    target_size: Tuple[int, int],
    crop_size: int = 2
) -> Union[np.ndarray, torch.Tensor]:
    """
    對輸入張量進行裁剪與縮放。

    參數
    ----------
    inputx : Union[np.ndarray, torch.Tensor]
        shape = (B, H, W, C)，batch 影像張量
    target_size : Tuple[int, int]
        目標輸出尺寸 (H_target, W_target)
    crop_size : int, 預設 2
        上下左右裁剪邊界大小

    回傳
    ----------
    Union[np.ndarray, torch.Tensor]
        裁剪與縮放後的張量，與輸入型態相同
    """
    is_numpy = isinstance(inputx, np.ndarray)
    if is_numpy:
        x = torch.from_numpy(inputx).float()
    else:
        x = inputx.float()

    # 假設輸入 shape = (B, H, W, C)，轉換成 PyTorch 預設 (B, C, H, W)
    x = x.permute(0, 3, 1, 2)

    # 裁剪
    x_cropped = x[:, :, crop_size:-crop_size, crop_size:-crop_size]

    # 縮放
    x_resized = F.interpolate(
        x_cropped,
        size=target_size,
        mode="nearest"
    )

    # 再轉回 (B, H, W, C)
    x_resized = x_resized.permute(0, 2, 3, 1)

    return x_resized.numpy() if is_numpy else x_resized


# ================================================================
# region get_random_cfd_slices
def get_random_cfd_slices(
    dynamic_fields: np.ndarray,
    static_fields: np.ndarray,
    num: int = 1
) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    從 CFD 資料中隨機選擇 num 組 (t, z)，並回傳切片。

    參數
    ----------
    dynamic_fields : np.ndarray
        CFD 動態資料，shape = (T, Z, Y, X, C_dyn)
    static_fields : np.ndarray
        CFD 靜態資料，shape = (Z, Y, X, C_static)
    num : int, 預設 1
        要選取的切片數量

    回傳
    ----------
    slices : np.ndarray
        shape = (num, Y, X, C_dyn + C_static)
    t_list : List[int]
        隨機選取的時間索引
    z_list : List[int]
        隨機選取的高度索引
    """
    T, Z, Y, X, C_dyn = dynamic_fields.shape
    _, _, _, C_static = static_fields.shape

    slices = np.zeros((num, Y, X, C_dyn + C_static), dtype=dynamic_fields.dtype)
    t_list, z_list = [], []

    for i in range(num):
        t = random.randint(0, T - 1)
        z = random.randint(0, Z - 1)
        dyn = dynamic_fields[t, z, :, :, :]  # (Y, X, C_dyn)
        sta = static_fields[z, :, :, :]      # (Y, X, C_static)
        slices[i] = np.concatenate((sta, dyn), axis=-1)

        t_list.append(t)
        z_list.append(z)

    return slices, t_list, z_list


# ================================================================
# region get_random_cfd_slices_pair
def get_random_cfd_slices_pair(
    dynamic_fields: torch.Tensor,
    static_fields: torch.Tensor,
    slice_count: int = 1,
    xy_size: Optional[Tuple[int, int]] = None,
    output_meta: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, Union[List[dict], torch.Tensor]]:
    """
    向量化版本的 CFD 切片生成函數，可用於資料增強與強化學習。

    參數
    ----------
    dynamic_fields : torch.Tensor
        CFD 動態資料，shape = (T, Z, Y, X, C_dyn)
    static_fields : torch.Tensor
        CFD 靜態資料，shape = (Z, Y, X, C_static)
    slice_count : int, 預設 1
        要取的切片數
    xy_size : Optional[Tuple[int, int]], 預設 None
        若指定，輸出裁剪大小 (h, w)，否則取原始大小
    output_meta : bool, 預設 False
        是否輸出完整 metadata

    回傳
    ----------
    t_slices : torch.Tensor
        shape = (N, h, w, C_total)，時間 t 的切片
    tn_slices : torch.Tensor
        shape = (N, h, w, C_total)，時間 tn 的切片
    metas : List[dict] 或 torch.Tensor
        若 output_meta=True，回傳 list of dict，否則回傳 Δt tensor
    """
    T, Z, Y, X, C_dyn = dynamic_fields.shape
    _, _, _, C_static = static_fields.shape

    h = Y if xy_size is None else xy_size[0]
    w = X if xy_size is None else xy_size[1]

    t_slices, tn_slices, metas = [], [], []

    for _ in range(slice_count):
        # 隨機取樣 t, tn, z
        t = random.randint(0, T - 2)       # 保證有 tn > t
        tn = random.randint(t + 1, T - 1)
        z = random.randint(0, Z - 1)
        sy = 0 if h == Y else random.randint(0, Y - h)
        sx = 0 if w == X else random.randint(0, X - w)

        # 擷取資料
        dyn_t = dynamic_fields[t, z, sy:sy+h, sx:sx+w, :]   # (h, w, C_dyn)
        dyn_tn = dynamic_fields[tn, z, sy:sy+h, sx:sx+w, :] # (h, w, C_dyn)
        sta = static_fields[z, sy:sy+h, sx:sx+w, :]         # (h, w, C_static)

        t_slice = torch.cat([sta, dyn_t], dim=-1)
        tn_slice = torch.cat([sta, dyn_tn], dim=-1)

        t_slices.append(t_slice)
        tn_slices.append(tn_slice)

        if output_meta:
            metas.append({"t": t, "tn": tn, "z": z, "sy": sy, "sx": sx})
        else:
            metas.append(tn - t)

    t_slices = torch.stack(t_slices, dim=0)
    tn_slices = torch.stack(tn_slices, dim=0)

    if output_meta:
        return t_slices, tn_slices, metas
    else:
        return t_slices, tn_slices, torch.tensor(metas, dtype=torch.int32)
