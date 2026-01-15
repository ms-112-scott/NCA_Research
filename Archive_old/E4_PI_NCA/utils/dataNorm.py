import torch


# =========================================================
# region calc_channel_ref_PT
def calc_channel_ref_PT(tensor: torch.Tensor, channel_names: list[str]) -> torch.Tensor:
    """
    根據 channel 名稱計算每個通道的無因次化參考值 ref
    適用於 PyTorch Tensor，回傳 shape = (C,) 的 ref tensor

    參數：
        tensor : torch.Tensor
            CFD dataset, shape = (B, C, H, W)
        channel_names : list[str]
            每個通道名稱，需與 tensor channel 對應順序一致

    回傳：
        ref : torch.Tensor
            每個 channel 的無因次化參考值，shape = (C,)
    """

    # --- 基本檢查 ---
    assert tensor.ndim == 4, "輸入 tensor 應為 (B, C, H, W)"
    B, C, H, W = tensor.shape
    assert len(channel_names) == C, "channel_names 數量需與 tensor channel 相同"

    # --- 計算通道標準差 (全域統計) ---
    # keepdim=False → 得到 (C,)
    channel_std = tensor.std(dim=(0, 2, 3), unbiased=False)
    ref = torch.ones(C, dtype=tensor.dtype, device=tensor.device)

    # --- 建立對應規則 ---
    for i, ch in enumerate(channel_names):
        # 幾何或遮罩通道，不縮放
        if ch in ["coord_y", "coord_x", "geo_mask"]:
            ref[i] = 1.0

        # 地形高度，以自身 std 為基準 (m)
        elif ch == "topo":
            ref[i] = channel_std[i]

        # 初始風場，不縮放
        elif ch in ["windInitX", "windInitY"]:
            ref[i] = 1.0

        # 速度分量，以 Uped std 為參考 (m/s)
        elif ch in ["uped", "vped", "Uped"]:
            uped_idx = channel_names.index("Uped")
            ref[i] = channel_std[uped_idx]

        # 能量相關，以 (Uped std)^2 為參考 (m^2/s^2)
        elif ch in ["TKEped", "Tuwped"]:
            uped_idx = channel_names.index("Uped")
            ref[i] = channel_std[uped_idx] ** 2

        # 未定義類型 → 預設不縮放
        else:
            ref[i] = 1.0

    return ref


# =========================================================
# region 無因次化（Non-dimensionalization by reference scale）
def nondimensionalize(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    依據每個 channel 的參考尺度 ref 進行無因次化：
        x* = x / ref

    參數:
    ----------
    x : torch.Tensor
        原始資料 (B, C, H, W)
    ref : torch.Tensor
        各 channel 的參考值 (C,)

    回傳:
    ----------
    x_nd : torch.Tensor
        無因次化後資料
    """
    # 確保 ref 不為零
    ref_adj = ref.clone()
    ref_adj[ref_adj == 0] = 1.0

    # 調整 shape 為 (1, C, 1, 1) 以便廣播
    ref_adj = ref_adj.view(1, -1, 1, 1)

    x_nd = x / ref_adj
    return x_nd


def inv_nondimensionalize(x_nd: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    無因次化反運算：
        x = x* * ref
    """
    ref = ref.view(1, -1, 1, 1)
    x = x_nd * ref
    return x


# endregion


# =========================================================
# region 標準化（Standardization）


def standardize(x: torch.Tensor, mean: torch.Tensor = None, std: torch.Tensor = None):
    """
    對每個 channel 進行 Z-score 標準化：
        z = (x - mean) / std

    如果 mean/std 未提供，則自動從 tensor 計算全 dataset 全域 mean/std。

    參數：
    ----------
    x : torch.Tensor
        原始資料 (B, C, H, W)
    mean : torch.Tensor, optional
        每個 channel 的平均值 (C,)
    std : torch.Tensor, optional
        每個 channel 的標準差 (C,)

    回傳：
    ----------
    z : torch.Tensor
        標準化後資料
    mean : torch.Tensor
        每個 channel 的平均值
    std : torch.Tensor
        每個 channel 的標準差
    """
    # 自動計算 mean/std
    if mean is None:
        mean = x.mean(dim=(0, 2, 3))
    if std is None:
        std = x.std(dim=(0, 2, 3), unbiased=False)

    std_adj = std.clone()
    std_adj[std_adj == 0] = 1.0

    z = (x - mean.view(1, -1, 1, 1)) / std_adj.view(1, -1, 1, 1)
    return z, mean, std_adj


def inv_standardize(z: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    """
    標準化反運算：
        x = z * std + mean
    """
    x = z * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
    return x


# endregion


# =========================================================
# region Class: NormalizeWrapper


class NormalizeWrapper:
    """
    整合「無因次化」與「標準化」的雙層處理流程，
    支援正向 (forward) 與反向 (inverse) 操作，
    並自動依 channel_names 計算 ref。
    """

    def __init__(self, channel_names: list[str]):
        """
        參數:
        ----------
        channel_names : list[str]
            每個通道名稱，需與 tensor channel 對應順序一致
        """
        self.channel_names = channel_names
        self.ref: torch.Tensor = None
        self.mean: torch.Tensor = None
        self.std: torch.Tensor = None

    def forward(self, x: torch.Tensor, steps=("nondim", "std")) -> torch.Tensor:
        """
        正向處理: 原始物理量 -> 模型輸入
        """
        # 計算每個 channel 的 ref
        self.ref = calc_channel_ref_PT(x, self.channel_names)

        # 無因次化
        if "nondim" in steps:
            x = nondimensionalize(x, self.ref)

        # 標準化
        if "std" in steps:
            if self.mean is None or self.std is None:
                x, mean, std = standardize(x)
                self.mean = mean
                self.std = std
            else:
                x, _, _ = standardize(x, self.mean, self.std)

        return x

    def inverse(self, x: torch.Tensor, steps=("nondim", "std")) -> torch.Tensor:
        """
        反向處理: 模型輸出 -> 還原物理量
        """
        # 標準化反運算
        if "std" in steps:
            if self.mean is None or self.std is None:
                raise ValueError("mean/std 未提供，無法執行 inverse standardization")
            x = inv_standardize(x, self.mean, self.std)

        # 無因次化反運算
        if "nondim" in steps:
            if self.ref is None:
                raise ValueError("ref 未提供，無法執行 inverse nondimensionalization")
            x = inv_nondimensionalize(x, self.ref)

        return x


# endregion
