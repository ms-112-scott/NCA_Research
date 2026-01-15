import torch
from typing import Union, List
import numpy as np
import random
import torchvision.transforms.functional as TF
from PIL import Image


def set_seed(seed: int):
    """設定所有隨機套件的種子以確保可重現性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 為了效能通常不設 deterministic，若需絕對重現可開啟
        # torch.backends.cudnn.deterministic = True


def load_target_image(path: str, size: int = 128, device="cpu"):
    """讀取並預處理目標圖片"""
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    img_tensor = TF.to_tensor(img).unsqueeze(0)  # [1, 3, H, W]
    return img_tensor.to(device)


def to_nchw(img: Union[torch.Tensor, list]) -> torch.Tensor:
    """
    輔助函式：確保圖片格式為 [Batch, Channel, Height, Width]。

    Args:
        img (torch.Tensor or list): 輸入的圖片張量或列表。

    Returns:
        torch.Tensor: 格式化後的 NCHW 張量。
    """
    img_tensor = torch.as_tensor(img)

    # 若輸入為 [C, H, W] -> 增加 Batch 維度變 [1, C, H, W]
    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)

    # 若輸入為 [B, H, W, C] (常見於 image io 讀取) -> 轉置為 [B, C, H, W]
    # 這裡假設若最後一維是 3，則為 Channel (簡易判斷，可根據需求調整)
    if img_tensor.ndim == 4 and img_tensor.shape[3] == 3:
        img_tensor = img_tensor.permute(0, 3, 1, 2)

    return img_tensor


def to_rgb(x):
    return x[..., :3, :, :] + 0.5
