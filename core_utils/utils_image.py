import numpy as np
from IPython.display import Image, display
import torch

from .utils_io import imencode

def imshow(arr, fmt='jpeg'):
    """在 notebook 中顯示圖像（支援 jpeg/png）"""
    display(Image(data=imencode(arr, fmt)))


def tile2d(images: torch.Tensor, n: int = 7) -> np.ndarray:
    """
    將前 n*n 張圖拼貼成一張大圖。

    Args:
        images (torch.Tensor): (B, H, W, C) 格式
        n (int): 拼貼行列數 (預設 7)

    Returns:
        np.ndarray: 拼貼後影像 (H*n, W*n, C)，值範圍保持原本 tensor 範圍
    """
    assert images.ndim == 4, f"Expect BHWC tensor, got {images.shape}"
    B, H, W, C = images.shape
    images = images[: n*n]

    # Torch -> Numpy
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()

    # Reshape + Transpose 拼貼
    images = images.reshape(n, n, H, W, C)          # (n, n, H, W, C)
    images = images.transpose(0, 2, 1, 3, 4)        # (n, H, n, W, C)
    tiled = images.reshape(n*H, n*W, C)             # (H*n, W*n, C)
    return tiled

def zoom(img, scale=4):
    """將圖像放大 scale 倍（不插值，只是像素重複）"""
    img = np.repeat(img, scale, axis=0)
    img = np.repeat(img, scale, axis=1)
    return img
