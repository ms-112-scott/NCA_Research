import numpy as np
from IPython.display import Image, display
import torch

from .utils_io import imencode


def imshow(arr, fmt="jpeg"):
    """在 notebook 中顯示圖像（支援 jpeg/png）"""
    display(Image(data=imencode(arr, fmt)))


import torch
import numpy as np


def tile2d(images: torch.Tensor, n: int = 7, pad: int = 2) -> np.ndarray:
    """
    將前 n*n 張圖拼貼成一張大圖，中間加入像素空白間隔。

    Args:
        images (torch.Tensor): (B, H, W, C) 格式
        n (int): 拼貼行列數 (預設 7)
        pad (int): 每張圖之間的空白像素數

    Returns:
        np.ndarray: 拼貼後影像 (H*n + pad*(n-1), W*n + pad*(n-1), C)
    """
    assert images.ndim == 4, f"Expect BHWC tensor, got {images.shape}"
    B, H, W, C = images.shape
    images = images[: n * n]

    # Torch -> Numpy
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()

    # 建立空白 canvas
    tiled_H = n * H + pad * (n - 1)
    tiled_W = n * W + pad * (n - 1)
    tiled = np.ones((tiled_H, tiled_W, C), dtype=images.dtype)  # 以 1 填充空白區域

    for idx, img in enumerate(images):
        row = idx // n
        col = idx % n
        start_h = row * (H + pad)
        start_w = col * (W + pad)
        tiled[start_h : start_h + H, start_w : start_w + W, :] = img

    return tiled


def zoom(img, scale=4):
    """將圖像放大 scale 倍（不插值，只是像素重複）"""
    img = np.repeat(img, scale, axis=0)
    img = np.repeat(img, scale, axis=1)
    return img
