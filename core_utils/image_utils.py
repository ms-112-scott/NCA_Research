import numpy as np
from IPython.display import Image, display

from .io_utils import imencode

def imshow(arr, fmt='jpeg'):
    """在 notebook 中顯示圖像（支援 jpeg/png）"""
    display(Image(data=imencode(arr, fmt)))

def tile2d(arr, w=None):
    """
    將一組圖像以網格方式拼接成一張大圖
    arr: shape [N, H, W, C]
    w: 每行要擺幾張圖，預設為 sqrt(N)
    """
    arr = np.asarray(arr)
    if w is None:
        w = int(np.ceil(np.sqrt(len(arr))))
    th, tw = arr.shape[1:3]
    pad = (w - len(arr)) % w
    arr = np.pad(arr, [(0, pad)] + [(0, 0)] * (arr.ndim - 1), mode='constant')
    h = len(arr) // w
    arr = arr.reshape([h, w] + list(arr.shape[1:]))
    arr = np.rollaxis(arr, 2, 1).reshape([th * h, tw * w] + list(arr.shape[4:]))
    return arr

def zoom(img, scale=4):
    """將圖像放大 scale 倍（不插值，只是像素重複）"""
    img = np.repeat(img, scale, axis=0)
    img = np.repeat(img, scale, axis=1)
    return img
