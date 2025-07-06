import io
import base64
import numpy as np
from PIL import Image as PILImage
import requests

def np2pil(arr):
    """將 numpy 陣列轉為 PIL Image"""
    if arr.dtype in [np.float32, np.float64]:
        arr = np.uint8(np.clip(arr, 0, 1) * 255)
    return PILImage.fromarray(arr)

def imwrite(file, arr, fmt=None):
    """將圖像儲存為檔案（自動推斷格式）"""
    arr = np.asarray(arr)
    if isinstance(file, str):
        fmt = file.rsplit('.', 1)[-1].lower()
        if fmt == 'jpg':
            fmt = 'jpeg'
        file = open(file, 'wb')
    np2pil(arr).save(file, fmt, quality=95)

def imencode(arr, fmt='jpeg'):
    """將圖像編碼為 bytes，用於 base64 或網頁輸出"""
    arr = np.asarray(arr)
    if len(arr.shape) == 3 and arr.shape[-1] == 4:
        fmt = 'png'
    f = io.BytesIO()
    imwrite(f, arr, fmt)
    return f.getvalue()

def im2url(arr, fmt='jpeg'):
    """將圖像轉為 base64 url，可直接插入 HTML"""
    encoded = imencode(arr, fmt)
    base64_str = base64.b64encode(encoded).decode('ascii')
    return f'data:image/{fmt.upper()};base64,{base64_str}'


def load_image(url, max_size=128):
    """
    從遠端 URL 載入圖像，轉為 premultiplied alpha 的 numpy array

    Args:
        url: 圖像 URL
        max_size: 最大邊長（圖像會等比例縮小）

    Returns:
        np.ndarray: shape [H, W, 4] 的 float32 圖像，值範圍 [0, 1]
    """
    response = requests.get(url)
    img = PILImage.open(io.BytesIO(response.content)).convert("RGBA")
    img.thumbnail((max_size, max_size), PILImage.LANCZOS)
    img = np.float32(img) / 255.0
    # Premultiply RGB by Alpha
    img[..., :3] *= img[..., 3:]
    return img

def load_emoji(emoji_char, max_size=128):
    """
    載入指定 emoji 對應的圖像（從 Google Noto Emoji GitHub）
    """
    code = hex(ord(emoji_char))[2:].lower()
    url = f'https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u{code}.png?raw=true'
    return load_image(url, max_size=max_size)