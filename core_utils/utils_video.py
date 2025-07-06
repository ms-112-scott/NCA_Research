import os
import numpy as np

# 設定 moviepy 使用本地 ffmpeg
os.environ['FFMPEG_BINARY'] = 'ffmpeg'

from moviepy.editor import ImageSequenceClip
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

def save_video(image_sequence, out_path, fps=15):
    """
    儲存一段圖像序列為 MP4 影片（高階介面）

    Args:
        image_sequence: List of np.ndarray, shape [H, W, 3] or [H, W]
        out_path: 輸出檔案路徑（建議使用 .mp4）
        fps: 幀率（預設 15）
    """
    clip = ImageSequenceClip(image_sequence, fps=fps)
    clip.write_videofile(out_path, codec='libx264')


class VideoWriter:
    """
    可連續寫入幀畫面的影片儲存器（低階介面，支援 context manager）

    Usage:
    ```python
    with VideoWriter("out.mp4", fps=30) as writer:
        for frame in frames:
            writer.add(frame)
    ```
    """

    def __init__(self, filename, fps=30.0, **kwargs):
        self.writer = None
        self.params = dict(filename=filename, fps=fps, **kwargs)

    def add(self, img):
        """
        新增一張幀畫面（img 可為 float 或 uint8）
        """
        img = np.asarray(img)
        if img.dtype in [np.float32, np.float64]:
            img = np.uint8(np.clip(img, 0, 1) * 255)
        if len(img.shape) == 2:  # 灰階轉 RGB
            img = np.repeat(img[..., None], 3, -1)
        if self.writer is None:
            h, w = img.shape[:2]
            self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
        self.writer.write_frame(img)

    def close(self):
        """手動關閉影片寫入器"""
        if self.writer:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
