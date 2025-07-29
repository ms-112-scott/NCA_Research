import os
import numpy as np
import cv2

def save_video(image_sequence, out_path, fps=15):
    """
    儲存一段圖像序列為 MP4 影片（高階介面）

    Args:
        image_sequence: List of np.ndarray, shape [H, W, 3] or [H, W]
        out_path: 輸出檔案路徑（建議使用 .mp4）
        fps: 幀率（預設 15）
    """
    if len(image_sequence) == 0:
        raise ValueError("image_sequence is empty")

    first_frame = image_sequence[0]
    if len(first_frame.shape) == 2:  # 灰階轉三通道
        height, width = first_frame.shape
        is_color = False
    else:
        height, width, _ = first_frame.shape
        is_color = True

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height), isColor=is_color)

    for img in image_sequence:
        img = np.asarray(img)
        if img.dtype in [np.float32, np.float64]:
            img = np.clip(img, 0, 1)
            img = (img * 255).astype(np.uint8)
        if len(img.shape) == 2 and is_color:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and not is_color:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        writer.write(img)

    writer.release()

class VideoWriter:
    """
    低階介面影片寫入器（支援 context manager）

    Usage:
    ```python
    with VideoWriter("out.mp4", fps=30) as writer:
        for frame in frames:
            writer.add(frame)
    ```
    """

    def __init__(self, filename, fps=30.0):
        self.filename = filename
        self.fps = fps
        self.writer = None
        self.size = None
        self.is_color = True

    def add(self, img):
        img = np.asarray(img)
        if img.dtype in [np.float32, np.float64]:
            img = np.clip(img, 0, 1)
            img = (img * 255).astype(np.uint8)
        if len(img.shape) == 2:  # 灰階
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if self.writer is None:
            h, w = img.shape[:2]
            self.size = (w, h)
            self.writer = cv2.VideoWriter(
                self.filename,
                cv2.VideoWriter_fourcc(*'mp4v'),
                self.fps,
                self.size,
                isColor=True
            )
        self.writer.write(img)

    def close(self):
        if self.writer:
            self.writer.release()
            self.writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
