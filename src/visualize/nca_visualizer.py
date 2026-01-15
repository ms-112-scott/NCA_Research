import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Callable
from IPython.display import clear_output, display


class NCAVisualizer:
    """
    NCA 訓練過程視覺化工具。
    處理 Loss 曲線繪製與即時圖片展示。
    """

    def __init__(self, to_rgb_fn: Callable[[torch.Tensor], np.ndarray] = None):
        """
        Args:
            to_rgb_fn: 將 Tensor [B, C, H, W] 轉為可視化 numpy array [B, H, W, 3] 的函式。
                       若為 None，則使用預設轉換。
        """
        self.to_rgb_fn = to_rgb_fn if to_rgb_fn else self._default_to_rgb

    def _default_to_rgb(self, x: torch.Tensor) -> np.ndarray:
        """預設轉換：取前3通道 -> Permute -> CPU -> Numpy"""
        # 假設 x 是 [B, C, H, W]
        rgb = x[:, :3, :, :]
        # 轉為 [B, H, W, C]
        rgb = rgb.permute(0, 2, 3, 1)
        return rgb.cpu().numpy()

    def show_training_state(
        self,
        loss_log: List[float],
        batch_x: torch.Tensor,
        current_step: int,
        lr: float,
        clear_previous: bool = True,
    ):
        """
        綜合顯示函式：畫 Loss 曲線並顯示當前 Batch 的圖片。
        """
        if clear_previous:
            clear_output(wait=True)

        # 1. 顯示文字統計
        print(f"Step: {current_step} | Loss: {loss_log[-1]:.6f} | LR: {lr:.6f}")

        # 2. 建立畫布 (左邊是 Loss，右邊是圖片)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # --- Plot Loss ---
        ax_loss = axes[0]
        ax_loss.plot(loss_log, ".", alpha=0.1, label="Step Loss")
        # 平滑曲線 (Optional)
        if len(loss_log) > 10:
            smooth_loss = np.convolve(loss_log, np.ones(10) / 10, mode="valid")
            ax_loss.plot(
                range(9, len(loss_log)), smooth_loss, "k-", alpha=0.5, label="Smoothed"
            )

        ax_loss.set_yscale("log")
        if len(loss_log) > 0:
            ax_loss.set_ylim(min(loss_log), max(loss_log[0], max(loss_log)))
        ax_loss.set_title("Training Loss (Log Scale)")
        ax_loss.set_xlabel("Steps")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend()

        # --- Plot Batch Images ---
        ax_img = axes[1]
        imgs_np = self.to_rgb_fn(batch_x)
        # 將 batch 中的圖片水平拼接
        grid_img = np.hstack(imgs_np)

        # 數值截斷到 [0, 1] 以正確顯示
        grid_img = np.clip(grid_img, 0.0, 1.0)

        ax_img.imshow(grid_img)
        ax_img.set_title(f"Batch @ Step {current_step}")
        ax_img.axis("off")

        plt.tight_layout()
        plt.show()
