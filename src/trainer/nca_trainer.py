import torch
import numpy as np
import torch.nn as nn
from typing import Callable, Dict, Any, Optional


class NCATrainer:
    """
    Neural Cellular Automata (NCA) 訓練器。
    負責核心訓練迴圈、樣本池管理與梯度更新。
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        pool: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        to_rgb_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: x[:, :3, ...],
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model: NCA 模型實例。
            config: 設定字典，包含訓練參數 (batch_size, step_n 等)。
            pool: 樣本池 Tensor [Pool_Size, C, H, W]。
            optimizer: 優化器。
            scheduler: 學習率排程器。
            loss_fn: 損失函數。
            to_rgb_fn: 將隱藏狀態轉為 RGB 的函式。
            device: 運算設備。
        """
        self.model = model
        self.config = config
        self.pool = pool
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.to_rgb_fn = to_rgb_fn

        self.device = (
            device
            if device
            else (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        )

        # 從 config 解析參數，並設定安全預設值
        self.batch_size = config.get("batch_size", 8)
        self.step_min = config.get("step_min", 32)
        self.step_max = config.get("step_max", 96)
        self.pool_reset_freq = config.get("pool_reset_freq", 8)
        self.use_gradient_checkpoint = config.get("use_gradient_checkpoint", False)

    def train_step(self, current_step: int) -> Dict[str, Any]:
        """
        執行單步訓練。

        Returns:
            Dict: 包含 'loss' (float), 'batch_x' (Tensor), 'lr' (float) 的字典。
        """
        # 1. 準備 Batch (Sample Pool Strategy)
        with torch.no_grad():
            # 隨機從 Pool 挑選索引
            pool_indices = np.random.choice(
                len(self.pool), self.batch_size, replace=False
            )
            x = self.pool[pool_indices].to(self.device)

            # Seed Injection: 防止模型遺忘種子狀態
            if current_step % self.pool_reset_freq == 0:
                x[:1] = self.model.seed(1)

        # 2. 隨機決定迭代次數 (Stochastic Depth)
        step_n = np.random.randint(self.step_min, self.step_max)

        # 3. 前向傳播 (Forward Pass)
        if self.use_gradient_checkpoint:
            x.requires_grad = True
            # 使用 checkpoint 節省顯存
            x = torch.utils.checkpoint.checkpoint_sequential(
                [self.model] * step_n, segments=min(step_n, 16), input=x
            )
        else:
            for _ in range(step_n):
                x = self.model(x)

        # 4. 計算損失
        rgb_img = self.to_rgb_fn(x)

        # Overflow Loss: 懲罰數值過大，保持數值穩定
        overflow_loss = (x - x.clamp(-1.0, 1.0)).abs().sum()
        loss = self.loss_fn(rgb_img) + overflow_loss

        # 5. 反向傳播與優化
        self.optimizer.zero_grad()
        loss.backward()

        # NCA 特有技巧：梯度正規化 (Gradient Normalization)
        with torch.no_grad():
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad /= p.grad.norm() + 1e-8

            self.optimizer.step()
            self.scheduler.step()

            # 6. 更新 Sample Pool (切斷梯度)
            self.pool[pool_indices] = x.detach().to(self.pool.device)

        return {
            "loss": loss.item(),
            "batch_x": x.detach(),  # 回傳 detach 的 tensor 供視覺化使用
            "lr": self.scheduler.get_last_lr()[0],
        }
