import torch
import torch.nn as nn


class MSELoss(nn.Module):
    """
    包裝 MSE Loss，使其介面與 VGGLoss 一致。
    初始化時固定 target，forward 時只需傳入 input。
    """

    def __init__(self, target_img: torch.Tensor):
        super().__init__()
        # 註冊 target 為 buffer (不會被視為模型參數，但會隨模型移動 device)
        self.register_buffer("target", target_img.detach())
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor):
        # 如果輸入 x 的 batch size 比 target 大，我們需要擴展 target
        # x: [B, C, H, W], target: [1, C, H, W]
        target_expanded = self.target.expand_as(x)
        return self.loss_fn(x, target_expanded)
