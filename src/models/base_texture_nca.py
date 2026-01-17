import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple


class BaseTextureNCA(nn.Module):
    """
    基礎紋理神經細胞自動機 (Neural Cellular Automata) 模型。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 NCA 模型。
        """
        super().__init__()

        # 1. 讀取配置參數
        self.chn = config.get("chn", 12)
        self.hidden_n = config.get("hidden_n", 96)
        self.update_rate = config.get("update_rate", 0.5)

        # [修改點 1] 記住 config 中的圖片大小，若無則預設 128
        self.img_size = config.get("img_size", 128)

        # 2. 定義神經網路層 (MLP)
        self.w1 = nn.Conv2d(self.chn * 4, self.hidden_n, kernel_size=1)
        self.w2 = nn.Conv2d(self.hidden_n, self.chn, kernel_size=1, bias=False)

        # 初始化最後一層為 0
        with torch.no_grad():
            self.w2.weight.data.zero_()

        # 3. 預先定義感知濾波器
        ident = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        sobel_y = sobel_x.T
        lap = torch.tensor([[1.0, 2.0, 1.0], [2.0, -12.0, 2.0], [1.0, 2.0, 1.0]])

        filters = torch.stack([ident, sobel_x, sobel_y, lap])
        self.register_buffer("filters", filters)

    def perchannel_conv(self, x: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
        b, ch, h, w = x.shape
        y = x.reshape(b * ch, 1, h, w)
        y = F.pad(y, (1, 1, 1, 1), mode="circular")
        y = F.conv2d(y, filters[:, None])
        y = y.reshape(b, -1, h, w)
        return y

    def perception(self, x: torch.Tensor) -> torch.Tensor:
        return self.perchannel_conv(x, self.filters)

    def forward(
        self, x: torch.Tensor, update_rate: Optional[float] = None
    ) -> torch.Tensor:
        rate = update_rate if update_rate is not None else self.update_rate

        y = self.perception(x)
        y = self.w1(y)
        y = torch.relu(y)
        y = self.w2(y)

        b, c, h, w = y.shape
        update_mask = (torch.rand(b, 1, h, w, device=x.device) + rate).floor()

        return x + y * update_mask

    def seed(self, n: int, sz: Optional[int] = None) -> torch.Tensor:
        """
        產生初始狀態 (全 0 張量)。

        [修改點 2] 若未指定 sz，則自動使用模型初始化時的 img_size
        """
        if sz is None:
            sz = self.img_size

        return torch.zeros(n, self.chn, sz, sz, device=self.w2.weight.device)
