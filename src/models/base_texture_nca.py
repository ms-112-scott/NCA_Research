import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple


class BaseTextureNCA(nn.Module):
    """
    基礎紋理神經細胞自動機 (Neural Cellular Automata) 模型。

    此模型透過模擬生物細胞的局部交互作用來生成紋理。
    感知層 (Perception) 提取特徵，隨後通過兩個 1x1 卷積層 (MLP) 計算狀態更新。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 NCA 模型。

        Args:
            config (dict): 設定檔，應包含以下鍵值:
                - chn (int): 通道數量 (State channels)，預設 12。
                - hidden_n (int): 隱藏層神經元數量，預設 96。
                - update_rate (float): 細胞隨機更新的機率，預設 0.5。
        """
        super().__init__()

        # 1. 讀取配置參數
        self.chn = config.get("chn", 12)
        self.hidden_n = config.get("hidden_n", 96)
        self.update_rate = config.get("update_rate", 0.5)

        # 2. 定義神經網路層 (MLP)
        # 輸入維度是 chn * 4，因為感知層會產生 4 種特徵 (Identity, Sobel_X, Sobel_Y, Laplacian)
        self.w1 = nn.Conv2d(self.chn * 4, self.hidden_n, kernel_size=1)
        self.w2 = nn.Conv2d(self.hidden_n, self.chn, kernel_size=1, bias=False)

        # 初始化最後一層為 0，確保訓練初期狀態更新量極小 (類似 Residual Learning 的概念)
        with torch.no_grad():
            self.w2.weight.data.zero_()

        # 3. 預先定義並註冊感知濾波器 (Perception Filters)
        # 這些是固定的卷積核，不需要訓練，因此使用 register_buffer
        ident = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        sobel_y = sobel_x.T  # Sobel Y 是 X 的轉置
        lap = torch.tensor([[1.0, 2.0, 1.0], [2.0, -12.0, 2.0], [1.0, 2.0, 1.0]])

        # 將濾波器堆疊成形狀 [4, 3, 3] 並註冊
        # [0]: Ident, [1]: Sobel X, [2]: Sobel Y, [3]: Laplacian
        filters = torch.stack([ident, sobel_x, sobel_y, lap])
        self.register_buffer("filters", filters)

    def perchannel_conv(self, x: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
        """
        執行深度分離卷積 (Depthwise-like convolution)。
        將同一組濾波器分別應用於輸入的每一個通道上。

        Args:
            x (torch.Tensor): 輸入張量 [Batch, Channel, Height, Width]
            filters (torch.Tensor): 濾波器 [Filter_N, Height, Width]

        Returns:
            torch.Tensor: 卷積結果 [Batch, Channel * Filter_N, Height, Width]
        """
        b, ch, h, w = x.shape

        # 1. Reshape: 將 batch 和 channel 合併，視為單通道圖片處理
        # [B, C, H, W] -> [B*C, 1, H, W]
        y = x.reshape(b * ch, 1, h, w)

        # 2. Padding: 使用循環填充 (Circular Padding) 以生成無縫紋理
        y = F.pad(y, (1, 1, 1, 1), mode="circular")

        # 3. Convolution: 使用 filters 對每個 "單通道圖片" 進行卷積
        # filters[:, None] 將形狀轉為 [4, 1, 3, 3] 以符合 Conv2d 權重格式
        # 輸出形狀: [B*C, 4, H, W]
        y = F.conv2d(y, filters[:, None])

        # 4. Reshape Back: 將 Batch 與 Channel 分離
        # [B*C, 4, H, W] -> [B, C, 4, H, W] -> [B, C*4, H, W]
        y = y.reshape(b, -1, h, w)

        return y

    def perception(self, x: torch.Tensor) -> torch.Tensor:
        """
        感知步驟：提取局部特徵。
        """
        return self.perchannel_conv(x, self.filters)

    def forward(
        self, x: torch.Tensor, update_rate: Optional[float] = None
    ) -> torch.Tensor:
        """
        前向傳播：Perception -> MLP -> Stochastic Update

        Args:
            x (torch.Tensor): 當前狀態 [Batch, Channel, Height, Width]
            update_rate (float, optional): 覆蓋預設的更新率。

        Returns:
            torch.Tensor: 更新後的狀態。
        """
        rate = update_rate if update_rate is not None else self.update_rate

        # 1. 感知 (Perception)
        y = self.perception(x)

        # 2. 處理 (Processing / MLP)
        y = self.w1(y)
        y = torch.relu(y)
        y = self.w2(y)

        # 3. 隨機更新 (Stochastic Update)
        b, c, h, w = y.shape

        # 產生隨機遮罩 (Mask)，模擬細胞的非同步更新
        # rand 值域 [0, 1)。若 rate=0.5:
        # rand < 0.5 (加上 rate 後 < 1.0) -> floor 為 0 -> 不更新
        # rand >= 0.5 (加上 rate 後 >= 1.0) -> floor 為 1 -> 更新
        # 註: 原本邏輯是 (rand + rate).floor()，這意味著 rate 越高，更新機率越高。
        update_mask = (torch.rand(b, 1, h, w, device=x.device) + rate).floor()

        return x + y * update_mask

    def seed(self, n: int, sz: int = 128) -> torch.Tensor:
        """
        產生初始狀態 (全 0 張量)。

        Args:
            n (int): Batch size.
            sz (int): 圖片長寬 (假設為正方形).

        Returns:
            torch.Tensor: 初始狀態張量 [N, Channel, sz, sz]
        """
        # 自動根據模型參數所在的 device 產生 tensor
        return torch.zeros(n, self.chn, sz, sz, device=self.w2.weight.device)
