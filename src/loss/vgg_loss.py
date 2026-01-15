import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List, Optional, Union, Tuple
from src.help.utils import to_nchw

__all__ = ["VGGLoss", "ot_loss", "create_vgg_loss"]


def project_sort(x: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
    """
    輔助函式：將特徵投影到隨機向量上並進行排序 (用於 Sliced Wasserstein / OT Loss)。

    Args:
        x (torch.Tensor): 輸入特徵 [Batch, Channel, N_pixels]
        proj (torch.Tensor): 投影矩陣 [Channel, Proj_N]

    Returns:
        torch.Tensor: 投影並排序後的特徵 [Batch, Proj_N, N_pixels]
    """
    # Einstein summation: 投影特徵
    # x shape: (b, c, n), proj shape: (c, p) -> result: (b, p, n)
    projected = torch.einsum("bcn,cp->bpn", x, proj)

    # 對最後一個維度 (N_pixels) 進行排序，取 values [0]
    return projected.sort(dim=-1)[0]


def ot_loss(
    source: torch.Tensor, target: torch.Tensor, proj_n: int = 32
) -> torch.Tensor:
    """
    計算 Sliced Optimal Transport Loss (Sliced Wasserstein Distance)。
    通過隨機投影將高維分佈距離轉化為一維排序後的距離。

    Args:
        source (torch.Tensor): 來源特徵 [Batch, Channel, N]
        target (torch.Tensor): 目標特徵 [Batch, Channel, M] (M 可與 N 不同)
        proj_n (int): 隨機投影的方向數量。預設為 32。

    Returns:
        torch.Tensor: 計算出的 Loss 純量。
    """
    ch, n = source.shape[-2:]

    # 建立隨機投影矩陣，並正規化 (Normalize)
    # 確保投影矩陣在正確的 device 上
    projs = F.normalize(torch.randn(ch, proj_n, device=source.device), dim=0)

    # 投影並排序
    source_proj = project_sort(source, projs)
    target_proj = project_sort(target, projs)

    # 若 source 與 target 的像素數量 (N vs M) 不同，需對 target 進行插值以匹配 source
    if source_proj.shape[-1] != target_proj.shape[-1]:
        target_proj = F.interpolate(target_proj, size=n, mode="nearest")

    # 計算均方誤差 (MSE)
    return (source_proj - target_proj).square().sum()


class VGGLoss(nn.Module):
    """
    基於 VGG16 的感知損失 (Perceptual Loss) 計算模組，
    結合了 Sliced Optimal Transport Loss 來衡量風格/紋理差異。
    """

    def __init__(
        self,
        target_img: Union[torch.Tensor, list],
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            target_img (torch.Tensor): 目標圖片 (Style Image)。
            device (torch.device, optional): 指定運算設備。若為 None 則跟隨 target_img 或預設。
        """
        super().__init__()

        # 1. 設定 VGG16 特徵提取器
        # 載入 ImageNet 預訓練權重
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features

        # 定義要提取特徵的層索引 (這些通常對應於 conv1_1, conv2_1 等)
        self.style_layers_idx = [1, 6, 11, 18, 25]

        # 為了節省記憶體與運算，截斷模型至最後一個需要的層
        self.model = vgg[: max(self.style_layers_idx) + 1]

        # 凍結參數 (不需訓練 VGG) 並設為評估模式
        self.model.eval()
        self.model.requires_grad_(False)

        # 2. 設定正規化參數
        # 使用 register_buffer 註冊常數，使其能隨模型 state_dict 儲存並自動移動 device
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        # 3. 處理目標圖片 (Target Image)
        # 確定設備
        if device is None:
            if isinstance(target_img, torch.Tensor):
                device = target_img.device
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.to(device)

        # 確保 target_img 格式正確並移至設備
        target_tensor = to_nchw(target_img).to(device)

        # 預先計算並快取目標圖片的風格特徵
        # 使用 detach() 確保不會對目標圖計算梯度，節省記憶體
        with torch.no_grad():
            self.target_styles = self.calc_styles(target_tensor)

    def calc_styles(self, imgs: torch.Tensor) -> List[torch.Tensor]:
        """
        提取圖片在 VGG 指定層的特徵，並重塑形狀以利 OT Loss 計算。

        Args:
            imgs (torch.Tensor): 輸入圖片 [Batch, 3, H, W]

        Returns:
            List[torch.Tensor]: 各層特徵列表，形狀為 [Batch, Channel, H*W]
        """
        # 正規化圖片 (ImageNet 標準)
        x = (imgs - self.mean) / self.std

        b, c, h, w = x.shape
        # 將原始輸入視為第一層特徵 (選擇性，視原算法邏輯而定)
        features = [x.reshape(b, c, h * w)]

        for i, layer in enumerate(self.model):
            x = layer(x)
            if i in self.style_layers_idx:
                b_feat, c_feat, h_feat, w_feat = x.shape
                # Flatten spatial dimensions: [B, C, H, W] -> [B, C, N]
                features.append(x.reshape(b_feat, c_feat, h_feat * w_feat))

        return features

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        前向傳播：計算輸入圖片與目標圖片特徵分佈之間的 OT Loss。

        Args:
            imgs (torch.Tensor): 生成的圖片或是需優化的圖片。

        Returns:
            torch.Tensor: 加總後的 Loss。
        """
        # 計算目前輸入圖片的特徵
        input_styles = self.calc_styles(imgs)

        # 累加所有指定層的 OT Loss
        loss = torch.tensor(0.0, device=imgs.device)

        # 使用 zip 同時遍歷輸入特徵與預存的目標特徵
        for x, y in zip(input_styles, self.target_styles):
            loss += ot_loss(x, y)

        return loss


def create_vgg_loss(target_img: Union[torch.Tensor, list]) -> VGGLoss:
    """
    工廠函式：建立 VGGLoss 實例 (保留此函式以維持向後相容性)。
    """
    return VGGLoss(target_img)
