import torch
import torch.nn.functional as F

# ============================================================
# region 子損失函式
# ============================================================


def continuity_loss(u: torch.Tensor, v: torch.Tensor, device: str) -> torch.Tensor:
    """
    質量守恆損失 (Continuity Equation Residual)

    對應方程：
        ∂u/∂x + ∂v/∂y ≈ 0

    使用 Sobel 卷積近似偏導數後，計算 MSE 殘差。

    Parameters
    ----------
    u : torch.Tensor
        x方向速度場，shape = (B, H, W)
    v : torch.Tensor
        y方向速度場，shape = (B, H, W)
    device : str
        計算裝置 ('cuda' 或 'cpu')

    Returns
    -------
    torch.Tensor
        單一標量損失 (float32)
    """
    # Sobel 核 (使用 float32)
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device
    )
    sobel_y = sobel_x.T

    # Conv2D kernel shape: (out_ch, in_ch, H, W)
    kernel_dx = sobel_x.view(1, 1, 3, 3)
    kernel_dy = sobel_y.view(1, 1, 3, 3)

    # Padding 處理邊界
    u_pad = F.pad(u.unsqueeze(1), (1, 1, 1, 1), mode="replicate")
    v_pad = F.pad(v.unsqueeze(1), (1, 1, 1, 1), mode="replicate")

    du_dx = F.conv2d(u_pad, kernel_dx)
    dv_dy = F.conv2d(v_pad, kernel_dy)

    # ∇·u = du/dx + dv/dy
    divergence = du_dx + dv_dy

    return F.mse_loss(divergence, torch.zeros_like(divergence, dtype=torch.float32))


def momentum_loss(
    x: torch.Tensor,
    device: str,
    rho: float = 1.0,
    mu: float = 1e-3,
    dx: float = 1.0,
    dy: float = 1.0,
) -> torch.Tensor:
    """
    動量守恆殘差 (包含湍流動量通量項)

    對應簡化方程：
        R_u = ρ(u ∂u/∂x + v ∂u/∂y) - μ∇²u - ∂Tuw/∂y
        R_v = ρ(u ∂v/∂x + v ∂v/∂y) - μ∇²v - ∂Tuw/∂x

    Parameters
    ----------
    x : torch.Tensor
        全通道張量，需包含以下 channel：
        ['coord_y', 'coord_x', 'geo_mask', 'topo',
         'windInitX', 'windInitY', 'uped', 'vped', 'Uped', 'TKEped', 'Tuwped']
    device : str
        運算裝置
    rho : float, optional
        流體密度，預設 1.0
    mu : float, optional
        動態黏度，預設 1e-3
    dx, dy : float, optional
        空間解析度，預設 1.0

    Returns
    -------
    torch.Tensor
        動量守恆損失 (float32)
    """
    u = x[:, 6].float()
    v = x[:, 7].float()
    Tuw = x[:, 10].float()

    # 卷積核 (Sobel + Laplacian)
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device
    ) / (8.0 * dx)
    sobel_y = sobel_x.T / (8.0 * dy)
    laplacian = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=device
    ) / (dx * dy)

    kernel_dx = sobel_x.view(1, 1, 3, 3)
    kernel_dy = sobel_y.view(1, 1, 3, 3)
    kernel_lap = laplacian.view(1, 1, 3, 3)

    def pad(t):  # 統一 padding 處理
        return F.pad(t.unsqueeze(1), (1, 1, 1, 1), mode="replicate")

    u_pad, v_pad, Tuw_pad = pad(u), pad(v), pad(Tuw)

    # 一階偏導
    du_dx = F.conv2d(u_pad, kernel_dx)
    du_dy = F.conv2d(u_pad, kernel_dy)
    dv_dx = F.conv2d(v_pad, kernel_dx)
    dv_dy = F.conv2d(v_pad, kernel_dy)

    # 二階導 (Laplace)
    lap_u = F.conv2d(u_pad, kernel_lap)
    lap_v = F.conv2d(v_pad, kernel_lap)

    # 湍流動量通量偏導
    dTuw_dx = F.conv2d(Tuw_pad, kernel_dx)
    dTuw_dy = F.conv2d(Tuw_pad, kernel_dy)

    # 動量方程殘差
    R_u = (
        rho * (u * du_dx.squeeze(1) + v * du_dy.squeeze(1))
        - mu * lap_u.squeeze(1)
        - dTuw_dy.squeeze(1)
    )
    R_v = (
        rho * (u * dv_dx.squeeze(1) + v * dv_dy.squeeze(1))
        - mu * lap_v.squeeze(1)
        - dTuw_dx.squeeze(1)
    )

    # 損失
    loss_u = (R_u**2).mean()
    loss_v = (R_v**2).mean()
    return 0.5 * (loss_u + loss_v)


def data_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    資料對應損失 (Data Supervision)

    對流場主要通道計算 MSE。

    Parameters
    ----------
    x, y : torch.Tensor
        模型預測與目標張量，shape = (B, C, H, W)

    Returns
    -------
    torch.Tensor
        MSE 損失
    """
    return ((x[:, 6:11].float() - y[:, 6:11].float()) ** 2).mean()


def Uvel_loss(x: torch.Tensor) -> torch.Tensor:
    """
    檢查 Uped 與 (u, v) 向量速度一致性的損失

    Parameters
    ----------
    x : torch.Tensor
        輸入張量 (包含 uped, vped, Uped channel)

    Returns
    -------
    torch.Tensor
        一致性損失 (float32)
    """
    u, v, Uped = x[:, 6:7].float(), x[:, 7:8].float(), x[:, 8:9].float()
    Uped_cal = torch.sqrt(u**2 + v**2 + 1e-8)
    diff = torch.abs(Uped - Uped_cal)
    return (diff**2).mean()


# ============================================================
# region 主 PINN 損失整合
# ============================================================
def pinn_loss(
    config: dict,
    x: torch.Tensor,
    y: torch.Tensor,
    prev_x: torch.Tensor = None,
    prev_delta=None,
) -> dict:
    """
    主 PINN Loss 組合函式 (隨機權重 0~4)
    - 各項權重為 0~4 的隨機浮點數
    - 若權重為 0 則跳過該損失計算
    - 返回各子損失字典形式，可直接累加作為總 loss
    """
    device = config.get("system", {}).get("device", "cpu")
    channel_names = config.get("system", {}).get("channel_names", [])

    # 取得速度欄位索引
    u_idx = channel_names.index("uped")
    v_idx = channel_names.index("vped")
    u_field = x[:, u_idx]
    v_field = x[:, v_idx]

    loss_dict = {}

    # ===== 隨機權重 =====
    λ_data = config.get("loss_weights", {}).get("data", 0.0)
    λ_uvel = config.get("loss_weights", {}).get("data", 0.0)
    λ_phys = config.get("loss_weights", {}).get("data", 0.0)

    # ===== 計算子損失 (僅在權重大於0時) =====
    if λ_data > 0.0:
        L_data = data_loss(x, y)
        loss_dict["L_data"] = λ_data * L_data

    if λ_uvel > 0.0:
        L_uvel = Uvel_loss(x)
        loss_dict["L_uvel"] = λ_uvel * L_uvel

    if λ_phys > 0.0:
        L_cont = continuity_loss(u_field, v_field, device)
        L_mom = momentum_loss(x, device)
        loss_dict["L_cont"] = λ_phys * L_cont
        loss_dict["L_mom"] = λ_phys * L_mom

    return loss_dict
