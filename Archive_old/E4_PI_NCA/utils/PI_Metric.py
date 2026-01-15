from collections import OrderedDict
import torch
from torch import nn
import piq
from typing import List, Tuple

# 索引 (Indices)
IDX_U_COMPONENTS = 6
IDX_V_COMPONENTS = 7
IDX_U_MAGNITUDE = 8
IDX_TKE = 9
IDX_T_UW = 10
IDX_DWDZ = 11  # dW/dz (z-gradient), 來自 "散度=0" 的推斷


# ===============================================
# region helper func
def ssim_per_channel(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    ssim_channels: list = [6, 7, 8, 9, 10],
) -> torch.Tensor:
    """
    [V2 - 修正版]
    對指定通道執行 SSIM 計算。
    使用 Target 的 min/max 作為唯一的正規化標準，以防止 pred 的異常值污染 SSIM 結果。
    """

    if ssim_channels is None:
        ssim_channels = list(range(pred.size(1)))

    ssim_values = []

    for c in ssim_channels:
        pred_c = pred[:, c : c + 1, :, :]
        target_c = target[:, c : c + 1, :, :]

        # --- 修正開始 ---
        # 1. 僅使用 Target 來定義數據範圍
        #    我們假設 target 的 min/max 是 "真實" 的範圍
        min_val = target_c.min()
        max_val = target_c.max()
        range_val = max_val - min_val
        # --- 修正結束 ---

        if range_val > 1e-6:
            # 2. 使用 Target 的範圍來正規化兩者
            #    我們也 .clamp() pred，防止其超出合理範圍
            pred_normalized = (
                (pred_c.clamp(min=min_val, max=max_val) - min_val) / range_val
            ) * data_range
            target_normalized = ((target_c - min_val) / range_val) * data_range

            # 3. 計算 SSIM
            ssim_c = piq.ssim(
                pred_normalized,
                target_normalized,
                data_range=data_range,
                reduction="mean",
            )
            ssim_values.append(ssim_c)
        else:
            # 如果 Target 範圍為零 (平坦)
            # 檢查 Pred 是否也平坦且相等
            if torch.allclose(pred_c, target_c, atol=1e-6):
                ssim_values.append(torch.tensor(1.0, device=pred.device))
            else:
                ssim_values.append(torch.tensor(0.0, device=pred.device))

    if not ssim_values:
        return torch.tensor(0.0, device=pred.device)

    return torch.mean(torch.stack(ssim_values))


# endregion
# ===============================================
# region ScaledPhysicsMetrics class
class ScaledPhysicsMetrics(nn.Module):
    """
    計算一組縮放到 [0, 1] 範圍的數據和物理 Metric，並計算加權綜合性能分數。

    V3 (L_u_bound) 更新:
    1.  [已保留] SSIM 函數使用 V2 Target-based normalization。
    2.  [已保留] 保留 "physical_continuity_score"。
    3.  [修改] 將 'l_u_score' 和 'rec_u_score' (基於 MAE)
        替換為 'physical_l_u_bound_score' (基於 relu)，
        使其與 L_u_bound 損失函數一致。
    4.  [修改] 權重從 7 個指標 (1/7) 重新平衡為 6 個指標 (1/6)。
    """

    def __init__(
        self,
        full_target_data: torch.Tensor,
        dydx_full: List[Tuple[float, float]],  # <-- 保留
        alpha: float = 2.25,
    ):
        super().__init__()
        self.alpha = alpha

        if len(dydx_full) != full_target_data.shape[0]:
            raise ValueError(
                f"dydx_full (len {len(dydx_full)}) 必須與 full_target_data (Batch={full_target_data.shape[0]}) 的長度匹配"
            )

        # --- I. Baseline: MAE Max ---
        target_subset = full_target_data[:, IDX_U_COMPONENTS : IDX_T_UW + 1]
        worst_pred = torch.zeros_like(target_subset)
        self.mae_max = torch.abs(worst_pred - target_subset).mean().item()

        # --- II. L_u_bound Baseline (基於 relu(U_recons - U_mag)) ---
        # [修改] 替換 L_U 和 Rec. U Baseline
        u_comp_t = full_target_data[:, IDX_U_COMPONENTS]
        v_comp_t = full_target_data[:, IDX_V_COMPONENTS]
        U_mag_t = full_target_data[:, IDX_U_MAGNITUDE]
        U_recons_t = torch.sqrt(u_comp_t**2 + v_comp_t**2)

        # Perfect = Target 自身的 L_u_bound 殘差 (應接近 0)
        l_u_bound_perfect_raw = torch.relu(U_recons_t - U_mag_t).mean().item()
        self.l_u_bound_perfect = l_u_bound_perfect_raw

        # Worst = 一個合理的 "最差" 尺度 (例如 U_recons_t 的均值)
        # 這代表 U_mag 完全失敗 (為 0)，而 U_recons 很大
        l_u_bound_worst_raw = torch.abs(U_recons_t).mean().item()
        self.l_u_bound_worst = l_u_bound_worst_raw

        # 確保 worst > perfect
        if self.l_u_bound_worst <= self.l_u_bound_perfect:
            self.l_u_bound_worst = self.l_u_bound_perfect + 1.0

        # --- III. P_flux Baseline ---
        TKE_t = full_target_data[:, IDX_TKE].clamp(min=1e-8)
        T_uw_t = full_target_data[:, IDX_T_UW]
        flux_bound_error_t = torch.abs(T_uw_t) - (self.alpha * TKE_t)
        P_flux_perfect = (flux_bound_error_t > 0).float().mean().item()
        self.p_flux_perfect = P_flux_perfect

        # --- IV. L_continuity Baseline (保留) ---
        (
            residual_target,
            horiz_div_target,
        ) = self._calculate_continuity_residual(
            full_target_data, dydx_full, return_div=True
        )
        self.continuity_mae_perfect = torch.abs(residual_target).mean().item()
        self.continuity_mae_worst = torch.abs(horiz_div_target).mean().item()
        if self.continuity_mae_worst <= self.continuity_mae_perfect:
            self.continuity_mae_worst = self.continuity_mae_perfect + 1.0

        # --- V. Composite Score 權重定義 (總和 1.0) ---
        # [修改] 從 7 個指標 (1/7) 調整為 6 個指標 (1/6)
        self.weights = OrderedDict(
            {
                "data_mae_score": 1 / 6,
                "structural_ssim_score": 1 / 6,
                "physical_p_k_neg_score": 1 / 6,
                "physical_p_flux_score": 1 / 6,
                "physical_l_u_bound_score": 1 / 6,  # <-- 修改名稱
                "physical_continuity_score": 1 / 6,
            }
        )

        print("--- Metric Baseline Calculated (Worst Score = 0) ---")
        print(f"MAE Max Baseline: {self.mae_max:.4e}")
        # [修改] 更新 print 敘述
        print(
            f"L_u_bound Worst Baseline: {self.l_u_bound_worst:.4e} (Perfect={self.l_u_bound_perfect:.4e})"
        )
        print(f"P_flux Perfect Baseline: {self.p_flux_perfect:.4e} (Target 違規比例)")
        print(
            f"Continuity Worst Baseline: {self.continuity_mae_worst:.4e} (Perfect={self.continuity_mae_perfect:.4e})"
        )

    def _calculate_continuity_residual(
        self,
        tensor: torch.Tensor,
        dydx: List[Tuple[float, float]],
        return_div: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        計算連續方程式殘差 R = dU/dx + dV/dy + dW/dz。(程式碼不變)
        """

        # 1. 提取場
        u_pred = tensor[:, IDX_U_COMPONENTS]  # (B, H, W)
        v_pred = tensor[:, IDX_V_COMPONENTS]  # (B, H, W)
        dW_dz_pred = tensor[:, IDX_DWDZ]  # (B, H, W)

        B, H_nn, W_nn = u_pred.shape
        device = tensor.device
        dtype = tensor.dtype

        # 2. 獲取物理間距 (vectorized)
        H_orig_batch = torch.tensor(
            [item[0] for item in dydx], device=device, dtype=dtype
        )
        W_orig_batch = torch.tensor(
            [item[1] for item in dydx], device=device, dtype=dtype
        )
        dy_spacing = H_orig_batch / H_nn
        dx_spacing = W_orig_batch / W_nn
        dy_b = dy_spacing.view(B, 1, 1)
        dx_b = dx_spacing.view(B, 1, 1)

        # 3. 計算像素梯度
        grad_u_y_px, grad_u_x_px = torch.gradient(u_pred, dim=(-2, -1))
        grad_v_y_px, grad_v_x_px = torch.gradient(v_pred, dim=(-2, -1))

        # 4. 轉換為物理梯度
        dU_dx_pred = grad_u_x_px / dx_b
        dV_dy_pred = grad_v_y_px / dy_b

        # 5. 計算物理殘差
        physics_residual = dU_dx_pred + dV_dy_pred + dW_dz_pred

        if return_div:
            horizontal_divergence = dU_dx_pred + dV_dy_pred
            return physics_residual, horizontal_divergence

        return physics_residual

    def _calculate_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        dydx: List[Tuple[float, float]],  # <-- 保留
    ) -> dict:
        """計算所有原始 (Raw) Metric 值。"""

        # --- 基本 MAE ---
        mae_raw = torch.abs(
            pred[:, IDX_U_COMPONENTS : IDX_T_UW + 1]
            - target[:, IDX_U_COMPONENTS : IDX_T_UW + 1]
        ).mean()

        # --- 結構相似性 ---
        ssim_values = ssim_per_channel(
            pred,
            target,
            ssim_channels=list(range(IDX_U_COMPONENTS, IDX_DWDZ + 1)),
        )
        ssim_val_raw = ssim_values.mean()

        # --- 物理量提取 ---
        TKE_pred = pred[:, IDX_TKE]
        T_uw_pred = pred[:, IDX_T_UW]
        u_pred = pred[:, IDX_U_COMPONENTS]
        v_pred = pred[:, IDX_V_COMPONENTS]
        U_pred = pred[:, IDX_U_MAGNITUDE]

        # --- L_u_bound Raw Metric (基於 relu(U_recons - U_mag)) ---
        # [修改] 替換 l_u_raw / rec_u_mae_raw
        U_reconstructed = torch.sqrt(u_pred**2 + v_pred**2)
        l_u_bound_raw = torch.relu(U_reconstructed - U_pred).mean()

        # --- P_k_neg (TKE 負值比例) ---
        p_k_neg_raw = (TKE_pred < 0).float().mean()

        # --- P_flux_over (Flux 超界比例) ---
        flux_bound_error = torch.abs(T_uw_pred) - (
            self.alpha * (TKE_pred.clamp(min=1e-8))
        )
        p_flux_over_raw = (flux_bound_error > 0).float().mean()

        # --- L_continuity (連續方程式殘差 MAE) (保留) ---
        residual_pred = self._calculate_continuity_residual(pred, dydx)
        continuity_mae_raw = torch.abs(residual_pred).mean()

        # [修改] 更新返回的字典
        return {
            "mae": mae_raw,
            "ssim": ssim_val_raw,
            "l_u_bound": l_u_bound_raw,  # <-- 修改
            "p_k_neg": p_k_neg_raw,
            "p_flux_over": p_flux_over_raw,
            "continuity_mae": continuity_mae_raw,
        }

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        dydx: List[Tuple[float, float]],  # <-- 保留
    ) -> dict:
        """計算所有 Scaled Score 並匯總為 Composite Score。"""

        raw_metrics = self._calculate_metrics(pred, target, dydx)
        scaled_scores = {}

        # Helper: 確保 Baseline/Perfect 值是 Tensor
        device = pred.device

        # [修改] 移除 l_u_p_t, l_u_w_t
        # [修改] 新增 l_u_bound baselines
        l_u_b_p_t = torch.tensor(self.l_u_bound_perfect, device=device)
        l_u_b_w_t = torch.tensor(self.l_u_bound_worst, device=device)

        p_flux_p_t = torch.tensor(self.p_flux_perfect, device=device)
        cont_mae_p_t = torch.tensor(self.continuity_mae_perfect, device=device)
        cont_mae_w_t = torch.tensor(self.continuity_mae_worst, device=device)

        # --- 1. Data and Structural Scores ---
        score_mae = 1.0 - (raw_metrics["mae"] / self.mae_max).clamp(max=1.0)
        scaled_scores["data_mae_score"] = score_mae
        scaled_scores["structural_ssim_score"] = raw_metrics["ssim"]

        # --- 2. Physical Sanity Scores (P_k_neg) ---
        score_p_k_neg = 1.0 - raw_metrics["p_k_neg"]
        scaled_scores["physical_p_k_neg_score"] = score_p_k_neg

        # --- 3. P_flux Score (相對 Min-Max Scaling) ---
        p_flux_max_range = 1.0 - p_flux_p_t  # Worst=1.0
        p_flux_relative = (raw_metrics["p_flux_over"] - p_flux_p_t).clamp(min=0.0)
        score_p_flux = 1.0 - (p_flux_relative / p_flux_max_range.clamp(min=1e-8)).clamp(
            max=1.0
        )
        scaled_scores["physical_p_flux_score"] = score_p_flux

        # --- 4. L_u_bound Score (相對 Min-Max Scaling) ---
        # [修改] 替換 L_U Score 和 Rec. U Score
        l_u_b_max_range = l_u_b_w_t - l_u_b_p_t
        l_u_b_relative = (raw_metrics["l_u_bound"] - l_u_b_p_t).clamp(min=0.0)
        score_l_u_bound = 1.0 - (
            l_u_b_relative / l_u_b_max_range.clamp(min=1e-8)
        ).clamp(max=1.0)
        scaled_scores["physical_l_u_bound_score"] = score_l_u_bound

        # --- 5. Continuity Score (相對 Min-Max Scaling) ---
        # [修改] 重新編號
        cont_mae_max_range = cont_mae_w_t - cont_mae_p_t
        cont_mae_relative = (raw_metrics["continuity_mae"] - cont_mae_p_t).clamp(
            min=0.0
        )
        score_continuity = 1.0 - (
            cont_mae_relative / cont_mae_max_range.clamp(min=1e-8)
        ).clamp(max=1.0)
        scaled_scores["physical_continuity_score"] = score_continuity

        # --- 6. COMPOSITE SCORE (加權求和) ---
        # [修改] 重新編號
        composite_score = torch.tensor(0.0, device=device)

        # self.weights 字典已在 __init__ 中更新
        for key, weight in self.weights.items():
            if key in scaled_scores:
                weighted_score = scaled_scores[key] * weight
                composite_score += weighted_score
            else:
                print(f"Warning: Metric {key} not found in scaled_scores.")

        scaled_scores["COMPOSITE_SCORE"] = composite_score

        # 也返回原始 metrics (便於日誌記錄)
        # return scaled_scores, raw_metrics
        return scaled_scores
