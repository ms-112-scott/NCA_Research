# %% [markdown]
# # imports

# %%
# ==========================
# 1. 標準庫
# ==========================
import sys
import os
import io
import json
import glob
import datetime
import random
from pathlib import Path
from typing import Dict, List, Union, Callable, Optional, Tuple
import inspect

# ==========================
# 2. 第三方套件
# ==========================
import numpy as np
import xarray as xr
import matplotlib.pylab as plt
from tqdm import trange
from IPython.display import clear_output, display, HTML
from scipy.ndimage import generic_filter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as Optimizer
import torch.nn.init as init

from torchsummary import summary

# import cv2  # 如果影片相關再啟用

# ==========================
# 3. 專案設定
# ==========================
project_root = "C:/Users/GAI/Desktop/Scott/NCA_Research"
if project_root not in sys.path:
    sys.path.append(project_root)

# ==========================
# 4. IPython 魔法指令 (Jupyter 專用)
# ==========================
%reload_ext autoreload
%autoreload 2

# ==========================
# 5. 專案自定義函式庫
# ==========================
from core_utils.plotting import (
    plt_HWC_split_channels,
    plt_CFD_channels,
    plt_random_cfd_slice
)


# from core_utils.utils_io import (
#     np2pil,      # numpy → PIL Image
#     imwrite,     # 儲存圖像為檔案
#     imencode,    # 編碼圖像為 byte stream
#     im2url,      # 圖像轉 base64 URL（HTML 顯示用）
#     load_emoji,   # 載入 emoji 圖像
#     load_cfd_npy
# )

# from core_utils.utils_image import (
#     imshow,      # 在 notebook 顯示圖像
#     tile2d,      # 多圖拼接
#     zoom         # 放大圖像
# )

# from core_utils.utils_video import (
#     save_video,  # 批次輸出影片
#     VideoWriter  # 逐幀寫入影片（支援 context manager）
# )

# from core_utils.ops_tf_np import (
#     to_rgb,
#     to_rgba,
#     to_alpha,
#     crop_and_resize,
#     get_random_cfd_slices,
#     get_random_cfd_slices_pair
# )


from core_utils.viz_train import (
    viz_pool,
    viz_batch,
    viz_loss,
    viz_epochs
)


# 6. 實驗項目 utils 函式庫導入
from E4_PI_NCA.utils.helper import (
    to_HWC,
    print_tensor_stats,
    split_cases,
    get_output_path,
    plot_HW3,
    channels_to_rgb,
    print_loss_dict,
    sort_pool_by_mse,
    log_globals
)

clear_output()

# %% [markdown]
# # global params

# %%
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)
print("output path : ", get_output_path())
CHANNELS_NAMES = ["geo_mask", "topo", "uped", "vped", "Uped", "TKEped", "Tuwped"]
torch.set_default_device(DEVICE)

# dataset params
CASE_DATA = np.load("../dataset/all_cases.npz", allow_pickle=True)
FINAL_EPOCH_SIZE = 1024
IMG_SIZE = 64

# model params
CHANNELS = 32

# trainning params
TOTAL_EPOCHS = 2000
TRAIN_BATCH_SIZE = 32
EPOCH_ITEM_REPEAT_NUM = 2
EPOCH_POOL_SIZE = FINAL_EPOCH_SIZE // EPOCH_ITEM_REPEAT_NUM
# final epoch size = EPOCH_POOL_SIZE * EPOCH_ITEM_REPEAT_NUM
REPEAT_NUM_PER_EPOCH = 1
ROLLOUT_MIN = 1
ROLLOUT_MAX = 10

BC_CHANNELS = [0,1]
IC_CHANNELS = [2,3]

EARLYSTOP_PATIENCE = 100
EARLYSTOP_DELTA = 1e-7

# %%
def set_global_seed(seed: int = 42) -> None:
    """
    設定 Python、NumPy、PyTorch 的隨機種子，確保結果可重現。

    Parameters
    ----------
    seed : int, optional
        隨機種子數值, 預設 42
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (單GPU & 多GPU)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 設定 cudnn 為 deterministic，確保卷積結果可重現
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[INFO] Global seed set to {seed}")


set_global_seed(1234)

# %% [markdown]
# # helper functions

# %%
def plt_acc_over_time(
    acc_list: List[np.ndarray],
    title: str = "Accuracy over time",
    ylabel: str = "Metric",
    smooth_window: int = 1,
    log_scale: bool = False
):
    """
    繪製多條 metric 隨時間變化折線圖，只畫線，不畫點。
    越前面的顏色越淡。
    
    Parameters
    ----------
    acc_list : list of np.ndarray
        每條 ndarray 是一個隨時間變化的 metric
    title : str
        圖片標題
    ylabel : str
        y 軸標籤
    smooth_window : int
        平滑的視窗大小 (滑動平均)
    log_scale : bool
        是否使用 log scale
    """
    plt.figure(figsize=(8,5))
    
    n = len(acc_list)
    for i, arr in enumerate(acc_list):
        if isinstance(arr, list):
            arr = np.array(arr)

        # 計算顏色深淺（越後面的越深）
        min_val = 0.0
        max_val =1.0
        alpha = min_val+ max_val-min_val * (i / max(1, n-1))  

        # 原始線
        plt.plot(
            range(1, len(arr)+1), arr,
            color="tab:blue", alpha=alpha,
            label=f"Run {i+1}" if i == n-1 else None
        )

        # 平滑線
        if smooth_window > 1 and len(arr) >= smooth_window:
            smooth_values = np.convolve(arr, np.ones(smooth_window)/smooth_window, mode='valid')
            plt.plot(
                range(1, len(smooth_values)+1), smooth_values,
                color="tab:red", alpha=alpha, linestyle='--'
            )

    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    if log_scale:
        plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.show()

# %%
def process_case_data_to_tensor_list(
    case_data: Union[np.ndarray, np.lib.npyio.NpzFile, list] = CASE_DATA,
    channels_n: int = CHANNELS,
) -> List[torch.Tensor]:
    """
    將整個 case_data 轉成 list of 0-1 torch.Tensor，shape = (channels_n, H, W)
    支援輸入 np.ndarray (B,C,H,W), npzfile 或 list of arrays
    """
    case_list = []

    # 將 npzfile 或單一 array 轉成 list of arrays
    if isinstance(case_data, np.lib.npyio.NpzFile):
        keys = list(case_data.files)
        arrays = [case_data[k] for k in keys]
    elif isinstance(case_data, np.ndarray):
        if case_data.ndim == 4:  # B,C,H,W
            arrays = [case_data[i] for i in range(case_data.shape[0])]
        else:  # C,H,W
            arrays = [case_data]
    elif isinstance(case_data, list):
        arrays = case_data
    else:
        raise ValueError(f"Unsupported case_data type: {type(case_data)}")

    for arr in arrays:
        if isinstance(arr, torch.Tensor):
            arr = arr.cpu().numpy()
        else:
            arr = arr.astype(np.float32)

        C, H, W = arr.shape
        # 補零到目標 channels
        if C < channels_n:
            pad = np.zeros((channels_n - C, H, W), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)

        case_tensor = torch.tensor(arr, dtype=torch.float32, device=DEVICE)
        case_list.append(case_tensor)

    return case_list

# %%
all_cases = process_case_data_to_tensor_list()

TRAIN_CASES, EVAL_CASES, TEST_CASES = split_cases(
    all_cases, train_ratio=0.7, eval_ratio=0.2, test_ratio=0.1
)
print(len(TRAIN_CASES), len(EVAL_CASES), len(TEST_CASES))

# %% [markdown]
# ## create_epoch_pool

# %%
def create_epoch_pool(
    train_case_list: List[torch.Tensor] = TEST_CASES,
    eval_case_list: Optional[List[torch.Tensor]] = EVAL_CASES,
    test_case_list: Optional[List[torch.Tensor]] = TEST_CASES,
    item_pool_repeats: int = EPOCH_ITEM_REPEAT_NUM,
    poolsize: int = EPOCH_POOL_SIZE,
    hw_size: int = IMG_SIZE,
    mode: str = "train",  # 'train', 'eval', 'test'
    eval_full_image_prob: float = 0.5,  # 評估時使用 full image 的機率
) -> torch.Tensor:
    """
    從指定的 case list 隨機生成一個 pool，每個樣本為裁剪的子區域或 full image。

    參數
    ----------
    train_case_list : List[torch.Tensor]
        訓練 case，每個 shape=(channels_n,H,W)
    eval_case_list : Optional[List[torch.Tensor]], default None
        驗證 case，每個 shape=(channels_n,H,W)，mode='eval' 才會使用
    test_case_list : Optional[List[torch.Tensor]], default None
        測試 case，每個 shape=(channels_n,H,W)，mode='test' 才會使用
    poolsize : int, default EPOCH_POOL_SIZE
        pool 中樣本數量
    hw_size : int, default IMG_SIZE
        裁剪區域大小，訓練模式固定使用
    mode : str, default 'train'
        模式：'train', 'eval', 'test'
    eval_full_image_prob : float, default 0.5
        評估模式時使用 full image 的機率

    回傳
    ----------
    torch.Tensor
        shape = (poolsize, channels_n, h_crop, w_crop)
        測試模式直接使用 full image
    """
    # 選擇對應 case list
    if mode == "train":
        case_list = train_case_list
    elif mode == "eval":
        if eval_case_list is None:
            raise ValueError("mode='eval' 需要提供 eval_case_list")
        case_list = eval_case_list
        poolsize = 1
    elif mode == "test":
        if test_case_list is None:
            raise ValueError("mode='test' 需要提供 test_case_list")
        case_list = test_case_list
        poolsize = 1  # 測試模式固定只取 1 個 sample
    else:
        raise ValueError(f"未知 mode='{mode}'，請選擇 'train', 'eval', 'test'")

    B = len(case_list)
    C, H, W = case_list[0].shape
    pool = []
    eval_fullsize_possibilities = np.random.rand()
    eval_size = hw_size if hw_size == H else np.random.randint(hw_size, H)
    # rand = +np.random.randint(-16, 16)
    for _ in range(poolsize):
        # 隨機選一個 case
        b_idx = np.random.randint(0, B)
        arr_tensor = case_list[b_idx]

        # 決定裁剪大小
        if mode == "train":
            hw_crop = hw_size
        elif mode == "eval":
            # 隨機裁剪或使用 full image
            if eval_fullsize_possibilities < eval_full_image_prob:
                hw_crop = H  # full image
            else:
                hw_crop = eval_size
        else:  # test
            hw_crop = H  # full image
        # hw_crop = hw_size
        # 隨機裁剪起始位置
        h_start = 0 if hw_crop == H else np.random.randint(0, H - hw_crop + 1)
        w_start = 0 if hw_crop == W else np.random.randint(0, W - hw_crop + 1)
        sub = arr_tensor[:, h_start : h_start + hw_crop, w_start : w_start + hw_crop]

        pool.append(sub)

    pool_tensor = torch.stack(
        pool, dim=0
    )  # shape = (poolsize, channels_n, hw_crop, hw_crop)

    # 複製 item pool
    if mode == "train":
        pool_tensor = pool_tensor.repeat((item_pool_repeats, 1, 1, 1))

    return pool_tensor

# %% [markdown]
# ## init_X

# %%
def init_X(target: torch.Tensor) -> torch.Tensor:
    """
    根據 target 初始化 X，形狀相同 (CHW 或 BCHW)。

    邏輯：
    - channel 0 當作 mask
    - channel 0~6 = mask
    - channel >= 7 = 0

    Parameters
    ----------
    target : torch.Tensor
        shape = (C, H, W) 或 (B, C, H, W)

    Returns
    -------
    torch.Tensor
        與 target 同 shape 的 tensor
    """
    X = torch.zeros_like(target, dtype=torch.float32)

    # BCHW
    mask = (target[:, 0:4, :, :]).float()
    X[:, 0:4, :, :] = mask 

    return X

# %% [markdown]
# # viz load data

# %%
Y_pool = create_epoch_pool()
X_pool = init_X(Y_pool)
# print("Epoch pool shape:", Y_pool.shape)  # (poolsize, hw_size, hw_size, 6)
print_tensor_stats(Y_pool, "Y Pool")
plt_HWC_split_channels(to_HWC(Y_pool[0,:9,:,:]))


# %% [markdown]
# # define Neural Net

# %% [markdown]
# ## model

# %%
def set_kernels(angle: float = 0.0, device: str = "cpu") -> torch.Tensor:
    """
    建立 NCA 用的基本卷積核，包含 identity, dx, dy (旋轉後)

    Parameters
    ----------
    angle : float, optional
        kernel 旋轉角度, 預設 0
    device : str, optional
        放置張量的裝置, 預設 "cpu"

    Returns
    -------
    torch.Tensor
        shape = (3, 3, 3)，三個 kernel
    """
    ident = torch.tensor([[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]])
    sobel_x = torch.tensor([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]])
    lap = torch.tensor([[1.0,2.0,1.0],[2.0,-12,2.0],[1.0,2.0,1.0]])
    
    kernel = torch.stack([ident, sobel_x, sobel_x.T, lap])

    return kernel

# %%
class CAModel(nn.Module):
    """
    Cellular Automata Model (Conv + perception kernels)
    """
    def __init__(self, channel_n: int = 16, kernel_count: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.channel_n = channel_n
        self.kernel_count = kernel_count
        self.hidden_dim = hidden_dim

        # 建立 rule_block
        self.rule_block = self.build_rule_block(
            in_channels=self.channel_n * self.kernel_count,
            hidden_dim=self.hidden_dim,
            out_channels=self.channel_n,
            num_hidden_layers=2
        )

    def build_rule_block(self, in_channels: int, hidden_dim: int, out_channels: int, num_hidden_layers: int = 3) -> nn.Sequential:
        """
        建立 rule block 的 Conv + Tanh 結構
        """
        layers = [nn.Conv2d(in_channels, hidden_dim, kernel_size=1), nn.Tanh()]
        for _ in range(num_hidden_layers):
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1))
            layers.append(nn.Tanh())
        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False))
        return nn.Sequential(*layers)

    def perchannel_conv(self, x: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
        """
        對每個 channel 做 depthwise convolution
        x: [B, C, H, W]
        filters: [filter_n, Hf, Wf]
        return: [B, C * filter_n, H, W]
        """
        b, ch, h, w = x.shape
        device = x.device
        filters = filters.to(device)
        y = x.reshape(b * ch, 1, h, w)
        y = F.pad(y, [1, 1, 1, 1], mode='circular')
        y = F.conv2d(y, filters[:, None])
        return y.reshape(b, -1, h, w)

    def perception(self, x: torch.Tensor) -> torch.Tensor:
        """
        定義感知 kernels: identity, sobel_x, sobel_y, laplacian
        """
        device = x.device
        ident = torch.tensor([[0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0],
                              [0.0, 0.0, 0.0]], device=device)
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                                [-2.0, 0.0, 2.0],
                                [-1.0, 0.0, 1.0]], device=device)
        lap = torch.tensor([[1.0, 2.0, 1.0],
                            [2.0, -12.0, 2.0],
                            [1.0, 2.0, 1.0]], device=device)
        filters = torch.stack([ident, sobel_x, sobel_x.T, lap])
        return self.perchannel_conv(x, filters)
    

    def forward_pass(self, x: torch.Tensor) -> torch.Tensor:
        """
        單步更新
        """
        y = self.perception(x)
        dx = self.rule_block(y)
        mask = x[:, :4, :, :]  # 前4 channel 保留
        updated = x + dx * x[:, 0:1, :, :]
        return torch.cat([mask, updated[:, 4:, :, :]], dim=1)

    def forward(self, x: torch.Tensor, n_times: int = 1) -> torch.Tensor:
        """
        多步迭代
        """
        for _ in range(n_times):
            x = self.forward_pass(x)
        return x

model = CAModel(channel_n=CHANNELS).to(DEVICE)
summary(model, (CHANNELS, IMG_SIZE, IMG_SIZE), device=DEVICE)


# %% [markdown]
# #### test ca model

# %% [markdown]
# ## EarlyStop

# %%
class EarlyStopper:
    """
    Early stopping helper
    """
    def __init__(self, patience=EARLYSTOP_PATIENCE, min_delta=EARLYSTOP_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def step(self, loss):
        if loss + self.min_delta < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


# %% [markdown]
# ## epoch step

# %% [markdown]
# ### train step

# %%
# ====== 訓練函式 ======
def train_one_epoch(
    model: nn.Module,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    Y_pool: np.array,
    X_pool: np.array,
    train_batch_size: int = TRAIN_BATCH_SIZE,
    repeats_per_epoch: int = REPEAT_NUM_PER_EPOCH,
    rollout_min: int = ROLLOUT_MIN,
    rollout_max: int = ROLLOUT_MAX,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    執行一個 epoch 的訓練迴圈，回傳最後一個 batch 的 loss 與對應資料。

    參數:
        model: nn.Module, 神經網路模型
        optimizer: PyTorch Optimizer
        loss_fn: nn.Module, 損失函數
        train_batch_size: int, 單次訓練的 batch 大小
        item_pool_repeats: int, 將 epoch pool 複製的次數
        repeats_per_epoch: int, 每個 epoch 重複迭代的次數
        rollout_min: int, 最小演化步數
        rollout_max: int, 最大演化步數

    回傳:
        tuple: (loss, Y_batch, X_batch, X_pred)
            - loss: torch.Tensor, 最後一個 batch 的 loss
            - Y_batch: torch.Tensor, 目標 batch
            - X_batch: torch.Tensor, 模型輸入 batch
            - X_pred: torch.Tensor, 模型輸出 batch
    """
    model.train()


    # 隨機挑一個 batch
    idx = torch.randint(0, len(Y_pool) - train_batch_size + 1, (1,))
    Y_batch = Y_pool[idx : idx + train_batch_size].to(DEVICE)
    X_batch = X_pool[idx : idx + train_batch_size].to(DEVICE).clone()
    with torch.no_grad():
        batch_count = len(X_batch)
        X_batch[-batch_count//8:] = init_X(X_batch[-batch_count//8:])



    # 隨機決定演化步數
    rollout_steps = random.randint(rollout_min, rollout_max)

    # 前向傳播
    X_pred = model(X_batch, n_times=rollout_steps)

    #更新pool
    X_pool[idx : idx + train_batch_size] = X_pred

    # 計算 loss 並反向傳播
    loss_dict = loss_fn(X_pred, Y_batch)
    total_loss = sum(loss_dict.values())

    with torch.no_grad():
        total_loss.backward()
        for p in model.parameters():
            p.grad /= p.grad.norm() + 1e-8  # normalize gradients
        optimizer.step()
        
        optimizer.zero_grad()

        batch_dict = {
            "Y": Y_batch,
            "X0": X_batch,
            "X1": X_pred,
            "diff": Y_batch - X_pred,
        }

    return loss_dict, batch_dict

# %% [markdown]
# ### eval step

# %%
def evaluate_one_epoch(
    model: torch.nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]],
    metric_fn: Callable[[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]] = None,
    rollout_steps: int = ROLLOUT_MAX,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], np.ndarray]:
    """
    在測試集上做評估，計算平均 loss 與 metric。
    """
    model.eval()
    total_loss_list = []
    X_batch_list = []
    X_pred_list = []
    metrics_list = []

    # 建立 epoch 的子區域池
    Y_batch = create_epoch_pool(mode="eval").to(DEVICE)  # torch.Size([1, 32, 64, 64])
    X_batch = init_X(Y_batch)

    rollout_steps = 20
    for _ in range(rollout_steps):
        X_batch_list.append(X_batch.clone())
        X_batch = model(X_batch, n_times=1)
        X_pred_list.append(X_batch.clone())

        # 計算 loss
        loss_dict = loss_fn(X_batch, Y_batch)
        total_loss = sum(loss_dict.values())
        total_loss_list.append(total_loss)

        # 計算 metric
        if metric_fn is not None:
            acc_metric = metric_fn(X_batch, Y_batch)
            metrics_list.append(acc_metric)

    x0 = torch.cat(X_batch_list, dim=0)
    x1 = torch.cat(X_pred_list, dim=0)
    y = Y_batch.repeat((rollout_steps, 1, 1, 1))

    metrics_array = np.array(metrics_list)

    batch_dict = {
        "Y": y,
        "X0": x0,
        "X1": x1,
        "diff": y - x1,
    }

    return loss_dict, batch_dict, metrics_array


# %% [markdown]
# ## training loop

# %%
from torch.optim.lr_scheduler import _LRScheduler


# ====== 主訓練流程 ======
def run_training(
    model: nn.Module,
    optimizer: Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]],
    metric_fn: Callable[[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]] = None,
    total_epochs: int = TOTAL_EPOCHS,
    lr_sched: Optional[_LRScheduler] = None,
) -> None:
    """
    執行完整訓練流程

    Parameters
    ----------
    model : torch.nn.Module
        要訓練的模型
    optimizer : torch.optim.Optimizer
        模型優化器
    loss_fn : torch.nn.Module
        損失函式
    total_epochs : int, optional
        訓練總迴圈數, default=100

    Returns
    -------
    None
        訓練過程中直接更新 model 參數
    """

    train_loss_log: list[float] = []
    eval_loss_log: list[float] = []
    eval_loss=0
    eval_metrics: List[np.ndarray] = []
    OutPath = get_output_path(Suffix="loss")

    # Early stopper
    early_stopper = EarlyStopper()

    Y_Pool = create_epoch_pool(mode="train").to(DEVICE)
    X_Pool = init_X(Y_Pool)

    for epoch in trange(total_epochs, desc="Training Epochs"):
        # renew some x pool
        # if epoch % 20 == 0:
        #     with torch.no_grad():
        #         renew_count = 20
        #         Y_Pool_new = create_epoch_pool(mode="train").to(DEVICE)[:renew_count]
        #         X_Pool_new = init_X(Y_Pool_new)[:renew_count]
        #         Y_Pool = Y_Pool_new
        #         X_Pool = X_Pool_new[:renew_count]

        # train step
        train_loss_dict, train_batch_dict = train_one_epoch(
            model, optimizer, loss_fn, Y_Pool, X_Pool
        )
        train_loss = sum(train_loss_dict.values())
        train_loss_log.append(train_loss.item())
        lr_sched.step()

        with torch.no_grad():
            X_Pool, Y_Pool = sort_pool_by_mse(X_Pool, Y_Pool)
                # eval step
            eval_loss_dict, eval_batch_dict, eval_metric = evaluate_one_epoch(
                model, loss_fn, metric_fn
            )

            eval_metrics.append(eval_metric)
            eval_loss = sum(eval_loss_dict.values())
            eval_loss_log.append(eval_loss.item())


        if (epoch + 1) % 50 == 0:
            clear_output(wait=True)
            print_loss_dict(train_loss_dict, eval_loss_dict)
            viz_loss(
                train_loss_log, eval_loss_log, log_scale=True, window=total_epochs // 20
            )
            print("train")
            plt_acc_over_time(eval_metrics, title="L2 Metric", ylabel="L2 Error")
            
            print("train")
            viz_epochs(train_batch_dict)
            print("eval")
            viz_epochs(eval_batch_dict)

            print("viz batch")
            viz_pool(
                to_HWC(train_batch_dict["X0"][:, 4:7, :, :]),
                epoch,
            )
            viz_pool(
                to_HWC(train_batch_dict["X1"][:, 4:7, :, :]),
                epoch,
            )
            print("viz pool")
            viz_pool(
                to_HWC(X_Pool[:, 4:7, :, :]),
                epoch,
            )

        # Early stopping
        if early_stopper.step(eval_loss) and epoch > 5000:
            print(f"Early stopping at epoch {epoch}")

            break

    # end of trainning loop
    viz_loss(
        train_loss_log,
        eval_loss_log,
        log_scale=True,
        window=total_epochs // 20,
        save_path=OutPath,
    )
    # 呼叫函式
    log_globals(globals(), log_dir=OutPath)

# %% [markdown]
# ## loss function

# %%
def divergence_loss(u, v):
    # u, v shape: [B, 1, H, W]  (速度分量)
    du_dx = torch.gradient(u, dim=-1)[0]  # ∂u/∂x
    dv_dy = torch.gradient(v, dim=-2)[0]  # ∂v/∂y
    div = du_dx + dv_dy
    return torch.mean(div**2)  # L2 loss on divergence


def data_mse_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    計算 x 與 y 之間的 MSE loss
    """
    mse = nn.MSELoss()
    x = x[:, 4:9, :, :]  # 只取前 7 個 channel
    y = y[:, 4:9, :, :]
    loss = mse(x, y)

    return loss


def obstacle_loss(x: torch.Tensor) -> torch.Tensor:
    """
    檢查 obstacle cell (mask=0) 內，其他 channel 的值應該趨近於 0

    Args:
        x: Tensor [B, C, H, W], channel 0 = geo_mask (1=air, 0=object)

    Returns:
        torch.Tensor: scalar loss
    """
    mask = x[:, 0:1, ...]  # [B, 1, H, W]
    phys = x[:, 4:7, ...]  # [B, 7, H, W] 取前 7 個物理量通道

    obj_mask = 1.0 - mask  # object cell = 1, air = 0
    masked_phys = phys * obj_mask  # 只保留物體內的值
    loss = torch.mean(masked_phys**2)
    return loss


def fft_loss(x: torch.Tensor, y: torch.Tensor, norm: str = "L2") -> torch.Tensor:
    """
    Compute per-channel FFT loss between x and y.

    Parameters
    ----------
    x, y : torch.Tensor
        Input and target tensors, shape = (B, C, H, W)
    norm : str
        'L1' or 'L2' for difference metric

    Returns
    -------
    torch.Tensor
        Scalar loss
    """
    # FFT: compute 2D FFT per channel
    X_fft = torch.fft.fft2(x[:, 4:7, ...], norm="ortho")  # (B, C, H, W), complex
    Y_fft = torch.fft.fft2(y[:, 4:7, ...], norm="ortho")

    # Compute magnitude difference
    diff = torch.abs(X_fft - Y_fft)  # magnitude difference

    if norm.upper() == "L1":
        loss = diff.mean()
    elif norm.upper() == "L2":
        loss = (diff**2).mean()
    else:
        raise ValueError("norm should be 'L1' or 'L2'")

    return loss


def Uvel_loss(x: torch.Tensor):
    Uped_cal = torch.sqrt(x[:, 4:5, ...] ** 2 + x[:, 5:6, ...] ** 2)  # (b,c,H,W)
    diff = torch.abs(x[:, 6:7, ...] - Uped_cal) 
    loss = (diff**2).mean()
    return loss

def custom_loss(x: torch.Tensor, y: torch.Tensor) -> dict:

    return {
        "mse_loss": data_mse_loss(x, y),
        "obstacle_loss": obstacle_loss(x),
        "Uvel_loss": Uvel_loss(x),
        # "fft_loss": fft_loss(x, y, norm="L2")/ 1e-2,
        # "divergence_loss": divergence_loss(x[:,2:3,...], x[:,3:4,...])
    }

# %% [markdown]
# ## metric function

# %%
def acc_metric(
    pred: torch.Tensor,
    target: torch.Tensor,
    use_pred_mask: bool = True,
    metric_type: str = "L2",
) -> np.array:
    """
    計算模型精度 metric, 支持用 pred 的第0 channel作為 mask.

    Parameters
    ----------
    pred : torch.Tensor
        模型預測, shape = (B, C, H, W)
    target : torch.Tensor
        Ground truth, shape = (B, C, H, W)
    use_pred_mask : bool
        是否用 pred 的第0 channel作為 mask (1 = 計算, 0 = 忽略)
    metric_type : str
        "L1", "L2", "relative"

    Returns
    -------
    torch.Tensor
        scalar metric
    """
    if use_pred_mask:
        mask = pred[:, 0:1, ...]  # shape = (B, 1, H, W)
        mask = (mask > 0.5).float()  # 可選 threshold
        pred = pred * mask
        target = target * mask

    if metric_type.upper() == "L1":
        return torch.mean(torch.abs(pred - target)).detach().cpu().numpy()
    elif metric_type.upper() == "L2":
        return torch.mean((pred - target) ** 2).detach().cpu().numpy()
    elif metric_type.lower() == "relative":
        return torch.mean(torch.abs(pred - target) / (torch.abs(target) + 1e-8)).detach().cpu().numpy()
    else:
        raise ValueError("metric_type should be 'L1', 'L2', or 'relative'")

# %% [markdown]
# # main process

# %%
model = CAModel(channel_n=CHANNELS).to(DEVICE)
optimizer = Optimizer.Adam(model.parameters(), lr=1e-3)
# lr_sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, [300, 600], 0.3)
lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

run_training(model, optimizer, loss_fn=custom_loss, metric_fn=acc_metric, lr_sched=lr_sched)

# %% [markdown]
# # test model

# %%
model2 = CAModel(channel_n=CHANNELS).to(DEVICE)
Y_test = create_epoch_pool(mode="test").to(DEVICE)
X_test = init_X(Y_test)
plt_HWC_split_channels(to_HWC(Y_test[0,:9,...]))
plt_HWC_split_channels(to_HWC(X_test[0,:9,...]))
model2.eval()
with torch.no_grad():
    print("start model rollout")
    for i in range(8):
        if i%1==0:
            X_test = model2(X_test, n_times=1)
            plt_HWC_split_channels(to_HWC(X_test[0,:9,...]))
            acc=acc_metric(X_test,Y_test)
            print(acc)
            
        
    print("start model rollout")
    eval_loss_dict, eval_batch_dict, eval_metric= evaluate_one_epoch(model, custom_loss, acc_metric)
    print("eval_metric",eval_metric)
    plt_acc_over_time(eval_metric, title="L2 Metric", ylabel="L2 Error")






