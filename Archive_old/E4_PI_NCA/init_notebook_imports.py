# ====================================================
# region imports
# init_env.py - 統一 Notebook 實驗環境初始化
# ====================================================
import sys
import os
import gc
import json
import glob
import random
import datetime
import tqdm
from pathlib import Path
from typing import Dict, List, Union, Callable, Optional, Tuple
import inspect

# ------------------------------------------------------------------------------
# 第三方套件
# ------------------------------------------------------------------------------
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
from torch.utils.data import DataLoader, random_split

from torchsummary import summary

import plotly.io as pio

pio.renderers.default = "vscode"

# ------------------------------------------------------------------------------
# 導入專案函式庫
# ------------------------------------------------------------------------------
from core_utils.plotting import *

from core_utils.viz_train import *

from core_utils.viz_batch_results import *

from E4_PI_NCA.utils.helper import *

from E4_PI_NCA.utils.dataNorm import NormalizeWrapper

from E4_PI_NCA.utils.NCA_dataset import NCA_Dataset

from E4_PI_NCA.utils.PinnLoss_v1 import *


# ------------------------------------------------------------------------------
# region 整理環境資訊
# ------------------------------------------------------------------------------
def show_env_info():
    print(f"📦 PyTorch: {torch.__version__}")
    print(f"📦 Numpy: {np.__version__}")
    print(
        f"📦 Matplotlib: {plt.__version__ if hasattr(plt, '__version__') else 'builtin'}"
    )
    print(f"🧠 Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    clear_output()


# optional: 清理 CUDA 與 cache
def reset_torch_env():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("🧹 Cleared CUDA cache and Python GC")


# ------------------------------------------------------------------------------
# region set_global_seed
# ------------------------------------------------------------------------------
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
    # 在 debug/開發階段可以先設為 deterministic=False, benchmark=True 來加速
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    print(f"[INFO] Global seed set to {seed}")


# ------------------------------------------------------------------------------
# region 預設執行
# ------------------------------------------------------------------------------
clear_output()
print("✅ Environment initialized. Use show_env_info() to check details.")
