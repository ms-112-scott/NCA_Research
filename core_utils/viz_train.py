import numpy as np
import matplotlib.pylab as plt
from core_utils.utils_image import imshow, tile2d
from core_utils.utils_io import  imwrite
from core_utils.ops_tf_np import to_rgb
import os

def viz_pool(pool, step_i, output_path='train_log'):
    """
    將 pool 中前49張圖像拼貼，並加上四邊漸變淡出效果，
    輸出圖片存檔（train_log/%04d_pool.jpg）
    """
    tiled_pool = tile2d(to_rgb(pool.x[:49]))
    fade = np.linspace(1.0, 0.0, 72)
    ones = np.ones(72)
    tiled_pool[:, :72] += (-tiled_pool[:, :72] + ones[None, :, None]) * fade[None, :, None]
    tiled_pool[:, -72:] += (-tiled_pool[:, -72:] + ones[None, :, None]) * fade[None, ::-1, None]
    tiled_pool[:72, :] += (-tiled_pool[:72, :] + ones[:, None, None]) * fade[:, None, None]
    tiled_pool[-72:, :] += (-tiled_pool[-72:, :] + ones[:, None, None]) * fade[::-1, None, None]
    imwrite(f'{output_path}/{step_i:04d}_pool.jpg', tiled_pool)

def viz_batch(batch_list, step_i, output_path='train_log'):
    """
    將訓練批次前後狀態左右合併、上下堆疊後輸出並顯示
    """
    lists = []
    for i in range(len(batch_list)):
        lists.append(np.hstack(batch_list[i][...,:3]))
    vis = np.vstack(lists)
    imwrite(f'{output_path}/batches_{step_i:04d}.jpg', vis)
    print('batch (before/after):')
    imshow(vis)

def moving_average(data, window=50):
    """計算移動平均"""
    return np.convolve(data, np.ones(window) / window, mode='valid')

def viz_loss(loss_log, log_scale=True, window=50, save_path=None):
    """
    繪製 loss log 曲線，加入移動平均與變異區間。

    Args:
        loss_log (list or np.ndarray): 原始 loss 值序列
        log_scale (bool): 是否以 log10 顯示
        window (int): 移動平均視窗大小
        save_path (str or None): 若提供，將圖像儲存為檔案
    """
    loss_log = np.asarray(loss_log)
    if log_scale:
        loss_log = np.log10(loss_log + 1e-8)

    x = np.arange(len(loss_log))
    ma = moving_average(loss_log, window)
    valid_x = x[window - 1:]

    std = np.array([
        loss_log[i - window + 1:i + 1].std() for i in range(window - 1, len(loss_log))
    ])

    plt.figure(figsize=(10, 4))
    plt.title("Loss history (log10)" if log_scale else "Loss history")
    plt.plot(x, loss_log, '.', alpha=0.1, label='raw')
    plt.plot(valid_x, ma, '-', color='blue', label='Moving Avg')
    plt.fill_between(valid_x, ma - std, ma + std, color='blue', alpha=0.2, label='±1 std')
    plt.xlabel("Step")
    plt.ylabel("Log Loss" if log_scale else "Loss")
    plt.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path+"/loss.png", dpi=150)
        print(f"[✔] Loss curve saved to: {save_path}")

    plt.show()