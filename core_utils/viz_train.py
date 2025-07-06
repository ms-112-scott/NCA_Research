import numpy as np
import matplotlib.pylab as pl
from core_utils.utils_image import imshow, tile2d
from core_utils.utils_io import  imwrite
from core_utils.ops_tensor import to_rgb

def viz_pool(pool, step_i):
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
    imwrite(f'train_log/{step_i:04d}_pool.jpg', tiled_pool)

def viz_batch(x0, x, step_i):
    """
    將訓練批次前後狀態左右合併、上下堆疊後輸出並顯示
    """
    vis0 = np.hstack(to_rgb(x0).numpy())
    vis1 = np.hstack(to_rgb(x).numpy())
    vis = np.vstack([vis0, vis1])
    imwrite(f'train_log/batches_{step_i:04d}.jpg', vis)
    print('batch (before/after):')
    imshow(vis)

def viz_loss(loss_log):
    """
    繪製損失函數歷史(log10尺度)
    """
    pl.figure(figsize=(10, 4))
    pl.title('Loss history (log10)')
    pl.plot(np.log10(loss_log), '.', alpha=0.1)
    pl.show()
