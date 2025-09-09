import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.display import clear_output

from utils.plotting.plt_cfd import*
from utils.plotting.plt_ndarray import*
from utils.plotting.plt_tensor import*

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------

# region plt_loss
def moving_average(arr, window):
    return np.convolve(arr, np.ones(window)/window, mode='valid')

def moving_std(arr, window):
    return np.array([
        np.std(arr[i-window+1:i+1]) if i >= window-1 else 0
        for i in range(len(arr))
    ])[window-1:]

def plt_loss(train_losses, val_losses=None, smooth_window=50):
    epochs = np.arange(len(train_losses))

    plt.figure(figsize=(10, 5))

    if len(train_losses) >= smooth_window:
        smooth_train = moving_average(train_losses, smooth_window)
        std_train = moving_std(train_losses, smooth_window)
        smooth_epochs = epochs[smooth_window - 1:]

        # plot smooth line
        plt.plot(smooth_epochs, smooth_train, color='blue', label='Train Loss (Smoothed)')
        # plot confidence band
        plt.fill_between(smooth_epochs,
                         smooth_train - std_train,
                         smooth_train + std_train,
                         color='blue', alpha=0.2, label='Train Loss ± std')
    else:
        plt.plot(epochs, train_losses, color='blue', label='Train Loss')

    # val loss
    if val_losses is not None:
        val_epochs = np.arange(len(val_losses))
        if len(val_losses) >= smooth_window:
            smooth_val = moving_average(val_losses, smooth_window)
            std_val = moving_std(val_losses, smooth_window)
            smooth_val_epochs = val_epochs[smooth_window - 1:]

            plt.plot(smooth_val_epochs, smooth_val, color='red', label='Val Loss (Smoothed)')
            plt.fill_between(smooth_val_epochs,
                             smooth_val - std_val,
                             smooth_val + std_val,
                             color='red', alpha=0.2, label='Val Loss ± std')
        else:
            plt.plot(val_epochs, val_losses, color='red', label='Val Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss with Confidence Interval')
    plt.yscale('log')

    # y 軸範圍以 train 為主
    ymin = max(min(train_losses) * 0.9, 1e-8)
    ymax = max(train_losses) * 1.1
    plt.ylim(ymin, ymax)

    plt.legend()
    plt.grid(True)
    plt.show()
# endregion

# region epoch_viz
def epoch_viz(x_batch, x_batch_before, target_batch, x_pool, all_losses,val_loss_history=None, channels=4):
    clear_output(wait=True)
    x_batch = tf.clip_by_value(x_batch, 0.0, 1.0)
    print("batch viz")
    plt_HWC_split_channels(x_batch[0,...,:channels].numpy())
    plt_HWC_split_channels(target_batch[0,...,:channels].numpy())
    plt_batch(x_batch_before)
    plt_batch(x_batch)
    plt_batch(target_batch)
    plt_loss(all_losses,val_loss_history)
    viz_pool(x_pool, rows=8, cols=12)
    pass
# endregion
