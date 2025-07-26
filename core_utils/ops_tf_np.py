import tensorflow as tf
import numpy as np
import random 



def to_rgba(x):
  return x[..., :4]

def to_alpha(x):
  return tf.clip_by_value(x[..., 3:4], 0.0, 1.0)

def to_rgb(x):
  # assume rgb premultiplied by alpha
  rgb, a = x[..., :3], to_alpha(x)
  return 1.0-a+rgb


#=====================================================================================================================================
#=====================================================================================================================================
# region crop_and_resize
def crop_and_resize(inputx, target_size, crop_size=2):
    if isinstance(inputx, np.ndarray):
        x = tf.convert_to_tensor(inputx, dtype=tf.float32)
    # 假設 x 是 4D tensor: (batch, height, width, channels)
    x_cropped = inputx[:, crop_size:-crop_size, crop_size:-crop_size, :]  # 上下左右各裁剪 2 像素
    x_resized = tf.image.resize(x_cropped, size=(target_size[0],target_size[1]), method='nearest')
    if isinstance(inputx, np.ndarray):
        x_resized = x_resized.numpy().astype(np.float32)
    return x_resized

#=====================================================================================================================================
#=====================================================================================================================================
# region get_random_cfd_slices
def get_random_cfd_slices(dynamic_fields, static_fields, num=1):
    """
    從 CFD 資料中隨機選擇 num 組 (t, z)，並回傳 shape=(num, Y, X, C) 的切片。

    Args:
        dynamic_fields (np.ndarray): CFD 時變資料，shape=(T, Z, Y, X, C_dynamic)
        static_fields (np.ndarray): CFD 靜態資料，shape=(Z, Y, X, C_static)
        num (int): 要選取的切片數量

    Returns:
        np.ndarray: 切片資料，shape = (num, Y, X, C_static + C_dynamic)
        list: 對應的 t index 列表
        list: 對應的 z index 列表
    """
    T, Z, Y, X, C_dyn = dynamic_fields.shape
    _, _, _, C_static = static_fields.shape

    slices = np.zeros((num, Y, X, C_dyn + C_static), dtype=dynamic_fields.dtype)
    t_list = []
    z_list = []

    for i in range(num):
        t = random.randint(0, T - 1)
        z = random.randint(0, Z - 1)
        dyn = dynamic_fields[t, z, :, :, :]     # shape = (Y, X, C_dyn)
        sta = static_fields[z, :, :, :]         # shape = (Y, X, C_static)
        slices[i] = np.concatenate((sta, dyn), axis=-1)

        t_list.append(t)
        z_list.append(z)

    return slices, t_list, z_list

#=====================================================================================================================================
#=====================================================================================================================================
# region get_random_cfd_slices_pair
def get_random_cfd_slices_pair(dynamic_fields, static_fields, slice_count=1, xy_size=None, output_meta=False):
    """
    向量化版本的 CFD 切片生成函數。可用於資料增強與強化學習。
    
    Args:
        dynamic_fields: tf.Tensor, shape=(T, Z, Y, X, C_dyn)
        static_fields: tf.Tensor, shape=(Z, Y, X, C_static)
        slice_count: int
        xy_size: (h, w) or None
        output_meta: bool
    
    Returns:
        t_slices: (N, h, w, C_total)
        tn_slices: (N, h, w, C_total)
        metadata_list: list of dict or tf.Tensor if output_meta=False
    """
    T, Z, Y, X, C_dyn = dynamic_fields.shape
    _, _, _, C_static = static_fields.shape
    h = Y if xy_size is None else xy_size[0]
    w = X if xy_size is None else xy_size[1]

    def single_sample(_):
        # 時間與空間隨機選取
        t = tf.random.uniform([], 0, T - 1, dtype=tf.int32)
        tn = tf.random.uniform([], t + 1, T, dtype=tf.int32)
        z = tf.random.uniform([], 0, Z, dtype=tf.int32)
        sy = tf.constant(0) if h == Y else tf.random.uniform([], 0, Y - h + 1, dtype=tf.int32)
        sx = tf.constant(0) if w == X else tf.random.uniform([], 0, X - w + 1, dtype=tf.int32)

        # 擷取資料
        dyn_t = dynamic_fields[t, z, sy:sy+h, sx:sx+w, :]
        dyn_tn = dynamic_fields[tn, z, sy:sy+h, sx:sx+w, :]
        sta = static_fields[z, sy:sy+h, sx:sx+w, :]

        t_slice = tf.concat([sta, dyn_t], axis=-1)
        tn_slice = tf.concat([sta, dyn_tn], axis=-1)

        if output_meta:
            meta = tf.stack([t, tn, z, sy, sx])
        else:
            meta = tn - t

        return t_slice, tn_slice, meta

    # 建立一個 slice_count 長度的 dummy tensor，讓 map_fn 遍歷
    dummy = tf.range(slice_count)
    t_slices, tn_slices, metas = tf.map_fn(single_sample, dummy, dtype=(tf.float32, tf.float32, tf.int32 if not output_meta else tf.int32))

    if output_meta:
        # 若需要 metadata，將 tf.Tensor 轉為 Python list of dict（較慢，但必要）
        metas = tf.transpose(metas)  # shape (5, N) → 每一列是一種 index
        metadata_list = [
            {
                "t": int(metas[0, i].numpy()),
                "tn": int(metas[1, i].numpy()),
                "sz": int(metas[2, i].numpy()),
                "sy": int(metas[3, i].numpy()),
                "sx": int(metas[4, i].numpy())
            }
            for i in range(slice_count)
        ]
        return t_slices, tn_slices, metadata_list
    else:
        return t_slices, tn_slices, metas  # (N,) tf.Tensor

