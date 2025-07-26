import numpy as np
import random 


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