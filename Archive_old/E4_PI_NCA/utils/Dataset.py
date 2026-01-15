import os
import h5py
import ast
import torch
import torch.nn.functional as F


# ================================================================================
# region load_all_cases_to_BCHW
def load_all_cases_to_BCHW(dataset_dir, target_size=(64, 64), device="cpu"):
    """
    讀取 dataset 下所有城市所有 case，將 wind_field 與 city_base concat，resize，最後合併成 BCHW tensor
    Args:
        dataset_dir: str, dataset 根目錄
        target_size: tuple(H, W), resize 目標大小
        device: str, 'cpu' or 'cuda'
    Returns:
        torch.Tensor, shape [B, C, H, W]
    """
    all_tensors = []
    city_size = []

    city_files = [
        f
        for f in os.listdir(dataset_dir)
        if f.endswith(".h5") and "GlobalMeta" not in f
    ]

    for city_file in city_files:
        city_path = os.path.join(dataset_dir, city_file)
        with h5py.File(city_path, "r") as f:
            city_base = torch.from_numpy(f["city_base"][()]).float()  # C,H,W

            for case_name in f["cases"]:
                # 取得該 case 的 group
                case_grp = f["cases"][case_name]

                # 讀取 wind_field (Dataset)
                wind_field = torch.from_numpy(
                    case_grp["wind_field"][()]
                ).float()  # C,H,W

                # --- 修正部分 ---
                # 2. 讀取 'meta' 屬性 (它是一個字串)
                meta_str = case_grp.attrs["meta"]

                # 3. 將字串解析回字典
                meta_dict = ast.literal_eval(meta_str)

                # 4. 從字典中取出 cityHW
                cityHW = meta_dict["cityHW"]

                # concat city_base + wind_field, shape: C,H,W
                combined = torch.cat([city_base, wind_field], dim=0)

                # add batch dim
                combined = combined.unsqueeze(0)  # 1,C,H,W

                # resize using torch.interpolate
                resized = F.interpolate(
                    combined, size=target_size, mode="bilinear", align_corners=False
                )

                all_tensors.append(resized.squeeze(0))  # C,H,W
                city_size.append(cityHW)

    # stack into B,C,H,W
    all_tensors = torch.stack(all_tensors, dim=0).to(device)
    return all_tensors, city_size


# ================================================================================
# region create_pool
def create_pool(BCHW, city_size, pool_size=512, channel_c=24):
    """
    建立 pool:
    - y: repeat BCHW 到 pool_size，channel 補0到 channel_c
    - x: copy y，但 channel 6 全部置0

    Args:
        BCHW: input tensor [B, C_in, H, W]
        city_size: city原始HW
        pool_size: int, pool 總大小
        channel_c: int, 最終 channel 數量
    Returns:
        x_pool, y_pool: [pool_size, channel_c, H, W]
    """
    B, C_in, H, W = BCHW.shape

    # 計算 repeat 次數
    repeat_times = (pool_size + B - 1) // B  # ceil(pool_size / B)

    # repeat tensor
    y = BCHW.repeat(repeat_times, 1, 1, 1)[:pool_size]  # trim 多餘部分
    city_size = (city_size * repeat_times)[:pool_size]

    # channel 補0到 channel_c
    if C_in < channel_c:
        pad = torch.zeros(
            (y.shape[0], channel_c - C_in, H, W), dtype=y.dtype, device=y.device
        )
        y = torch.cat([y, pad], dim=1)

    # x = y copy
    x = y.clone()
    # channel index 6 全部置0 (第7個通道)
    if channel_c > 6:
        x[:, 6:] = 0.0

    return x, y, city_size


# ================================================================================
# region dataloader

# class MultiCityWindDataset(Dataset):
#     def __init__(self, h5_folder, enable_ageing=True, max_age=100):
#         self.h5_folder = Path(h5_folder)
#         self.enable_ageing = enable_ageing
#         self.max_age = max_age

#         self.city_base = {}  # city_name -> fixed base channels
#         self.samples = []    # list of dicts
#         self.pools = []      # list of dicts, 1:1 對應 samples

#         # 讀取 HDF5
#         h5_files = [f for f in self.h5_folder.glob("*.h5") if f.name != "GlobalMetaData.h5"]
#         for h5_file in h5_files:
#             city_name = h5_file.stem
#             with h5py.File(h5_file, "r") as f:
#                 self.city_base[city_name] = f["city_base"][()]  # np array

#                 for case_key in f["cases"]:
#                     wf = f["cases"][case_key]["wind_field"][()]
#                     meta_str = f["cases"][case_key].attrs["meta"]
#                     try:
#                         meta = eval(meta_str)
#                     except:
#                         meta = {"raw": meta_str}

#                     # 儲存 sample
#                     sample_entry = {
#                         "city": city_name,
#                         "case": case_key,
#                         "meta": meta
#                     }
#                     self.samples.append(sample_entry)

#                     # 對應 pool (每個 sample 一個獨立 pool)
#                     pool_entry = {
#                         "x": wf.copy(),  # 可被更新
#                         "age": 0
#                     }
#                     self.pools.append(pool_entry)

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         sample = self.samples[idx]
#         pool = self.pools[idx]

#         city_base = torch.tensor(self.city_base[sample["city"]], dtype=torch.float32)
#         x = torch.tensor(pool["x"], dtype=torch.float32)  # 從 pool 取
#         y = x.clone()                                     # placeholder
#         meta = sample["meta"]

#         if self.enable_ageing:
#             pool["age"] += 1
#             if pool["age"] > self.max_age:
#                 pool["age"] = 0
#                 pool["x"] = city_base.clone().numpy()  # reset

#         return city_base, x, y, meta

#     def update_pool(self, idx, new_x):
#         """訓練後把 output 更新回 pool"""
#         self.pools[idx]["x"] = new_x.cpu().numpy()
#         self.pools[idx]["age"] = 0


# from torch.utils.data import DataLoader
# # 初始化整個實驗環境
# import sys
# sys.path.append("C:/Users/GAI/Desktop/Scott/NCA_Research")
# from core_utils.plotting import *
# from E4_PI_NCA.utils.helper import *


# names = [
#     "coord_y",
#     "coord_x",
#     "geo_mask",
#     "topo",
#     "windInitX",
#     "windInitY",
#     "uped",
#     "vped",
#     "Uped",
#     "TKEped",
#     "Tuwped",
# ]

# # 1️⃣ 指定資料夾
# h5_folder = Path("../dataset_h5")

# # 2️⃣ 建立 Dataset
# dataset = MultiCityWindDataset(h5_folder=h5_folder, enable_ageing=True, max_age=100)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
# count=0
# for city_base, x, y, meta in dataloader:
#     count+=1
#     # 移到 GPU
#     city_base = city_base.to(device)
#     print(city_base.shape)
#     x = x.to(device)
#     y = y.to(device)
#     print(x.shape, y.shape)
#     # plt_HWC_split_channels(to_HWC(city_base[0]), channel_names=names[:4])
#     # plt_HWC_split_channels(to_HWC(x[0]), channel_names=names[-7:])

#     # meta 如果包含 tensor，也要轉 GPU
#     # meta = {k: v.to(device) if torch.is_tensor(v) else v for k,v in meta.items()}

#     # 可以送進模型
#     # output = model(x)
#     if count>10:
#         break
