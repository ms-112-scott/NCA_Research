from torch.utils.data import Dataset
import torch
import numpy as np


class NCA_Dataset(Dataset):
    def __init__(self, BCHW, channel_names, pool_size=(64, 32)):
        """
        Dataset 結構：
        - self.y: 原始資料 (B, C, H, W)
        - self.x_pool: 共用池 (pool_count, pool_channel, H, W)
        - 每次取樣只會從 x_pool 挑出部分 sample，
          但所有 y 皆來自同一批 self.y。
        """
        self.y = BCHW
        self.channel_names = channel_names
        self.pool_size = pool_size

        # 建立共用池（僅 1 個）
        self.x_pool, self.y_pool = self.create_pool()

    def __len__(self):
        # 回傳 dataset 的樣本數（對應 self.y）
        return len(self.x_pool)

    def __getitem__(self, idx):
        """
        每次取樣：
        - y 取對應樣本
        - x 取自共享池（例如循環取）
        """
        x = self.x_pool[idx]
        y = self.y_pool[idx]
        return (
            idx,
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )

    def create_pool(self):
        """
        建立共享的 x_pool 與 y_pool。
        x_pool 只保留特定通道，其餘補 0。
        """
        B, C, H, W = self.y.shape
        pool_count, pool_channel = self.pool_size

        x_pool = np.zeros((pool_count, pool_channel, H, W), dtype=np.float32)
        y_pool = np.zeros((pool_count, C, H, W), dtype=np.float32)

        keep_channels = [
            "coord_y",
            "coord_x",
            "geo_mask",
            "topo",
            "windInitX",
            "windInitY",
        ]

        for i in range(pool_count):
            y_idx = i % B  # 循環使用資料
            y_pool[i] = self.y[y_idx]

            for c, name in enumerate(self.channel_names):
                if c >= pool_channel:
                    break
                x_pool[i, c] = self.y[y_idx, c] if name in keep_channels else 0.0

        return x_pool, y_pool

    def update_x_pool(self, batch_indices, x_new):
        """
        更新共享池中的指定樣本。
        """
        x_new = x_new.detach().cpu().numpy()
        for i, idx in enumerate(batch_indices):
            pool_idx = idx % self.x_pool.shape[0]
            self.x_pool[pool_idx] = x_new[i]
