gpt 
我在做模型超參數搜索尋找適合的超參數組 我有訓練好多個模型每一個模型資料夾如下，我想要繪製出parrallel plot 對應分析成效

plot 需求
    - 依照最後一軸loss上色 藍到紅 透明度(0.1~1.0)
    - 軸:
    ("model", "channels"),
    ("model", "hidden_dim"),
    ("model", "num_hidden_layers"),
    ("training", "batch_size"),
    ("optim", "lr"),
    loss losses.npz的最後一個數值

- folder 結構如下
ouput/
    train model 1/
        loss/
            loss.png (檔名可能不同)
            losses.npz (檔名可能不同)
        config.json

- config.json如下
{
    "system": {
        "device": "cuda",
        "output_path": "..\\outputs\\E4-4.4_UrbanTales_GNCA_overfit_20251008-215533",
        "channel_names": [
            "geo_mask",
            "topo",
            "uped",
            "vped",
            "Uped",
            "TKEped",
            "Tuwped"
        ]
    },
    "dataset": {
        "dataset_npz_path": "../dataset/all_cases.npz",
        "dataset_size": [
            64,
            64
        ],
        "final_epoch_size": 1024,
        "img_size": 64
    },
    "model": {
        "channels": 32,
        "hidden_dim": 128,
        "kernel_count": 5,
        "num_hidden_layers": 5
    },
    "training": {
        "total_epochs": 1500,
        "batch_size": 16,
        "epoch_item_repeat_num": 2,
        "epoch_pool_size": 512,
        "repeat_num_per_epoch": 1,
        "rollout_min": 1,
        "rollout_max": 8,
        "save_interval": 200
    },
    "earlystop": {
        "patience": 100,
        "delta": 1e-07
    },
    "channels": {
        "bc": [
            0,
            1
        ],
        "ic": [
            2,
            3
        ]
    },
    "loss_weights": {
        "mse": 1.0,
        "obstacle": 3.0,
        "uvel": 2.0
    },
    "optim": {
        "lr": 0.001
    }
}