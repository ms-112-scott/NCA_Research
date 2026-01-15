import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import math


def flatten_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def collect_results(root_dir: str):
    records = []
    for subdir, _, _ in os.walk(root_dir):
        config_path = os.path.join(subdir, "config.json")
        if not os.path.exists(config_path):
            continue

        loss_dir = Path(subdir) / "loss"
        if not loss_dir.exists():
            continue
        npz_files = list(loss_dir.glob("*.npz"))
        if len(npz_files) == 0:
            continue

        with open(config_path, "r") as f:
            config = json.load(f)
        flat_config = flatten_dict(config)

        try:
            losses = np.load(npz_files[0])
            if "losses" in losses:
                loss_values = losses["losses"]
            else:
                loss_values = list(losses.values())[0]
            final_loss = float(loss_values[-1])
        except Exception as e:
            print(f"[Warning] Fail to read loss file in {subdir}: {e}")
            continue

        flat_config["loss"] = final_loss
        flat_config["run_path"] = subdir
        records.append(flat_config)

    df = pd.DataFrame(records)
    print(f"✅ Loaded {len(df)} runs from {root_dir}")
    return df


def plot_parallel_from_folder(
    root_dir: str,
    target_fields: list[tuple[str, str]],
    n_best: int = 10,
    title: str = None,
    label_angle: float = 30,
):
    df = collect_results(root_dir)
    if df.empty:
        print("❌ No valid experiments found.")
        return

    flat_cols = [f"{a}.{b}" for a, b in target_fields] + ["loss"]
    missing = [c for c in flat_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in config: {missing}")

    df_sel = df[flat_cols + ["run_path"]].copy()
    df_sel = df_sel.sort_values(by="loss", ascending=True).reset_index(drop=True)
    df_top = df_sel.head(n_best)

    # --- 從 run_path 取得 model_name ---
    df_top.insert(0, "model_name", df_top["run_path"].apply(lambda p: Path(p).name))

    # 準備 dimensions (隱藏原本 label)
    dimensions = []
    for c in flat_cols:
        vals = pd.to_numeric(df_top[c], errors="coerce")
        vmin, vmax = vals.min(), vals.max()
        tick_vals = np.linspace(vmin, vmax, 5)
        tick_text = [
            f"{v:.3g}" if isinstance(v, (float, np.floating)) else str(v)
            for v in tick_vals
        ]
        dimensions.append(
            dict(
                range=[vmin, vmax],
                label=c,  # 隱藏原本 label
                values=vals,
                tickvals=tick_vals,
                ticktext=tick_text,
            )
        )

    # 設定圖寬
    n_axes = len(flat_cols)
    fig_width = max(800, n_axes * 120)

    # 單一 trace
    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=pd.to_numeric(df_top["loss"]),
                colorscale="RdBu",
                showscale=False,
            ),
            dimensions=dimensions,
        )
    )

    if title is None:
        title = f"Top {n_best} Hyperparameter Runs (Parallel Coordinates)"

    fig.update_layout(
        title=title, width=fig_width, height=600, margin=dict(l=50, r=50, t=100, b=50)
    )

    fig.show()

    # --- 調整欄位順序 & 科學記號 ---
    df_top = df_top.sort_values(by="loss", ascending=True).reset_index(drop=True)
    cols = df_top.columns.tolist()
    cols.remove("loss")
    cols.insert(1, "loss")
    df_top = df_top[cols]

    df_top["loss"] = df_top["loss"].apply(lambda x: f"{x:.4f}")

    return df_top
